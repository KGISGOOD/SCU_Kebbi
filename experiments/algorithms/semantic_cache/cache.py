"""Append-only answer cache. FAISS labels are stable IDs, never LRU positions.

Single-threaded and in-memory only. No persistence, eviction or KV cache.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import unicodedata


NORMALIZATION_VERSION = "nfc_strip_collapse_whitespace_v1"


def normalize_question(query):
    """Conservative exact-key policy: preserve case, punctuation and negation."""
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    normalized = " ".join(unicodedata.normalize("NFC", query).split())
    if not normalized:
        raise ValueError("query must not be empty after normalization")
    return normalized


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Entry:
    cache_id: str
    query: str
    answer: str
    namespace: str


class AnswerCache:
    def __init__(self, mode, identity, dimension, threshold=0.75):
        if mode not in {"none", "exact", "semantic"}:
            raise ValueError("mode must be none, exact or semantic")
        if not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("threshold must be finite and within [-1, 1]")
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        self.mode, self.threshold = mode, threshold
        self.namespace = fingerprint({"identity": identity, "mode": mode,
                                      "exact_normalization": NORMALIZATION_VERSION})
        self.identity_hash = fingerprint(identity)
        self.dimension = dimension
        self.entries, self.exact = {}, {}
        self.labels = {}  # explicit FAISS int64 label -> stable cache_id
        self.index = None
        self.valid = True
        if mode == "semantic":
            import faiss
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))

    def check_identity(self, identity):
        if fingerprint(identity) != self.identity_hash:
            raise ValueError("Cache identity changed: create a fresh cache; answers cannot be reused")

    def key(self, query):
        normalized = normalize_question(query)
        return normalized if self.mode == "exact" else query

    def check_consistency(self):
        if not self.valid:
            raise RuntimeError("Cache invalid after index mutation failure; create a fresh cache")
        if (len(self.exact) != len(self.entries) or set(self.exact.values()) != set(self.entries)
                or any(e.namespace != self.namespace or e.cache_id != key
                       or self.exact.get(self.key(e.query)) != key for key, e in self.entries.items())):
            raise RuntimeError("Cache entry/key/namespace mapping inconsistent")
        if self.index is not None:
            import faiss
            labels = [int(v) for v in faiss.vector_to_array(self.index.id_map)]
            if (self.index.ntotal != len(self.entries) or len(labels) != len(set(labels))
                    or set(labels) != set(self.labels)
                    or set(self.labels.values()) != set(self.entries)
                    or len(self.labels) != len(self.entries)):
                raise RuntimeError("FAISS labels and Cache IDs inconsistent")

    def vector(self, value):
        import numpy as np
        array = np.asarray(value, dtype=np.float32).reshape(1, -1)
        if array.shape != (1, self.dimension) or not np.isfinite(array).all():
            raise ValueError("invalid cache embedding shape or non-finite values")
        norm = float(np.linalg.norm(array))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("zero or invalid cache embedding norm")
        return np.ascontiguousarray(array / norm)

    def lookup(self, query, vector=None):
        self.check_consistency()
        key = self.key(query)
        if self.mode == "none":
            return None, None, None
        if self.mode == "exact":
            entry = self.entries.get(self.exact.get(key))
            return entry, None, entry
        if self.index.ntotal == 0:
            return None, None, None
        scores, labels = self.index.search(self.vector(vector), 1)
        label, score = int(labels[0, 0]), float(scores[0, 0])
        if label not in self.labels or not math.isfinite(score):
            raise RuntimeError("Cache index/ID mapping invalid")
        candidate = self.entries[self.labels[label]]
        return (candidate if score >= self.threshold else None), score, candidate

    def insert(self, query, answer, vector=None):
        self.check_consistency()
        key = self.key(query)
        if self.mode == "none":
            return None
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("empty/invalid generation must not be cached")
        if key in self.exact:
            existing = self.entries[self.exact[key]]
            if answer != existing.answer:
                raise ValueError("append-only Cache cannot overwrite an existing key with a different answer")
            return existing.cache_id
        cache_id = fingerprint({"namespace": self.namespace, "key": key})
        entry = Entry(cache_id, query, answer, self.namespace)
        if self.index is not None:
            import numpy as np
            label = len(self.labels)
            array = self.vector(vector)
            try:
                self.index.add_with_ids(array, np.asarray([label], dtype=np.int64))
            except Exception:
                self.valid = False  # never serve an index that may have partially mutated
                raise
            self.labels[label] = cache_id
        self.entries[cache_id] = entry
        self.exact[key] = cache_id
        self.check_consistency()
        return cache_id
