"""Append-only answer cache. FAISS labels are stable IDs, never LRU positions.

Single-threaded and in-memory only. No persistence, eviction or KV cache.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math


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
        self.namespace = fingerprint(identity)
        self.dimension = dimension
        self.entries, self.exact = {}, {}
        self.labels = {}  # explicit FAISS int64 label -> stable cache_id
        self.index = None
        if mode == "semantic":
            import faiss
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))

    def check_identity(self, identity):
        if fingerprint(identity) != self.namespace:
            raise ValueError("Cache identity changed: create a fresh cache; answers cannot be reused")

    def vector(self, value):
        import numpy as np
        array = np.asarray(value, dtype=np.float32).reshape(1, -1)
        if array.shape != (1, self.dimension) or not np.isfinite(array).all():
            raise ValueError("invalid cache embedding shape or non-finite values")
        norm = float(np.linalg.norm(array))
        if norm <= 0:
            raise ValueError("zero cache embedding")
        return np.ascontiguousarray(array / norm)

    def lookup(self, query, vector=None):
        if self.mode == "none":
            return None, None, None
        if self.mode == "exact":
            key = self.exact.get(query)  # deliberately no strip/case/Unicode folding
            entry = self.entries.get(key)
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
        if self.mode == "none":
            return None
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("empty/invalid generation must not be cached")
        if query in self.exact:
            return self.exact[query]
        cache_id = fingerprint({"namespace": self.namespace, "query": query})
        entry = Entry(cache_id, query, answer, self.namespace)
        if self.index is not None:
            import numpy as np
            label = len(self.labels)
            self.index.add_with_ids(self.vector(vector), np.asarray([label], dtype=np.int64))
            self.labels[label] = cache_id
        self.entries[cache_id] = entry
        self.exact[query] = cache_id
        return cache_id
