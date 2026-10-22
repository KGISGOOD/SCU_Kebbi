"""Read-only, 60-way FAQ answering with the same dense protocol as A2 evaluation."""
from __future__ import annotations

import csv
import math
from pathlib import Path

from .common import (CSV_SOURCES, DataError, assert_input_hashes, load_source_faqs,
                     read_checkpoint_provenance, read_jsonl, resolve_path, sha256_file)


def load_answer_mapping(root: Path):
    """Preserve CSV text verbatim; use already-prefixed IDs, never array positions."""
    sources = {row["faq_id"]: row for row in load_source_faqs(root)}
    answers = {}
    for group, _, relative in CSV_SOURCES:
        path = root / relative
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, strict=True)
                for number, row in enumerate(reader, 2):
                    if None in row or any(value is None for value in row.values()):
                        raise DataError("row %d: inconsistent CSV columns" % number)
                    faq_id = row["編號"]
                    if faq_id not in sources or faq_id in answers:
                        raise DataError("duplicate or unknown FAQ ID: %r" % faq_id)
                    if (row["問題內容"] != sources[faq_id]["question"] or
                            group != sources[faq_id]["source_group"]):
                        raise DataError("%s: source question/group changed" % faq_id)
                    answer = row["ai答覆"]
                    if not answer.strip():
                        raise DataError("%s: empty ai答覆" % faq_id)
                    answers[faq_id] = dict(sources[faq_id], answer=answer,
                                          category=row.get("關切維度", row.get("問題分類", "")))
        except (OSError, UnicodeError, csv.Error, KeyError, ValueError) as exc:
            raise DataError("%s: %s" % (path, exc)) from exc
    if set(answers) != set(sources):
        raise DataError("CSV answer mapping must contain all 60 FAQ IDs")
    return answers


def load_answering_data(root: Path, candidates: Path):
    paths = [root / relative for _, _, relative in CSV_SOURCES] + [candidates]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise DataError("missing FAQ input: %s. Copy the existing A2 corpus from the GPU "
                        "project; do not substitute A1 canonical." % ", ".join(missing))
    hashes = {str(path): sha256_file(path) for path in paths}
    answers = load_answer_mapping(root)
    issues = []
    corpus = read_jsonl(candidates, issues)
    if issues:
        raise DataError("invalid candidate JSONL: %r" % issues)
    seen = set()
    for number, row in enumerate(corpus, 1):
        faq_id = row.get("faq_id")
        if not isinstance(faq_id, str) or faq_id not in answers or faq_id in seen:
            raise DataError("candidate line %d: duplicate/unknown FAQ ID %r" % (number, faq_id))
        seen.add(faq_id)
        for key in ("question", "source_group"):
            if row.get(key) != answers[faq_id][key]:
                raise DataError("%s candidate %s differs from original CSV" % (faq_id, key))
    if len(corpus) != 60 or seen != set(answers):
        raise DataError("expected exactly 60 candidates; missing=%s" % sorted(set(answers) - seen))
    assert_input_hashes(hashes)
    return sorted(corpus, key=lambda row: row["faq_id"]), answers


class FAQAnswerer:
    """One selected model encodes the corpus once and every subsequent query.

    No disk vector cache, generated answers, evaluation queries or result exports.
    """

    def __init__(self, *, root: Path, model: str, model_tag: str,
                 candidates: Path, device="auto", batch_size=32, max_length=512,
                 offline=False, allow_unreviewed=False, threshold=None):
        if batch_size < 1 or max_length < 1:
            raise DataError("batch size and max length must be positive")
        if threshold is not None and (not math.isfinite(threshold) or not -1 <= threshold <= 1):
            raise DataError("threshold must be finite and in [-1, 1]")
        if model_tag not in ("A1", "A2-1", "A2-2"):
            raise DataError("unknown model tag: %s" % model_tag)
        self.corpus, self.answers = load_answering_data(root, candidates)
        path = resolve_path(root, model)
        # Only this explicit original model ID is remote/cache-resolvable. A typo
        # in a checkpoint path must never become a Hub ID or a different model.
        original_id = model == "BAAI/bge-m3"
        if not path.is_dir() and (not original_id or model_tag != "A1"):
            raise DataError("local model directory does not exist: %s" % path)
        provenance = read_checkpoint_provenance(path, require_a2=model_tag != "A1")
        if provenance is not None:
            expected = {"a2_1": "A2-1", "a2_2": "A2-2"}[provenance["mode"]]
            if model_tag != expected:
                raise DataError("model tag %s contradicts checkpoint %s" % (model_tag, expected))
            if provenance["provisional"] and not allow_unreviewed:
                raise DataError("provisional checkpoint: explicitly pass --allow-unreviewed for manual use")
        if path.is_dir() and not (path / "config.json").is_file():
            raise DataError("missing model config.json: %s" % path)
        self.model_tag = model_tag
        self.model_name = str(path) if path.is_dir() else model
        self.provisional = bool(allow_unreviewed or (provenance and provenance["provisional"]))
        self.threshold = threshold
        if offline:
            import os
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from experiments.shared.embeddings import BGEM3Embedder

        if device not in ("auto", "cpu", "cuda"):
            raise DataError("device must be auto, cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise DataError("CUDA requested but unavailable")
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.embedder = BGEM3Embedder(model_name=self.model_name, device=self.device,
                                    batch_size=batch_size, normalize_embeddings=True)
        self.embedder.model.max_seq_length = max_length
        self.embedder.model.float()
        self.embedder.model.eval()
        self.ids = tuple(row["faq_id"] for row in self.corpus)
        self._vectors = self._encode([row["question"] for row in self.corpus])

    def _encode(self, texts):
        import numpy as np
        import torch

        with torch.no_grad():
            values = np.asarray(self.embedder.encode(texts), dtype=np.float32)
        if (values.ndim != 2 or values.shape[0] != len(texts) or values.shape[1] == 0
                or not np.isfinite(values).all()):
            raise DataError("invalid/non-finite embedding shape or values")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms <= 0).any():
            raise DataError("zero/non-finite embedding norm")
        return values / norms

    def answer(self, query: str):
        import numpy as np

        if not isinstance(query, str) or not query.strip():
            raise DataError("query must be nonempty text")
        vector = self._encode([query])
        if vector.shape[1] != self._vectors.shape[1]:
            raise DataError("query/candidate embedding dimensions differ")
        scores = (vector @ self._vectors.T)[0]
        if not np.isfinite(scores).all():
            raise DataError("non-finite cosine scores")
        index = min(range(len(self.ids)), key=lambda i: (-float(scores[i]), self.ids[i]))
        faq_id = self.ids[index]
        faq = self.answers.get(faq_id)
        if not faq or not faq["answer"].strip():
            raise DataError("no valid standard answer for %s" % faq_id)
        score = float(scores[index])
        accepted = self.threshold is None or score >= self.threshold
        return {"faq_id": faq_id, "question": faq["question"],
                "answer": faq["answer"] if accepted else None,
                "similarity": score, "accepted": accepted,
                "model": self.model_name, "model_tag": self.model_tag,
                "provisional": self.provisional}
