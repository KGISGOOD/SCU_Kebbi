"""Cache before A0. Evaluation labels never enter this API."""
from __future__ import annotations

import time
from .cache import AnswerCache, normalize_question, NORMALIZATION_VERSION
from ..shared.a0_adapter import measured, sync


class CachedA0:
    def __init__(self, adapter, mode, threshold=0.75):
        self.adapter = adapter
        self.cache = AnswerCache(mode, adapter.identity, adapter.embedder.embedding_dim, threshold)

    def answer(self, query, *, history=None):
        sync()
        started = time.perf_counter()
        timings = {"cache_embedding_seconds": 0.0, "cache_lookup_seconds": 0.0,
                   "a0_seconds": 0.0, "cache_update_seconds": 0.0,
                   "retrieval_seconds": 0.0, "generation_seconds": 0.0,
                   "prompt_check_seconds": 0.0}
        stage = "request_validation"
        try:
            if history:
                raise ValueError("single-turn Cache only: nonempty history is not supported; do not reuse contextual answers")
            normalized_query, timings["normalization_seconds"] = measured(normalize_question, query)
            stage = "identity"
            _, timings["identity_seconds"] = measured(self.cache.check_identity, self.adapter.identity)
            vector = None
            if self.cache.mode == "semantic":
                stage = "cache_embedding"
                vector, timings["cache_embedding_seconds"] = measured(self.adapter.embed, query)
            stage = "cache_lookup"
            (entry, score, candidate), timings["cache_lookup_seconds"] = measured(self.cache.lookup, query, vector)
            hit = entry is not None
            detail = {}
            inserted_id = None
            if hit:
                answer = entry.answer
            else:
                stage = "a0"
                (answer, detail), timings["a0_seconds"] = measured(self.adapter.answer, query)
                for key in ("retrieval_seconds", "generation_seconds", "prompt_check_seconds"):
                    timings[key] = detail[key]
                stage = "cache_update"
                inserted_id, timings["cache_update_seconds"] = measured(self.cache.insert, query, answer, vector)
            result = {"status": "success", "mode": self.cache.mode, "query": query,
                      "normalized_query": normalized_query, "normalization_version": NORMALIZATION_VERSION,
                      "context_scope": "single_turn", "semantic_safety_verified": False,
                      "answer": answer, "hit": hit, "similarity": score,
                      "threshold": self.cache.threshold if self.cache.mode == "semantic" else None,
                      "candidate_cache_id": candidate.cache_id if candidate else None,
                      "candidate_query": candidate.query if candidate else None,
                      "answer_source_query": entry.query if entry else query,
                      "cache_id": entry.cache_id if entry else inserted_id,
                      "namespace": self.cache.namespace, "cache_size": len(self.cache.entries),
                      "a0_detail": detail, "timings": timings}
        except Exception as error:
            # Caller can preserve failed timings without turning an error into
            # a successful answer or cache entry.
            sync()
            error.cache_failure = {"status": "failed", "stage": stage, "mode": self.cache.mode,
                                   "query": query, "error": str(error), "timings": timings,
                                   "end_to_end_seconds": time.perf_counter() - started}
            raise
        sync()
        timings["end_to_end_seconds"] = time.perf_counter() - started
        return result
