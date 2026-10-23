"""Cache before A0. Evaluation labels never enter this API."""
from __future__ import annotations

import time
from .cache import AnswerCache
from ..shared.a0_adapter import measured, sync


class CachedA0:
    def __init__(self, adapter, mode, threshold=0.75):
        self.adapter = adapter
        self.cache = AnswerCache(mode, adapter.identity, adapter.embedder.embedding_dim, threshold)

    def answer(self, query):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        sync()
        started = time.perf_counter()
        timings = {"cache_embedding_seconds": 0.0, "cache_lookup_seconds": 0.0,
                   "a0_seconds": 0.0, "cache_update_seconds": 0.0}
        stage = "identity"
        try:
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
                stage = "cache_update"
                inserted_id, timings["cache_update_seconds"] = measured(self.cache.insert, query, answer, vector)
            result = {"status": "success", "mode": self.cache.mode, "query": query,
                      "answer": answer, "hit": hit, "similarity": score,
                      "threshold": self.cache.threshold if self.cache.mode == "semantic" else None,
                      "candidate_cache_id": candidate.cache_id if candidate else None,
                      "candidate_query": candidate.query if candidate else None,
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
