"""Reuse CachedA0 with a decoder adapter, never share answer caches."""
from copy import deepcopy

from experiments.algorithms.semantic_cache.pipeline import CachedA0
from experiments.algorithms.semantic_cache.threshold_data import judge_reuse


class DecoderAdapter:
    def __init__(self, engine, variant, decoder, repeat, warmup, threshold):
        self.engine, self.decoder = engine, decoder
        self.embedder = engine.adapter.embedder
        self.embed = engine.adapter.embed
        self.repeat, self.warmup = repeat, warmup
        self.identity = {"engine": deepcopy(engine.identity), "variant": variant,
                         "decoder": decoder, "repeat": repeat, "warmup": warmup,
                         "threshold": threshold, "scope": "single_turn"}
        self.case, self.last_miss = None, None

    def answer(self, query):
        case = self.case
        if case is None or query != case["question"]:
            raise ValueError("decoder request/scenario mismatch")
        snapshot, inputs = self.engine.prepare({"scenario_id": case["scenario_id"], "query": query})
        config = self.engine.config
        version, length = {
            "standard": ("SD0", None),
            "draft_1b": ("SD2-1", config["standard_draft_tokens"]),
            "prompt_lookup": (f"SD2-4-k{config['prompt_lookup_tokens']}", config["prompt_lookup_tokens"]),
        }[self.decoder]
        result = self.engine.generate(snapshot, inputs, version, length, self.repeat, self.warmup,
                                      config["seed"] + self.repeat)
        self.last_miss = {"snapshot": snapshot, "generation": result}
        return result["output_text"], {
            "retrieval_seconds": snapshot["retrieval_seconds"],
            "prompt_check_seconds": snapshot["prompt_and_tokenization_seconds"],
            "generation_seconds": result["generation_seconds"],
            "input_tokens": snapshot["input_token_count"], "documents": snapshot["documents"]}


class IntegratedPipeline:
    def __init__(self, engine, variant, decoder, repeat, warmup, threshold):
        if variant not in {"INT0", "INT1", "INT2", "INT3"}:
            raise ValueError("unknown integrated variant")
        self.variant = variant
        self.decoder = decoder if variant in {"INT2", "INT3"} else "standard"
        self.adapter = DecoderAdapter(engine, variant, self.decoder, repeat, warmup, threshold)
        self.pipeline = CachedA0(self.adapter, "semantic" if variant in {"INT1", "INT3"} else "none", threshold)
        self.origins = {}

    def answer(self, case):
        self.adapter.case, self.adapter.last_miss = case, None
        # CachedA0 measures the ACTUAL independent online request, including
        # embedding, lookup, full retrieval, prompt, generation and cache write.
        result = self.pipeline.answer(case["question"], history=case.get("history"))
        cache_id = result["cache_id"]
        origin = self.origins.get(cache_id) if result["hit"] else None
        if result["hit"] and (origin is None or self.adapter.last_miss is not None):
            raise RuntimeError("cache origin missing or Hit unexpectedly executed RAG")
        if not result["hit"] and self.adapter.last_miss is None:
            raise RuntimeError("Miss did not execute full A0+decoder")
        payload = origin["payload"] if origin else self.adapter.last_miss
        snapshot, generation = payload["snapshot"], payload["generation"]
        if result["answer"] != generation["output_text"]:
            raise RuntimeError("stored answer and origin generation do not agree")
        if not result["hit"] and cache_id is not None:
            self.origins.setdefault(cache_id, {"case": deepcopy(case), "payload": deepcopy(payload)})
        # Labels are consulted only AFTER serving the answer, not during lookup.
        judgment = judge_reuse(case, origin["case"] if origin else None)
        return {"scenario_id": case["scenario_id"], "query_order": case["query_order"],
                "question": case["question"], "system_variant": self.variant, "decoder": self.decoder,
                "repeat_index": self.adapter.repeat, "warmup": self.adapter.warmup,
                "status": "success", "error": None, "fallback": False,
                "answer": result["answer"], "cache_hit": result["hit"],
                "similarity": result["similarity"], "cache_id": cache_id,
                "candidate_cache_id": result["candidate_cache_id"],
                "cache_namespace": result["namespace"], "cache_size": result["cache_size"],
                "reused_from_scenario": origin["case"]["scenario_id"] if origin else None,
                "answer_reuse_safety": judgment, "answer_quality_review": None,
                "semantic_safety_verified": False, "review_status": case["review_status"],
                "timings": result["timings"],
                "end_to_end_seconds": result["timings"]["end_to_end_seconds"],
                "end_to_end_kind": "independent online wall-clock request; no shared retrieval",
                "generated_token_count": 0 if result["hit"] else generation["generated_token_count"],
                "input_token_count": 0 if result["hit"] else snapshot["input_token_count"],
                "output_token_ids": generation["output_token_ids"],
                "output_token_sha256": generation["output_token_sha256"],
                "output_token_scope": "stored answer origin" if result["hit"] else "this request",
                "prompt_identity": None if result["hit"] else snapshot["prompt_identity"],
                "retrieval_identity": None if result["hit"] else snapshot["retrieval_identity"],
                "answer_origin_prompt_identity": snapshot["prompt_identity"],
                "target_identity": generation["target_identity"],
                "draft_identity": generation["draft_identity"],
                "cache_threshold": result["threshold"],
                # Do not expose SD's synthetic E2E as an integrated timing.
                "decoding_measurement": None if result["hit"] else {
                    k: v for k, v in generation.items() if k not in {
                        "end_to_end_seconds", "end_to_end_kind", "end_to_end_speedup_sd0",
                        "shared_preparation_seconds", "generation_speedup_sd0", "exact_token_match_sd0"}},
                "prompt_snapshot": None if result["hit"] else snapshot,
                "exact_token_match_int0": None,
                "token_comparison_status": "not_applicable_cache_reuse" if result["hit"] else "pending"}
