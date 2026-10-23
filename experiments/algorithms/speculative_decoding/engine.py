"""A0 retrieval/prompt preparation once, identical tokens for every mode."""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import time

from experiments.algorithms.shared.a0_adapter import A0Adapter, measured, sync, tree_hash
from experiments.algorithms.semantic_cache.cache import fingerprint
from .compatibility import check_api, check_pair, tokenizer_identity


@contextmanager
def assisted_cache_policy(target, draft):
    """HF 5.16.1 fills None fields from model.generation_config too.

Sanitize both request-local model defaults, not the installed package. HF
creates DynamicCache initially and reuses candidate-generator-owned past
within this request. Restore the exact original objects even on an error.
This engine is sequential; temporary model configuration is not thread-safe.
"""
    target_original, draft_original = target.generation_config, draft.generation_config
    target_local, draft_local = deepcopy(target_original), deepcopy(draft_original)
    for config in (target_local, draft_local):
        config.use_cache = True
        config.cache_implementation = None
        config.cache_config = None
    try:
        target.generation_config, draft.generation_config = target_local, draft_local
        yield
    finally:
        target.generation_config, draft.generation_config = target_original, draft_original


class SDEngine:
    def __init__(self, root, config, with_draft):
        self.api = check_api()
        self.config = config
        # Check the actual local checkpoint configuration, not its folder name.
        for key, role, dimensions in (("model", "3B Target", (3072, 28)),
                                      ("draft_model", "1B Draft", (2048, 16))):
            if key == "draft_model" and not with_draft:
                continue
            path = Path(config[key]).expanduser()
            path = (path if path.is_absolute() else Path(root) / path).resolve()
            data = json.loads((path / "config.json").read_text(encoding="utf-8-sig"))
            if (data.get("model_type") != "llama"
                    or (data.get("hidden_size"), data.get("num_hidden_layers")) != dimensions):
                raise ValueError(f"{role}: config does not match standard Llama 3.2 {dimensions}; check local path")
        self.adapter, self.a0_load_seconds = measured(A0Adapter, root, config)
        self.target, self.tokenizer = self.adapter.llm.model, self.adapter.llm.tokenizer
        self.draft, self.draft_path, self.draft_hash = None, None, None
        self.draft_load_seconds = 0.0
        self.compatibility = {"target_tokenizer": tokenizer_identity(self.tokenizer),
                              "draft": "not loaded for standalone SD0"}
        if self.target.config.model_type != "llama":
            raise ValueError("Target must be the local Llama 3.2 3B Instruct checkpoint")
        self.generation = deepcopy(self.target.generation_config)
        self.generation.update(do_sample=False, num_beams=1, max_new_tokens=config["max_new_tokens"],
                               temperature=1.0, top_p=1.0, top_k=50, repetition_penalty=1.0,
                               pad_token_id=self.tokenizer.pad_token_id,
                               eos_token_id=self.tokenizer.eos_token_id,
                               use_cache=True, cache_implementation="dynamic",
                               return_dict_in_generate=False, output_scores=False, output_logits=False,
                               output_attentions=False, output_hidden_states=False)
        # Leave compiled/custom decoding out of this instrumented V1; both
        # branches use the same explicit policy, preserving forward counters.
        if not hasattr(self.generation, "disable_compile"):
            raise ValueError("reviewed compile-control API missing")
        self.generation.disable_compile = True
        if (getattr(self.generation, "assistant_ensemble_weight", None) is not None
                or getattr(self.generation, "speculation_type", None) is not None):
            raise ValueError("only standard Target-verified speculation is allowed, no ensemble/custom speculation")
        # Reject inherited alternative decoding paths rather than silently
        # replacing them and claiming parity with the existing greedy A0.
        if self.generation.get_generation_mode().value != "greedy_search":
            raise ValueError("Target GenerationConfig does not select standard greedy decoding")
        if with_draft:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            raw = Path(config["draft_model"]).expanduser()
            self.draft_path = (raw if raw.is_absolute() else Path(root) / raw).resolve()
            if not self.draft_path.is_dir() or not (self.draft_path / "config.json").is_file():
                raise ValueError(f"existing LOCAL 1B Draft directory required: {self.draft_path}")
            self.draft_hash = tree_hash(self.draft_path)
            def load():
                tokenizer = AutoTokenizer.from_pretrained(str(self.draft_path), local_files_only=True,
                                                          use_fast=True, trust_remote_code=False)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                draft = AutoModelForCausalLM.from_pretrained(str(self.draft_path), local_files_only=True,
                                                            torch_dtype=self.adapter.llm.torch_dtype,
                                                            trust_remote_code=False)
                draft.to(config["device"]).eval()
                return draft, tokenizer
            (self.draft, draft_tokenizer), self.draft_load_seconds = measured(load)
            self.compatibility = check_pair(self.target, self.tokenizer, self.draft, draft_tokenizer)
            self.draft_defaults = deepcopy(self.draft.generation_config)
        self.identity = {"a0": self.adapter.identity, "api": self.api,
                         "target": self.adapter.source_hashes["model"], "draft": self.draft_hash,
                         "draft_path": str(self.draft_path) if self.draft_path else None,
                         "compatibility": self.compatibility,
                         "effective_generation_config": self.generation.to_dict(),
                         "cache_policy_by_mode": {
                             "SD0": "use_cache=True, explicit dynamic (unchanged)",
                             "assisted": "use_cache=True, selector=None in call and temporary model defaults; HF DynamicCache",
                             "request_scope": "fresh outer call; only HF candidate generator reuses Draft past within call"},
                         "target_parameters": sum(p.numel() for p in self.target.parameters()),
                         "draft_parameters": sum(p.numel() for p in self.draft.parameters()) if self.draft else None,
                         "residency": "target+bge+draft" if self.draft else "target+bge",
                         "baseline_residency_note": "SD0 never calls Draft, but Draft is resident in comparisons",
                         "loading_seconds": {"a0_including_hashing": self.a0_load_seconds,
                                             "draft_model_and_tokenizer": self.draft_load_seconds},
                         "device": config["device"], "precision": str(self.adapter.llm.torch_dtype)}
        self.identity["actual_dtypes"] = {
            "target": str(next(self.target.parameters()).dtype),
            "draft": str(next(self.draft.parameters()).dtype) if self.draft is not None else None}
        self.identity["attention_implementation"] = {
            "target": getattr(self.target.config, "_attn_implementation", None),
            "draft": getattr(self.draft.config, "_attn_implementation", None) if self.draft is not None else None}

    def prepare(self, scenario):
        from experiments.baselines.rag_baseline.prompts import build_prompt, create_context_from_documents
        sync()
        start = time.perf_counter()
        query = scenario["query"]
        self.adapter.validate_query(query)
        (docs, scores), retrieval = measured(self.adapter.pipeline.retrieve, query)
        if len(docs) != 5 or len(scores) != 5:
            raise ValueError("SD comparison requires exactly five existing A0 retrieval documents")
        def construct():
            prompt = build_prompt(create_context_from_documents(docs), query)
            # Same tokenizer call/default special-token policy as A0, except
            # truncation is forbidden and checked before generation.
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
            count = inputs["input_ids"].shape[1]
            limits = [self.target.config.max_position_embeddings]
            if self.draft is not None:
                limits.append(self.draft.config.max_position_embeddings)
            if count > self.config["max_input_tokens"] or count + self.config["max_new_tokens"] > min(limits):
                raise ValueError(f"{scenario['scenario_id']}: full prompt has {count} tokens; no truncation allowed")
            return prompt, {k: v.to(self.config["device"]) for k, v in inputs.items()}
        (prompt, inputs), prompt_seconds = measured(construct)
        sync()
        preparation = time.perf_counter() - start
        documents = [{"rank": i, "score": float(s), "content": d.page_content, "metadata": d.metadata}
                     for i, (d, s) in enumerate(zip(docs, scores), 1)]
        ids = inputs["input_ids"][0].tolist()
        snapshot = {"scenario_id": scenario["scenario_id"], "question": query, "prompt": prompt,
                    "input_token_ids": ids, "input_token_count": len(ids), "documents": documents,
                    "retrieval_seconds": retrieval, "prompt_and_tokenization_seconds": prompt_seconds,
                    "shared_preparation_seconds": preparation,
                    "retrieval_identity": fingerprint(documents), "prompt_identity": fingerprint(ids)}
        return snapshot, inputs

    def generate(self, snapshot, inputs, version, length, repeat, warmup, seed):
        import torch
        sync()
        execution_start = time.perf_counter()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        common = deepcopy(self.generation)
        if any(key in inputs for key in ("past_key_values", "cache_params", "mems", "past_buckets_states", "state", "cache_implementation")):
            raise ValueError("fresh benchmark requests must not supply a previous request's KV cache/selector")
        assistant = self.draft if version != "SD0" else None
        if version != "SD0" and assistant is None:
            raise ValueError("assisted mode requested without local Draft; no fallback")
        settings = None
        if assistant is not None:
            # CandidateGenerator copies this TARGET call config for Draft
            # generate(), then passes its own past_key_values on later rounds.
            common.cache_implementation = None
            common.cache_config = None
            assistant.generation_config = deepcopy(self.draft_defaults)
            assistant.generation_config.use_cache = True
            assistant.generation_config.cache_implementation = None
            assistant.generation_config.cache_config = None
            assistant.generation_config.num_assistant_tokens = length
            assistant.generation_config.num_assistant_tokens_schedule = "constant"
            assistant.generation_config.assistant_confidence_threshold = 0.0
            settings = {k: getattr(assistant.generation_config, k) for k in
                        ("num_assistant_tokens", "num_assistant_tokens_schedule", "assistant_confidence_threshold",
                         "use_cache", "cache_implementation", "cache_config")}
        selected = common.get_generation_mode(assistant_model=assistant).value
        if selected != ("assisted_generation" if assistant is not None else "greedy_search"):
            raise ValueError(f"unexpected generation mode {selected}; no fallback")
        local_inputs = {k: v.clone() for k, v in inputs.items()}
        calls = {"target": 0, "draft": 0}
        def counter(name):
            def count(module, args):
                calls[name] += 1
            return count
        handles = [self.target.register_forward_pre_hook(counter("target"))]
        if assistant is not None:
            handles.append(assistant.register_forward_pre_hook(counter("draft")))
        cuda = self.config["device"] == "cuda"
        if cuda:
            sync()
            torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated() if cuda else None
        free_before = list(torch.cuda.mem_get_info()) if cuda else None
        policy = assisted_cache_policy(self.target, assistant) if assistant is not None else nullcontext()
        try:
            with policy:
                sync()
                start = time.perf_counter()
                with torch.inference_mode():
                    output = self.target.generate(**local_inputs, generation_config=common,
                                                  **({"assistant_model": assistant} if assistant is not None else {}))
                sync()
                generation_seconds = time.perf_counter() - start
            if generation_seconds <= 0:
                raise RuntimeError("nonpositive generation timer; cannot calculate throughput")
            peak = torch.cuda.max_memory_allocated() if cuda else None
            reserved = torch.cuda.max_memory_reserved() if cuda else None
        finally:
            for handle in handles:
                handle.remove()
        count = snapshot["input_token_count"]
        if output.ndim != 2 or output.shape[0] != 1 or not torch.equal(output[0, :count], inputs["input_ids"][0]):
            raise RuntimeError("generation output does not preserve the original prompt")
        ids = output[0, count:].tolist()
        if not ids or calls["target"] <= 0 or (assistant is not None and calls["draft"] <= 0):
            raise RuntimeError("no output or missing Target/Draft forward; cannot certify requested decoding path")
        after = {k: getattr(assistant.generation_config, k) for k in settings} if settings else None
        if settings != after:
            raise RuntimeError("fixed assistant settings changed during generation")
        text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        sync()
        execution = time.perf_counter() - execution_start
        return {"scenario_id": snapshot["scenario_id"], "experiment_version": version,
                "question": snapshot["question"], "repeat_index": repeat, "warmup": warmup,
                "seed": seed, "status": "generated_pending_comparison", "fallback": False, "error": None,
                "prompt_identity": snapshot["prompt_identity"], "retrieval_identity": snapshot["retrieval_identity"],
                "target_identity": self.identity["target"], "draft_identity": self.identity["draft"],
                "output_text": text, "output_token_ids": ids, "output_token_sha256": fingerprint(ids),
                "input_token_count": count, "generated_token_count": len(ids),
                "generation_seconds": generation_seconds, "tokens_per_second": len(ids) / generation_seconds,
                "retrieval_seconds": snapshot["retrieval_seconds"],
                "prompt_and_tokenization_seconds": snapshot["prompt_and_tokenization_seconds"],
                "shared_preparation_seconds": snapshot["shared_preparation_seconds"],
                "mode_execution_seconds": execution,
                "end_to_end_seconds": snapshot["shared_preparation_seconds"] + execution,
                "end_to_end_kind": "controlled sum: shared preparation measured once + this mode execution",
                "assistant_candidate_limit": length, "assistant_settings_before": settings,
                "assistant_settings_after": after, "generation_mode": selected, "forward_calls": calls,
                "effective_use_cache": common.use_cache,
                "effective_cache_implementation": common.cache_implementation,
                "kv_cache_scope": "fresh request; HF internal reuse within generation only",
                "acceptance_rate": None, "acceptance_status": "unavailable; forward counts are not accepted token counts",
                "cuda_peak_allocated_bytes": peak, "cuda_peak_reserved_bytes": reserved,
                "cuda_allocated_before_bytes": before, "cuda_free_total_before_bytes": free_before,
                "residency": self.identity["residency"], "exact_token_match_sd0": None,
                "stop_reason": "eos" if ids[-1] == self.tokenizer.eos_token_id else
                               "token_limit" if len(ids) >= self.config["max_new_tokens"] else "other",
                "generation_speedup_sd0": None, "end_to_end_speedup_sd0": None}

    def verify_sources(self):
        self.adapter.verify_sources()
        if self.draft_path is not None and tree_hash(self.draft_path) != self.draft_hash:
            raise ValueError("Draft model source changed during run")
