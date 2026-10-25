"""Bounded FP32 comparison on an immutable saved A0 prompt; no FP16 fallback."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import traceback

from .diagnose_logits import Observer, read
from .reporting import write_json, append_jsonl, token_difference
from experiments.algorithms.semantic_cache.cache import fingerprint
from experiments.algorithms.shared.a0_adapter import file_hash


def memory_state():
    import torch
    free, total = torch.cuda.mem_get_info()
    allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    return {"free_bytes": free, "total_bytes": total, "allocated_bytes": allocated,
            "reserved_bytes": reserved, "usable_estimate_bytes": free + max(0, reserved - allocated),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()}


def configure(config):
    import torch
    if config["device"] != "cuda" or config["precision"] != "fp32":
        raise ValueError("FP32 control requires explicit CUDA/fp32; no CPU or lower-precision fallback")
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; stop FP32 control")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return memory_state()


def bind(engine):
    import torch
    models = {"target": engine.target}
    if engine.draft is not None:
        models["draft"] = engine.draft
    policy = {"attention_backend": "sdpa", "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
              "tf32_cudnn": torch.backends.cudnn.allow_tf32,
              "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "autocast": torch.is_autocast_enabled(), "models": {}}
    if policy["tf32_matmul"] or policy["tf32_cudnn"] or policy["autocast"]:
        raise ValueError("FP32 policy not effective")
    for name, model in models.items():
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError(f"{name}: not all floating parameters are FP32; no fallback")
        if any(p.device.type != "cuda" for p in model.parameters()):
            raise ValueError(f"{name}: parameters offloaded from CUDA; not this comparison")
        if getattr(model.config, "_attn_implementation", None) != "sdpa":
            setter = getattr(model, "set_attn_implementation", None)
            if setter is None:
                raise ValueError(f"{name}: cannot explicitly select SDPA")
            setter("sdpa")
        if getattr(model.config, "_attn_implementation", None) != "sdpa":
            raise ValueError(f"{name}: attention backend did not become SDPA")
        policy["models"][name] = {"dtype": str(next(model.parameters()).dtype),
                                 "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                                 "attention_backend": model.config._attn_implementation}
    engine.identity["attention_implementation"] = {name: "sdpa" for name in models}
    engine.identity["fp32_control"] = policy
    return policy


def ensure_memory(engine, prompt_length):
    """Conservative KV+workspace headroom check, not a guarantee against OOM."""
    budget = 0
    for model in (engine.target, engine.draft):
        if model is None:
            continue
        c = model.config
        head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        heads = getattr(c, "num_key_value_heads", None) or c.num_attention_heads
        length = prompt_length + engine.config["max_new_tokens"] + max(8, engine.config["standard_draft_tokens"]) + 1
        budget += 2 * c.num_hidden_layers * heads * head_dim * length * 4
    state = memory_state()
    state.update(estimated_kv_bytes=budget, workspace_reserve_bytes=1024 ** 3,
                 estimate_limitation="headroom heuristic; allocator/workspace OOM still fails without fallback")
    if state["usable_estimate_bytes"] < budget + state["workspace_reserve_bytes"]:
        raise RuntimeError(f"Insufficient FP32 KV/workspace headroom: {state}")
    return state


class Capture(Observer):
    def before(self, module, args, kwargs):
        super().before(module, args, kwargs)
        if self.pending is not None:
            complete = all(i in self.seen for i in range(self.wanted + 1))
            self.pending["observed_prefix_sha256"] = (fingerprint([self.seen[i] for i in range(self.wanted + 1)])
                                                       if complete else None)
            mask = kwargs.get("attention_mask")
            self.pending["attention_mask_type"] = type(mask).__name__
            self.pending["attention_mask_shape"] = list(mask.shape) if hasattr(mask, "shape") else None
            self.pending["attention_mask_all_ones"] = bool((mask == 1).all().item()) if getattr(mask, "ndim", 0) == 2 else None
            for key in ("position_ids", "cache_position"):
                value = kwargs.get(key)
                self.pending[key] = value.detach().cpu().tolist() if hasattr(value, "detach") else None
            cache = kwargs.get("past_key_values")
            try:
                self.pending["cache_layer_lengths_before"] = [int(cache.get_seq_length(i)) for i in range(module.config.num_hidden_layers)] if cache is not None else None
            except (AttributeError, TypeError, IndexError):
                self.pending["cache_layer_lengths_before"] = "unavailable; no inference change"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-run", type=Path, required=True, help="original failed SD benchmark, read-only")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    root = args.project_root.resolve()
    def locate(p):
        return (p if p.is_absolute() else root / p).resolve()
    source, output = locate(args.source_run), locate(args.run_dir)
    base = root / "experiments/results/algorithms"
    if output == base or not output.is_relative_to(base) or output.is_relative_to(source) or args.repeats < 2:
        raise ValueError("NEW independent output and at least two repeats required")
    rows = [read_line(line) for line in (source / "results.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    old = next(r for r in rows if r["scenario_id"] == "SC-E16-N" and r["experiment_version"] == "SD0")
    snapshot_path = (source / old["prompt_snapshot"]).resolve()
    if source not in snapshot_path.parents:
        raise ValueError("snapshot path escapes source")
    snapshot = read(snapshot_path)
    ids = snapshot["input_token_ids"]
    if fingerprint(ids) != snapshot["prompt_identity"] or len(ids) != snapshot["input_token_count"]:
        raise ValueError("saved prompt identity/count mismatch")
    original = read(source / "run.json")["config"]
    source_hashes = {str(p): file_hash(p) for p in (source / "run.json", source / "results.jsonl", snapshot_path)}
    config = deepcopy(original)
    config["precision"] = "fp32"
    output.mkdir(parents=True, exist_ok=False)
    stage = "cuda_preflight"
    try:
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[key] = "1"
        import torch
        from .engine import SDEngine
        before = configure(config)
        write_json(output / "fp32.config.json", config)
        write_json(output / "source.json", {"source_run": str(source), "source_config": original,
                   "source_files_sha256": source_hashes,
                   "source_prompt": str(snapshot_path), "memory_before_loading": before,
                   "changed": {"precision": [original["precision"], "fp32"]},
                   "policy": "all new FP32 methods compare only against NEW FP32 SD0; instrumented technical test, not speed evidence"})
        stage = "loading_fp32_models"
        engine = SDEngine(root, config, with_draft=True)
        bind(engine)
        if engine.identity["target"] != old["target_identity"] or engine.identity["draft"] != old["draft_identity"]:
            raise ValueError("local model source hashes differ from original evidence")
        if engine.tokenizer(snapshot["prompt"], return_tensors="pt", truncation=False)["input_ids"][0].tolist() != ids:
            raise ValueError("tokenizer/prompt token IDs changed")
        if len(ids) > config["max_input_tokens"] or any(len(ids) + config["max_new_tokens"] > m.config.max_position_embeddings
                                                       for m in (engine.target, engine.draft)):
            raise ValueError("saved prompt violates existing input/context limits; no truncation")
        write_json(output / "identity.json", {**engine.identity, "memory_after_loading": ensure_memory(engine, len(ids))})
        inputs = {"input_ids": torch.tensor([ids], device="cuda")}
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        variants = [("SD0", None), ("SD2-1", config["standard_draft_tokens"])] + [(f"SD2-2-k{k}", k) for k in (2, 4, 6, 8)]
        captured, controls = [], {}
        # Preserve the saved prompt exactly. Historical retrieval times are not
        # used to infer FP32 end-to-end performance from this instrumented replay.
        prefix = ids + old["output_token_ids"][:81]
        stage = "single_query_generation"
        for repeat in range(args.repeats):
            for version, length in variants:
                available = ensure_memory(engine, len(ids))
                observer = Capture(prefix, engine.tokenizer)
                handles = [engine.target.register_forward_pre_hook(observer.before, with_kwargs=True),
                           engine.target.register_forward_hook(observer.after, with_kwargs=True)]
                try:
                    row = engine.generate(snapshot, inputs, version, length, repeat, False, config["seed"] + repeat)
                finally:
                    for handle in handles:
                        handle.remove()
                row["measurement_id"] = f"FP32-{len(captured) + 1:04d}"
                if version == "SD0":
                    controls[repeat] = row
                    prefix = ids + row["output_token_ids"][:81]
                ref = controls[repeat]
                equality = row["output_token_ids"] == ref["output_token_ids"]
                expected_prefix = fingerprint(ids + ref["output_token_ids"][:81]) if len(ref["output_token_ids"]) >= 82 else None
                observations = observer.records
                for observed in observations:
                    observed["matches_new_fp32_baseline_prefix"] = (expected_prefix is not None
                        and observed["observed_prefix_sha256"] == expected_prefix)
                entry = {"generation": row, "exact_token_match_new_fp32_sd0": equality,
                         "token_equivalence_status": "matched_new_fp32_sd0" if equality else "failed_token_mismatch",
                         "token_difference": token_difference(row, ref, engine.tokenizer),
                         "target_forward_at_token82": observations, "mapping_errors": observer.mapping_errors,
                         "memory_before_generation": available, "memory_after_generation": memory_state(),
                         "timing_eligible": False}
                captured.append(entry)
                append_jsonl(output / "results.jsonl", entry)
                print(f"FP32 {version} repeat={repeat}: exact_tokens={equality}", flush=True)
        engine.verify_sources()
        if any(file_hash(Path(p)) != digest for p, digest in source_hashes.items()):
            raise ValueError("original evidence changed during FP32 control; do not commit comparison")
        repeatable = len({tuple(r["output_token_ids"]) for r in controls.values()}) == 1
        equal = all(r["exact_token_match_new_fp32_sd0"] for r in captured)
        captures_ok = all(any(o["matches_new_fp32_baseline_prefix"] for o in r["target_forward_at_token82"])
                          if len(controls[r["generation"]["repeat_index"]]["output_token_ids"]) >= 82 else True for r in captured)
        passed = equal and repeatable and captures_ok
        write_json(output / "summary.json", {"status": "passed_single_query" if passed else "failed_or_unverified",
            "provisional": True, "complete_evaluation_passed": False, "precision": "fp32", "tf32": False,
            "attention_backend": "sdpa", "measurements": len(captured), "baseline_repeatable": repeatable,
            "failed_token_measurements": sum(not r["exact_token_match_new_fp32_sd0"] for r in captured),
            "token82_capture_verified_when_reached": captures_ok,
            "baseline_token82_reached": all(len(r["output_token_ids"]) >= 82 for r in controls.values()),
            "timing_eligible": False, "formal_qualification": False,
            "interpretation": "a pass supports sensitivity to numerical precision/path on this query, not proof that FP16 is the only cause or universal losslessness"})
        print(f"FP32 control: {'PASS (single query only)' if passed else 'FAIL/UNVERIFIED'}: {output}")
        return 0 if passed else 2
    except Exception as exc:
        try:
            memory = memory_state()
        except Exception:
            memory = None
        write_json(output / "failed.json", {"stage": stage, "error": str(exc), "memory": memory,
            "precision_fallback": False, "traceback": traceback.format_exc()})
        raise


def read_line(line):
    import json
    return json.loads(line)


if __name__ == "__main__":
    raise SystemExit(main())
