"""Read-only Target forward telemetry for SC-E16-N; never a benchmark/pass waiver."""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import traceback

from .reporting import write_json
from experiments.algorithms.semantic_cache.cache import fingerprint


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def config_differences(expected, actual, prefix=""):
    """Compare the same serialized effective layer. No behavioral keys ignored.

    JSON round-tripping already makes tuples/lists equivalent in both inputs.
    Missing, null, numeric type changes and metadata remain visible: without
    evidence of the actual field difference, none is safe to waive.
    """
    if isinstance(expected, dict) != isinstance(actual, dict):
        obj = actual if isinstance(actual, dict) else expected
        return [{"field": (prefix + "." if prefix else "") + key,
                 "expected": expected.get(key) if isinstance(expected, dict) else expected,
                 "actual": actual.get(key) if isinstance(actual, dict) else actual,
                 "expected_container_type": type(expected).__name__,
                 "actual_container_type": type(actual).__name__} for key in sorted(obj)] or [
                     {"field": prefix, "expected": expected, "actual": actual}]
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(set(expected) | set(actual)):
            name = prefix + "." + key if prefix else key
            if key not in expected or key not in actual:
                result.append({"field": name, "expected_present": key in expected,
                               "actual_present": key in actual, "expected": expected.get(key),
                               "actual": actual.get(key)})
            else:
                result.extend(config_differences(expected[key], actual[key], name))
        return result
    return ([] if type(expected) is type(actual) and expected == actual else
            [{"field": prefix, "expected": expected, "actual": actual,
              "expected_type": type(expected).__name__, "actual_type": type(actual).__name__}])


def describe(scores, tokenizer):
    import torch
    scores = scores.detach().float().cpu()
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite raw Target logits")
    top = torch.topk(scores, 5)
    def token(i):
        value = float(scores[i])
        return {"id": i, "token": tokenizer.convert_ids_to_tokens(i),
                "logit": value, "rank": int((scores > value).sum()) + 1,
                "equal_logit_count": int((scores == value).sum())}
    return {"eot": token(128009), "jian": token(26892),
            "eot_minus_jian": float(scores[128009] - scores[26892]),
            "argmax": int(scores.argmax()), "top5": [token(int(i)) for i in top.indices]}


class Observer:
    """Observe causal logit at one absolute input position; never edit tensors/cache.

    logits position t predicts token t+1. Position mapping is accepted only when
    the input positions and logits_to_keep mapping can be established explicitly.
    A candidate suffix may exist beyond t; the entire causal prefix must match.
    """
    def __init__(self, prefix, tokenizer):
        self.prefix, self.tokenizer = prefix, tokenizer
        self.wanted = len(prefix) - 1
        self.seen, self.pending, self.records = {}, None, []
        self.calls, self.mapping_errors = 0, []

    def before(self, module, args, kwargs):
        self.calls += 1
        ids = kwargs.get("input_ids", args[0] if args else None)
        cache = kwargs.get("past_key_values")
        try:
            cache_length = int(cache.get_seq_length()) if cache is not None else 0
        except (AttributeError, TypeError):
            cache_length = None
        self.pending = None
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1:
            self.mapping_errors.append({"call": self.calls, "reason": "input_ids unavailable/batch != 1"})
            return
        values = ids[0].detach().cpu().tolist()
        positions = kwargs.get("cache_position")
        method = "cache_position"
        if positions is None:
            positions = kwargs.get("position_ids")
            method = "position_ids (single unpadded sequence)"
        if positions is not None:
            positions = positions.detach().reshape(-1).cpu().tolist()
        elif cache_length is not None:
            positions = list(range(cache_length, cache_length + len(values)))
            method = "contiguous append from pre-forward cache length"
        if positions is None or len(positions) != len(values):
            self.mapping_errors.append({"call": self.calls, "reason": "ambiguous input position mapping"})
            return
        # HF can crop/rewrite speculative suffixes. Overwrite positions, never
        # assume cached candidate tokens from an earlier round were accepted.
        for pos, value in zip(positions, values):
            self.seen[pos] = value
        matching = all(self.seen.get(i) == value for i, value in enumerate(self.prefix))
        keep = kwargs.get("logits_to_keep", kwargs.get("num_logits_to_keep", 0))
        if hasattr(keep, "detach"):
            indices = keep.detach().reshape(-1).cpu().tolist()
            try:
                output_positions = [positions[i] for i in indices]
            except IndexError:
                output_positions = []
        elif type(keep) is int and keep >= 0:
            output_positions = positions[-keep:] if keep else positions
        else:
            output_positions = []
        self.pending = {"call": self.calls, "input_shape": list(ids.shape),
                        "parameter_dtype": str(next(module.parameters()).dtype),
                        "attention_implementation": getattr(module.config, "_attn_implementation", None),
                        "input_positions": positions, "position_mapping": method,
                        "logits_to_keep": keep if type(keep) is int else str(keep),
                        "output_positions": output_positions,
                        "cache_type": type(cache).__name__ if cache is not None else None,
                        "cache_length_before": cache_length,
                        "use_cache": kwargs.get("use_cache"),
                        "causal_prefix_matches_fixed_prefix": matching}

    def after(self, module, args, kwargs, output):
        state = self.pending
        if state is None:
            return
        logits = getattr(output, "logits", None)
        if logits is None:
            self.mapping_errors.append({"call": self.calls, "reason": "forward returned no logits"})
            return
        positions = state.pop("output_positions")
        if logits.shape[1] != len(positions):
            self.mapping_errors.append({"call": self.calls, "reason": "logit length disagrees with explicit mapping",
                                        "shape": list(logits.shape), "mapped_length": len(positions)})
            return
        if self.wanted not in positions:
            return
        index = positions.index(self.wanted)
        state.update(logits_shape=list(logits.shape), logits_dtype=str(logits.dtype),
                     absolute_predictor_position=self.wanted, local_logit_index=index,
                     predicts_new_token_index=81,
                     raw_logits=describe(logits[0, index], self.tokenizer))
        self.records.append(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True, help="existing failed SD benchmark directory")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    def locate(p):
        return (p if p.is_absolute() else root / p).resolve()
    source, destination = locate(args.source_run), locate(args.run_dir)
    if (root / "experiments/results/algorithms").resolve() not in destination.parents:
        raise ValueError("diagnostics must use a NEW directory under experiments/results/algorithms")
    rows = [json.loads(line) for line in (source / "results.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    rows = [r for r in rows if r["scenario_id"] == "SC-E16-N"]
    baseline = next(r for r in rows if r["experiment_version"] == "SD0")
    snapshot_path = (source / baseline["prompt_snapshot"]).resolve()
    if source not in snapshot_path.parents:
        raise ValueError("prompt snapshot must belong to source run")
    snapshot = read(snapshot_path)
    prompt_ids = snapshot["input_token_ids"]
    if fingerprint(prompt_ids) != snapshot["prompt_identity"] or len(prompt_ids) != snapshot["input_token_count"]:
        raise ValueError("saved prompt identity/count mismatch")
    generated = baseline["output_token_ids"]
    if len(generated) != 82 or generated[81] != 128009:
        raise ValueError("source SD0 does not match the user-confirmed 82-token EOS case")
    variants = [("SD0", None), ("SD2-1", None)] + [(f"SD2-2-k{k}", k) for k in (2, 4, 6, 8)]
    paired = {}
    for version, _ in variants:
        matches = [r for r in rows if r["experiment_version"] == version
                   and r["repeat_index"] == baseline["repeat_index"] and r["warmup"] == baseline["warmup"]]
        if len(matches) != 1:
            raise ValueError(f"missing/duplicate source pair: {version}")
        r = matches[0]
        if (r["seed"] != baseline["seed"] or r["prompt_identity"] != snapshot["prompt_identity"]
                or r["retrieval_identity"] != snapshot["retrieval_identity"]
                or r["target_identity"] != baseline["target_identity"]
                or r["output_token_ids"][:81] != generated[:81]):
            raise ValueError(f"source identity/prefix mismatch: {version}")
        paired[version] = r
    supplied_config = read(locate(args.config))
    config = read(source / "run.json")["config"]
    input_config_diff = config_differences(config, supplied_config)
    source_identity = read(source / "identity.json") if (source / "identity.json").is_file() else {}
    destination.mkdir(parents=True, exist_ok=False)
    try:
        write_json(destination / "input_config_diff.json", {
            "differences": input_config_diff, "execution_config": config,
            "policy": "replay recorded benchmark configuration, never silently use changed local settings"})
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[name] = "1"
        import torch
        from .engine import SDEngine
        from transformers.generation.utils import GenerationMixin
        from transformers.generation.candidate_generator import AssistedCandidateGenerator
        write_json(destination / "installed_sources.json", {
            "assisted_decoding": inspect.getsource(GenerationMixin._assisted_decoding),
            "sample": inspect.getsource(GenerationMixin._sample),
            "prepare_cache": inspect.getsource(GenerationMixin._prepare_cache_for_generation),
            "candidate_generator": inspect.getsource(AssistedCandidateGenerator)})
        engine = SDEngine(root, config, with_draft=True)
        if engine.identity["target"] != baseline["target_identity"]:
            raise ValueError("Target source differs from recorded baseline")
        if any(engine.identity["draft"] != r["draft_identity"] for r in paired.values()):
            raise ValueError("Draft source differs from recorded generation")
        if engine.tokenizer(snapshot["prompt"], return_tensors="pt", truncation=False)["input_ids"][0].tolist() != prompt_ids:
            raise ValueError("current tokenizer does not reproduce saved prompt IDs")
        fixed = prompt_ids + generated[:81]
        inputs = {"input_ids": torch.tensor([prompt_ids], device=config["device"])}
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        write_json(destination / "identity.json", {"engine": engine.identity, "source_run": str(source),
                   "source_prompt": str(snapshot_path), "prompt_identity": snapshot["prompt_identity"],
                   "prefix_sha256": fingerprint(fixed), "new_token_index": 81,
                   "performance_measurement": False, "gold_certification": False})
        # Full-prefix recomputation is a reference path, NOT the SD0 ground truth.
        full = torch.tensor([fixed], device=config["device"])
        with torch.inference_mode():
            reference = engine.target(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False,
                                      return_dict=True, logits_to_keep=1)
        write_json(destination / "fixed_prefix_reference.json", {
            "path": "full prefix, use_cache=False; same dtype/backend; no generation",
            "logits_dtype": str(reference.logits.dtype),
            "raw_logits": describe(reference.logits[0, -1], engine.tokenizer)})
        del reference, full
        source_template = source_identity.get("effective_generation_config")
        template_diff = (config_differences(source_template, engine.identity["effective_generation_config"])
                         if isinstance(source_template, dict) and source_template else [])
        template_verified = isinstance(source_template, dict) and bool(source_template) and not template_diff
        write_json(destination / "template_config_comparison.json", {
            "source_location": "identity.json:effective_generation_config",
            "current_location": "engine.identity.effective_generation_config",
            "status": "verified_match" if template_verified else "different" if template_diff else "unverified",
            "differences": template_diff,
            "scope": "shared pre-call template only; NOT per-mode effective call config"})
        records, all_config_diff, config_checks = [], [], []
        for version, length in variants:
            observer = Observer(fixed, engine.tokenizer)
            handles = [engine.target.register_forward_pre_hook(observer.before, with_kwargs=True),
                       engine.target.register_forward_hook(observer.after, with_kwargs=True)]
            try:
                row = engine.generate(snapshot, inputs, version,
                                      config["standard_draft_tokens"] if version == "SD2-1" else length,
                                      baseline["repeat_index"], baseline["warmup"], baseline["seed"])
            finally:
                for handle in handles:
                    handle.remove()
            captures = [r for r in observer.records if r["causal_prefix_matches_fixed_prefix"]]
            source_effective = paired[version].get("effective_generation_config")
            source_saved = isinstance(source_effective, dict) and bool(source_effective)
            diffs = config_differences(source_effective, row["effective_generation_config"]) if source_saved else []
            # Missing evidence is one unverified configuration, not hundreds of
            # fabricated parameter mismatches. Never substitute the template.
            critical = ("eos_token_id", "pad_token_id", "max_new_tokens", "do_sample", "num_beams",
                        "use_cache", "cache_implementation", "cache_config")
            missing_critical = [k for k in critical if not source_saved or k not in source_effective]
            check = {"version": version, "source_location": "results.jsonl:effective_generation_config",
                     "current_location": "generation.effective_generation_config",
                     "status": "unverified" if not source_saved or missing_critical else "different" if diffs else "verified_match",
                     "reason": "per-call effective config was not saved" if not source_saved else None,
                     "missing_critical_fields": missing_critical, "differences": diffs}
            config_checks.append(check)
            all_config_diff.extend({"version": version, **d} for d in diffs)
            old_ids, new_ids = generated, row["output_token_ids"]
            first_diff = next((i for i in range(min(len(old_ids), len(new_ids))) if old_ids[i] != new_ids[i]),
                              min(len(old_ids), len(new_ids)) if len(old_ids) != len(new_ids) else None)
            record = {"version": version, "generation": row, "target_forward_observations": observer.records,
                      "generation_config_differences": diffs,
                      "generation_config_check": check,
                      "first_different_new_token_index": first_diff,
                      "mapping_errors": observer.mapping_errors,
                      "reproduces_source_tokens": row["output_token_ids"] == paired[version]["output_token_ids"],
                      "exact_token_match_sd0": row["output_token_ids"] == generated,
                      "capture_complete": bool(captures),
                      "timings": "instrumented/synchronizing: NOT usable for performance comparisons"}
            write_json(destination / f"{version}.json", record)
            for difference in diffs:
                print("CONFIG_DIFF " + json.dumps({"version": version, **difference}, ensure_ascii=False), flush=True)
            if check["status"] == "unverified":
                print("CONFIG_UNVERIFIED " + json.dumps(check, ensure_ascii=False), flush=True)
            records.append(record)
        engine.verify_sources()
        complete = all(r["capture_complete"] for r in records)
        replay_sd0 = records[0]["generation"]["output_token_ids"]
        replay_equal = all(r["generation"]["output_token_ids"] == replay_sd0 for r in records)
        configs_equal = (all(c["status"] == "verified_match" for c in config_checks)
                         and template_verified and not input_config_diff)
        write_json(destination / "generation_config_diff.json", {
            "differences": all_config_diff, "per_mode_checks": config_checks,
            "source_template_status": "verified_match" if template_verified else "different" if template_diff else "unverified",
            "comparison_layer": "saved per-call effective config vs current per-call effective config; template compared separately",
            "normalization": "none; no field difference proven behavior-neutral"})
        for difference in input_config_diff:
            print("CONFIG_DIFF " + json.dumps(difference, ensure_ascii=False), flush=True)
        argmax = {r["version"]: [o["raw_logits"]["argmax"] for o in r["target_forward_observations"]
                                 if o["causal_prefix_matches_fixed_prefix"]] for r in records}
        baseline_argmax = set(argmax.get("SD0", []))
        path_difference = any(set(values) != baseline_argmax for mode, values in argmax.items() if mode != "SD0")
        evidence_category = ("configuration_identity_not_verified" if not configs_equal else
                             "raw_target_argmax_differs_by_forward_path" if complete and path_difference else
                             "raw_logits_do_not_explain_decision; inspect acceptance/processors/stop path" if complete and not replay_equal else
                             "no_reproduced_mismatch" if replay_equal else "insufficient_capture")
        write_json(destination / "diagnostic_summary.json", {
            "capture_complete": complete, "equivalence_passed": replay_equal and configs_equal and complete and replay_sd0 == generated,
            "baseline_reproduces_source": replay_sd0 == generated,
            "replay_token_equivalence_only": replay_equal, "generation_config_verified": configs_equal,
            "source_reproduced": all(r["reproduces_source_tokens"] for r in records),
            "argmax_at_fixed_prefix": argmax,
            "evidence_category": evidence_category,
            "root_cause": "not established; differing raw argmax supports a numerical/path hypothesis but does not prove FP16 or cache fault",
            "root_cause_evidence_sufficient": False,
            "generation_calls": len(records), "fixed_prefix_forward_calls": 1,
            "resume_authorized_by_this_tool": False,
            "interpretation": "raw forward logits only; inspect installed assisted acceptance/logits processors before attributing a cause"})
        if not complete:
            raise RuntimeError("missing unambiguously mapped divergence-position logits; inspect mapping_errors")
        print(f"Diagnostic captured (not an equivalence pass or benchmark): {destination}")
    except Exception:
        write_json(destination / "failed.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
