"""Research-data INT runner, preserving the existing independent online timing."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import random
import traceback

from .common import ROOT, locate, new_output, read_json, write_json, fingerprint
from .runtime import load_input, verify_input, append_row


def configuration(root, path):
    """Same contract/defaults as integrated_efficiency.benchmark; no core edits."""
    config = read_json(path)
    required = {"model", "bge", "knowledge_index", "device", "precision", "seed", "max_new_tokens",
                "max_input_tokens", "standard_draft_tokens", "repeats", "warmups"}
    optional = {"draft_model", "decoder", "threshold", "prompt_lookup_tokens", "max_matching_ngram_size"}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - optional:
        raise ValueError("use the existing integrated_efficiency config format")
    for key, value in {"decoder": "draft_1b", "threshold": 0.75, "prompt_lookup_tokens": 4,
                       "max_matching_ngram_size": 2,
                       "draft_model": "LOCAL_MODELS/Llama-3.2-1B-Instruct"}.items():
        config.setdefault(key, value)
    for key in ("seed", "max_new_tokens", "max_input_tokens", "standard_draft_tokens", "repeats", "warmups",
                "prompt_lookup_tokens", "max_matching_ngram_size"):
        minimum = 0 if key == "seed" else 2 if key == "max_new_tokens" else 1
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key}: integer >= {minimum} required")
    if config["max_input_tokens"] > 2048:
        raise ValueError("preserve A0 full-input <=2048 policy; never truncate Top-5")
    if config["decoder"] not in {"standard", "draft_1b", "prompt_lookup"}:
        raise ValueError("invalid decoder")
    if config["device"] not in {"cpu", "cuda"} or config["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("invalid device/precision")
    if type(config["threshold"]) not in {int, float} or not math.isfinite(config["threshold"]) or not -1 <= config["threshold"] <= 1:
        raise ValueError("threshold must be finite in [-1,1]")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--config", required=True, help="existing integrated_efficiency local config")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--view", help="same merged review view used at export")
    parser.add_argument("--mode", choices=("all", "int0", "int1", "int2", "int3"), default="all")
    parser.add_argument("--allow-unreviewed", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    if root != ROOT:
        raise ValueError("project-root must match imported checkout")
    directory, config_path = locate(root, args.adapter_dir), locate(root, args.config)
    view = locate(root, args.view) if args.view else None
    manifest, phases, _, receipt = load_input(root, directory, "int", config_path, args.allow_unreviewed, view)
    config = configuration(root, config_path)
    variants = ["INT0", "INT1", "INT2", "INT3"] if args.mode == "all" else [args.mode.upper()]
    if not manifest["provisional"] and args.mode != "all":
        raise ValueError("formal comparison requires all variants, including paired INT0")
    has_sd = any(v in {"INT2", "INT3"} for v in variants)
    with_draft = has_sd and config["decoder"] == "draft_1b"
    for name in ["model", "bge", "knowledge_index"] + (["draft_model"] if with_draft else []):
        if not isinstance(config[name], str) or not locate(root, config[name]).is_dir():
            raise ValueError(f"{name}: existing local directory required; no download")
    run = new_output(root, args.run_dir)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    rows, current, stage = [], None, "loading"
    try:
        write_json(run / "run.json", {"runner": "research_int_v1", "config": config,
                   "variants": variants, "split": manifest["split"], "purpose": manifest["purpose"],
                   "provisional": manifest["provisional"], "research_input": receipt,
                   "cache_policy": "empty per variant/repeat/warmup; initial_fill then measured on SAME instance",
                   "timing_policy": "unchanged independent online E2E; fill cost and warmups reported separately"})
        write_json(run / "scenarios.snapshot.json", phases)
        from experiments.algorithms.speculative_decoding.engine import SDEngine
        from experiments.algorithms.integrated_efficiency.pipeline import IntegratedPipeline
        from experiments.algorithms.integrated_efficiency.reporting import report
        engine = SDEngine(root, config, with_draft=with_draft)
        if has_sd and config["decoder"] == "prompt_lookup":
            from experiments.algorithms.speculative_decoding.prompt_lookup import check_lookup_api
            engine.identity["prompt_lookup_api"] = check_lookup_api()
        identity = {"engine": engine.identity, "research_input": receipt, "effective_config": config,
                    "variants": variants, "residency_policy": "same resident models and precision for every variant"}
        experiment_hash = fingerprint(identity)
        write_json(run / "identity.json", identity)
        order = list(variants)
        random.Random(config["seed"]).shuffle(order)
        for warmup, repeats in ((True, config["warmups"]), (False, config["repeats"])):
            for repeat in range(repeats):
                # Each mode has its OWN cache; neither warmups nor repeats carry it forward.
                pipelines = {v: IntegratedPipeline(engine, v, config["decoder"], repeat, warmup,
                                                     config["threshold"]) for v in variants}
                offset_index = 0
                for phase, cases in phases.items():
                    for case in cases:
                        offset = (offset_index + repeat) % len(order)
                        offset_index += 1
                        for position, variant in enumerate(order[offset:] + order[:offset]):
                            current = {"scenario_id": case["scenario_id"], "variant": variant,
                                       "phase": phase, "repeat": repeat, "warmup": warmup}
                            stage = "online_request"
                            # No reference answer/expected behavior is passed. Pair labels
                            # are consumed by the existing pipeline AFTER answer serving.
                            allowed = ("scenario_id", "question", "query_order", "review_status",
                                       "reference_scenario_id", "expected_answer_reuse", "reuse_judgments")
                            request = {k: case[k] for k in allowed}
                            if manifest["provisional"]:
                                request["review_status"] = "needs_review"
                            row = pipelines[variant].answer(request)
                            row.update(measurement_id=f"RI{len(rows) + 1:06d}", phase=phase,
                                       question_id=case["question_id"], order_position=position,
                                       research_metadata=case["research_metadata"],
                                       provisional=manifest["provisional"], dataset_identity=manifest["dataset_identity"],
                                       experiment_identity=experiment_hash)
                            append_row(run / "measurements.raw.jsonl", row)
                            rows.append(row)
                            print(f"{row['measurement_id']} {phase} {variant} {case['question_id']} hit={row['cache_hit']}", flush=True)
                stage = "cache_consistency"
                for pipeline in pipelines.values():
                    pipeline.pipeline.cache.check_consistency()
        stage = "source_verification"
        engine.verify_sources()
        verify_input(root, directory, "int", config_path, args.allow_unreviewed, view, receipt)
        if fingerprint(identity) != experiment_hash:
            raise ValueError("experiment identity changed")
        stage = "reporting_and_token_comparison"
        summaries, failures = {}, 0
        for phase in phases:
            destination = run / phase
            destination.mkdir()
            group = [r for r in rows if r["phase"] == phase]
            if group:
                # Original paired-token validation and timing formulas, unchanged.
                failures += report(destination, group, manifest["provisional"])
                summaries[phase] = read_json(destination / "summary.json")
            else:
                summaries[phase] = {"status": "not_applicable", "reason": "cold workload; no initial fill"}
                write_json(destination / "summary.json", summaries[phase])
        write_json(run / "summary.json", {"provisional": manifest["provisional"], "phases": summaries,
                   "initial_fill_cost_excluded_from_measured": True, "answer_quality_verified": False,
                   "fill_cost_definition": "initial_fill/summary.json per-variant total_online_seconds, warmups excluded",
                   "speedup_definition": "unchanged paired INT0/variant measured online E2E; fill cost not hidden in measured denominator"})
        if failures:
            raise ValueError(f"{failures} input/token/repeatability failures; inspect phase results")
        write_json(run / "completed.json", {"status": "completed", "source_verification": "passed",
                   "provisional": manifest["provisional"], "answer_quality_verified": False,
                   "research_input": receipt, "measured_count": sum(not r["warmup"] and r["phase"] == "measured" for r in rows),
                   "initial_fill_count": sum(not r["warmup"] and r["phase"] == "initial_fill" for r in rows)})
    except BaseException as exc:
        write_json(run / "failed.json", {"error": str(exc), "stage": stage, "request": current,
                   "traceback": traceback.format_exc(), "saved_raw_measurements": len(rows), "fallback": False})
        raise


if __name__ == "__main__":
    main()
