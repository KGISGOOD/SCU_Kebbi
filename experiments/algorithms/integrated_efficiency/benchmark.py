"""INT0 / INT1 / INT2 / INT3: independent online requests, offline resources."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

from experiments.algorithms.semantic_cache.threshold_data import load_sequence
from experiments.algorithms.semantic_cache.cache import fingerprint
from experiments.algorithms.speculative_decoding.reporting import write_json, append_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("all", "int0", "int1", "int2", "int3"), default="all")
    parser.add_argument("--decoder", choices=("standard", "draft_1b", "prompt_lookup"))
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--prompt-lookup-tokens", type=int)
    parser.add_argument("--max-matching-ngram-size", type=int)
    parser.add_argument("--scenarios", type=Path, default=Path("experiments/algorithms/integrated_efficiency/scenarios.jsonl"))
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--allow-unreviewed", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()

    def locate(raw):
        path = Path(raw).expanduser()
        return (path if path.is_absolute() else root / path).resolve()

    config_path, scenario_path, run = locate(args.config), locate(args.scenarios), locate(args.run_dir)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    required = {"model", "bge", "knowledge_index", "device", "precision", "seed", "max_new_tokens",
                "max_input_tokens", "standard_draft_tokens", "repeats", "warmups"}
    optional = {"draft_model", "decoder", "threshold", "prompt_lookup_tokens", "max_matching_ngram_size"}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - optional:
        raise ValueError(f"config requires {sorted(required)}; optional {sorted(optional)}")
    for key, value in {"decoder": "draft_1b", "threshold": 0.75, "prompt_lookup_tokens": 4,
                       "max_matching_ngram_size": 2, "draft_model": "LOCAL_MODELS/Llama-3.2-1B-Instruct"}.items():
        config.setdefault(key, value)
    for key, override in (("decoder", args.decoder), ("threshold", args.threshold),
                          ("prompt_lookup_tokens", args.prompt_lookup_tokens),
                          ("max_matching_ngram_size", args.max_matching_ngram_size),
                          ("repeats", args.repeats), ("warmups", args.warmups)):
        if override is not None:
            config[key] = override
    for key in ("seed", "max_new_tokens", "max_input_tokens", "standard_draft_tokens", "repeats", "warmups",
                "prompt_lookup_tokens", "max_matching_ngram_size"):
        minimum = 0 if key == "seed" else 2 if key == "max_new_tokens" else 1
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key}: integer >= {minimum} required")
    if config["max_input_tokens"] > 2048:
        raise ValueError("preserve existing A0 full-input policy: <=2048; never truncate Top-5")
    if config["decoder"] not in {"standard", "draft_1b", "prompt_lookup"}:
        raise ValueError("invalid decoder")
    if config["device"] not in {"cpu", "cuda"} or config["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("invalid device/precision")
    if (type(config["threshold"]) not in {int, float} or not math.isfinite(config["threshold"])
            or not -1 <= config["threshold"] <= 1):
        raise ValueError("finite threshold in [-1,1] required")
    variants = ["INT0", "INT1", "INT2", "INT3"] if args.mode == "all" else [args.mode.upper()]
    has_sd = any(v in {"INT2", "INT3"} for v in variants)
    with_draft = has_sd and config["decoder"] == "draft_1b"
    resources = ["model", "bge", "knowledge_index"] + (["draft_model"] if with_draft else [])
    for key in resources:
        if not isinstance(config[key], str) or not config[key].strip() or not locate(config[key]).is_dir():
            raise ValueError(f"{key}: existing local directory required; no downloads")
        if run == locate(config[key]) or locate(config[key]) in run.parents:
            raise ValueError("run must not be inside model/index directories")
    if (root / "experiments/results/algorithms").resolve() not in run.parents:
        raise ValueError("run-dir must be a NEW child of experiments/results/algorithms")
    cases = load_sequence(scenario_path, "functional", args.allow_unreviewed)
    provisional = any(r["review_status"] != "approved" or any(j["review_status"] != "approved"
                      for j in r.get("reuse_judgments", [])) for r in cases)
    run.mkdir(parents=True, exist_ok=False)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    from experiments.algorithms.shared.a0_adapter import file_hash
    hashes = {"config": file_hash(config_path), "scenarios": file_hash(scenario_path)}
    write_json(run / "run.json", {"status": "running", "config": config, "variants": variants,
                                  "source_hashes": hashes, "dataset_identity": hashes["scenarios"],
                                  "provisional": provisional, "purpose": "functional integrated online benchmark",
                                  "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  "cache_policy": "own empty cache per variant/repeat/warmup; own Miss answers only"})
    write_json(run / "scenarios.snapshot.json", cases)
    rows, current, stage = [], None, "loading"
    wall_start = time.perf_counter()
    try:
        import torch
        from experiments.algorithms.speculative_decoding.engine import SDEngine
        from .pipeline import IntegratedPipeline
        from .reporting import report
        engine = SDEngine(root, config, with_draft=with_draft)
        if has_sd and config["decoder"] == "prompt_lookup":
            from experiments.algorithms.speculative_decoding.prompt_lookup import check_lookup_api
            engine.identity["prompt_lookup_api"] = check_lookup_api()
        identity = {"engine": engine.identity, "dataset_identity": hashes["scenarios"],
                    "effective_config": config, "variant_decoders": {
                        v: config["decoder"] if v in {"INT2", "INT3"} else "standard" for v in variants},
                    "hardware": {"gpu": torch.cuda.get_device_name(0) if config["device"] == "cuda" else None,
                                 "torch_cuda": torch.version.cuda},
                    "residency_policy": "same resident models for all variants in this run; no hidden precision changes",
                    "timing_policy": "independent online wall time per request, including each own retrieval"}
        write_json(run / "identity.json", identity)
        identity_hash = fingerprint(identity)
        order = list(variants)
        random.Random(config["seed"]).shuffle(order)
        for warmup, repeats in ((True, config["warmups"]), (False, config["repeats"])):
            for repeat in range(repeats):
                # These are distinct AnswerCache instances with fresh FAISS ID maps.
                pipelines = {v: IntegratedPipeline(engine, v, config["decoder"], repeat, warmup, config["threshold"])
                             for v in variants}
                for case_index, case in enumerate(cases):
                    offset = (case_index + repeat) % len(order)
                    for position, variant in enumerate(order[offset:] + order[:offset]):
                        current = {"scenario_id": case["scenario_id"], "variant": variant,
                                   "repeat": repeat, "warmup": warmup}
                        stage = "online_request"
                        row = pipelines[variant].answer(case)
                        row.update(measurement_id=f"I{len(rows) + 1:06d}", order_position=position,
                                   provisional=provisional, dataset_identity=hashes["scenarios"],
                                   experiment_identity=identity_hash)
                        append_jsonl(run / "measurements.raw.jsonl", row)
                        rows.append(row)
                        print(f"{row['measurement_id']} {variant}/{row['decoder']} {case['scenario_id']} "
                              f"warmup={warmup} repeat={repeat} hit={row['cache_hit']} "
                              f"online={row['end_to_end_seconds']:.4f}s", flush=True)
                stage = "cache_consistency"
                for pipeline in pipelines.values():
                    pipeline.pipeline.cache.check_consistency()
        stage = "source_verification"
        engine.verify_sources()
        if hashes != {"config": file_hash(config_path), "scenarios": file_hash(scenario_path)}:
            raise ValueError("config/scenarios changed during run")
        if fingerprint(identity) != identity_hash:
            raise ValueError("experiment identity changed during run")
        stage = "reporting_and_token_comparison"
        failures = report(run, rows, provisional)
        if failures:
            raise ValueError(f"{failures} input/token/repeatability failures; inspect results")
        write_json(run / "completed.json", {"status": "completed", "provisional": provisional,
                                            "source_verification": "passed", "answer_quality_verified": False,
                                            "measured_count": sum(not r["warmup"] for r in rows),
                                            "warmup_count": sum(r["warmup"] for r in rows),
                                            "paired_token_comparison_available": "INT0" in variants,
                                            "wall_seconds_including_load_warmups_reports_and_hashing": time.perf_counter() - wall_start})
    except BaseException as error:
        write_json(run / "failed.json", {"status": "failed", "stage": stage, "request": current,
                                         "error": str(error), "exception_type": type(error).__name__,
                                         "traceback": traceback.format_exc(), "fallback": False,
                                         "cache_failure": getattr(error, "cache_failure", None),
                                         "saved_raw_measurements": len(rows)})
        print(f"INT FAILED stage={stage} request={current}: {error}", flush=True)
        raise


if __name__ == "__main__":
    main()
