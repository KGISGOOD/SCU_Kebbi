"""SD0 / SD2-1 / SD2-2; all models local, outputs exclusively in a NEW run."""
from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import random
import time
import traceback
import json

from .reporting import write_json, append_jsonl, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("all", "sd0", "sd2_1", "sd2_2", "sd2_3", "sd2_4"), default="all")
    parser.add_argument("--max-matching-ngram-size", type=int)
    parser.add_argument("--scenarios", type=Path, default=Path("experiments/algorithms/speculative_decoding/scenarios.jsonl"))
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--scenario-id", help="explicit single-scenario diagnostic; never a complete Evaluation")
    args = parser.parse_args()
    if args.mode == "sd2_3":
        from .compatibility import adaptive_policy
        adaptive_policy()
    root = args.project_root.resolve()
    def locate(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else root / path).resolve()
    config_path, scenario_path, run = locate(args.config), locate(args.scenarios), locate(args.run_dir)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    required = {"model", "draft_model", "bge", "knowledge_index", "device", "precision", "seed",
                "max_new_tokens", "max_input_tokens", "standard_draft_tokens", "repeats", "warmups"}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - {"max_matching_ngram_size"}:
        raise ValueError(f"config requires {sorted(required)}; optional max_matching_ngram_size")
    if args.max_matching_ngram_size is not None:
        config["max_matching_ngram_size"] = args.max_matching_ngram_size
    if args.mode == "sd2_4":
        config.setdefault("max_matching_ngram_size", 2)
    if "max_matching_ngram_size" in config and (type(config["max_matching_ngram_size"]) is not int or config["max_matching_ngram_size"] < 1):
        raise ValueError("max_matching_ngram_size must be a positive integer")
    for key, override in (("repeats", args.repeats), ("warmups", args.warmups)):
        if override is not None:
            config[key] = override
    for key in ("seed", "max_new_tokens", "max_input_tokens", "standard_draft_tokens", "repeats", "warmups"):
        minimum = 0 if key == "seed" else 2 if key == "max_new_tokens" else 1
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"invalid {key}; minimum {minimum}")
    if config["max_input_tokens"] > 2048:
        raise ValueError("A0 comparison input limit must not exceed its existing 2048-token policy")
    if config["device"] not in {"cuda", "cpu"} or config["precision"] not in {"fp16", "bf16", "fp32"}:
        raise ValueError("invalid device/precision")
    for key in ("model", "draft_model", "bge", "knowledge_index"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"invalid {key} path")
        if key != "draft_model" or args.mode not in {"sd0", "sd2_4"}:
            if not locate(config[key]).is_dir():
                raise ValueError(f"{key}: LOCAL directory missing: {locate(config[key])}; no download")
    cases = []
    for number, line in enumerate(scenario_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank scenario line {number}")
        row = json.loads(line)
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip()
                                           for k in ("scenario_id", "query", "category", "review_status")):
            raise ValueError(f"invalid scenario line {number}")
        if row["review_status"] not in {"needs_review", "approved"}:
            raise ValueError(f"invalid review_status at line {number}")
        if any(row.get(k) for k in ("history", "messages", "context")):
            raise ValueError("SD V1 uses A0 single-turn questions only")
        cases.append(row)
    if not cases or len({r["scenario_id"] for r in cases}) != len(cases):
        raise ValueError("empty scenarios or duplicate scenario_id")
    source_indices = {r["scenario_id"]: i for i, r in enumerate(cases)}
    if args.scenario_id:
        cases = [r for r in cases if r["scenario_id"] == args.scenario_id]
        if len(cases) != 1:
            raise ValueError("diagnostic scenario ID absent from the supplied existing source")
    provisional = any(r["review_status"] != "approved" for r in cases)
    if provisional and not args.allow_unreviewed:
        raise ValueError("scenarios require human review; --allow-unreviewed is development only")
    if (root / "experiments/results/algorithms").resolve() not in run.parents:
        raise ValueError("run-dir must be a NEW child of experiments/results/algorithms")
    for key in ("model", "draft_model", "bge", "knowledge_index"):
        if run == locate(config[key]) or locate(config[key]) in run.parents:
            raise ValueError("run-dir cannot be inside model/index resources")
    run.mkdir(parents=True, exist_ok=False)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    from experiments.algorithms.shared.a0_adapter import file_hash
    hashes = {"config": file_hash(config_path), "scenarios": file_hash(scenario_path)}
    variants = [("SD0", None)]
    if args.mode in {"all", "sd2_1"}:
        variants.append(("SD2-1", config["standard_draft_tokens"]))
    if args.mode in {"all", "sd2_2"}:
        variants.extend((f"SD2-2-k{k}", k) for k in (2, 4, 6, 8))
    if args.mode == "sd2_4":
        variants.extend((f"SD2-4-k{k}", k) for k in (2, 4, 6, 8))
    write_json(run / "run.json", {"status": "running", "config": config, "source_hashes": hashes,
                                  "mode": args.mode, "variants": variants, "provisional": provisional,
                                  "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  "purpose": "controlled A0 speculative decoding development benchmark"})
    if args.scenario_id:
        write_json(run / "diagnostic_scope.json", {"selected_scenario_id": args.scenario_id,
                   "complete_evaluation": False, "source_sha256": hashes["scenarios"],
                   "generation_policy_changed": False})
    write_json(run / "scenarios.snapshot.json", cases)
    raw = run / "measurements.raw.jsonl"
    with raw.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("")
    rows, stage, current = [], "loading", None
    wall_start = time.perf_counter()
    try:
        from .engine import SDEngine
        import torch
        engine = SDEngine(root, config, with_draft=args.mode in {"all", "sd2_1", "sd2_2"})
        if args.mode == "sd2_4":
            from .prompt_lookup import check_lookup_api
            engine.identity["prompt_lookup_api"] = check_lookup_api()
        engine.identity["hardware"] = {"cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
                                      "gpu": torch.cuda.get_device_name(0) if config["device"] == "cuda" else None,
                                      "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory
                                      if config["device"] == "cuda" else None,
                                      "cudnn": torch.backends.cudnn.version(),
                                      "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                      "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}
        write_json(run / "identity.json", engine.identity)
        if args.scenario_id:
            import inspect
            from transformers.generation.utils import GenerationMixin
            from transformers.generation.candidate_generator import AssistedCandidateGenerator
            write_json(run / "installed_generation_sources.json", {
                "transformers": engine.api["transformers"],
                "assisted_decoding": inspect.getsource(GenerationMixin._assisted_decoding),
                "prepare_cache": inspect.getsource(GenerationMixin._prepare_cache_for_generation),
                "candidate_generator": inspect.getsource(AssistedCandidateGenerator)})
        order = list(variants)
        random.Random(config["seed"]).shuffle(order)
        preparation_total = 0.0
        for index, scenario in enumerate(cases):
            current = {"scenario_id": scenario["scenario_id"]}
            stage = "prepare_a0_prompt"
            snapshot, inputs = engine.prepare(scenario)
            preparation_total += snapshot["shared_preparation_seconds"]
            snapshot_file = f"prompt_{index + 1:04d}.json"
            write_json(run / snapshot_file, snapshot)
            for warmup, repeats in ((True, config["warmups"]), (False, config["repeats"])):
                for repeat in range(repeats):
                    offset = (source_indices[scenario["scenario_id"]] + repeat) % len(order)
                    sequence = order[offset:] + order[:offset]
                    for position, (version, length) in enumerate(sequence):
                        current = {"scenario_id": scenario["scenario_id"], "experiment_version": version,
                                   "repeat_index": repeat, "warmup": warmup, "candidate_limit": length}
                        stage = "generation"
                        result = engine.generate(snapshot, inputs, version, length, repeat, warmup,
                                                 config["seed"] + repeat)
                        result.update(measurement_id=f"M{len(rows) + 1:06d}", order_position=position,
                                      prompt_snapshot=snapshot_file, provisional=provisional,
                                      source_scenario_index=source_indices[scenario["scenario_id"]])
                        append_jsonl(raw, result)
                        rows.append(result)
                        print(f"{result['measurement_id']} {version} {scenario['scenario_id']} "
                              f"warmup={warmup} repeat={repeat} tokens={result['generated_token_count']} "
                              f"generation={result['generation_seconds']:.4f}s", flush=True)
        stage = "verify_sources"
        engine.verify_sources()
        if hashes != {"config": file_hash(config_path), "scenarios": file_hash(scenario_path)}:
            raise ValueError("config/scenario source changed during run")
        stage = "comparison_and_reporting"
        failures = report(run, rows, engine.identity, provisional, engine.tokenizer)
        if failures:
            raise ValueError(f"{failures} token-equivalence/repeatability failures; inspect results, no lossless claim")
        write_json(run / "completed.json", {"status": "completed", "source_verification": "passed",
                                            "measured_count": sum(not r["warmup"] for r in rows),
                                            "warmup_count": sum(r["warmup"] for r in rows),
                                            "unique_shared_preparation_seconds": preparation_total,
                                            "wall_seconds_including_load_warmup_reporting_and_hashing": time.perf_counter() - wall_start,
                                            "provisional": provisional})
    except BaseException as error:
        failure = {"status": "failed", "stage": stage, "measurement": current,
                   "error": str(error), "exception_type": type(error).__name__,
                   "traceback": traceback.format_exc(), "saved_raw_measurements": len(rows), "fallback": False}
        write_json(run / "failed.json", failure)
        print(f"SD FAILED stage={stage}: {error}", flush=True)
        raise


if __name__ == "__main__":
    main()
