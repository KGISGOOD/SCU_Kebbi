"""Calibrate on one split, freeze thresholds, evaluate independently."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import time
import traceback

from .data import load_cases, split_manifest, check_separation
from .metrics import fit_threshold
from .reporting import write_json, append, detection_report, context_report
from experiments.algorithms.semantic_cache.cache import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("calibrate", "evaluate", "all"), default="all")
    parser.add_argument("--mode", choices=("detection", "all"), default="detection", help="all additionally runs TI3-3 generation")
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--calibration", type=Path, default=Path("experiments/algorithms/topic_identification/calibration.jsonl"))
    parser.add_argument("--evaluation", type=Path, default=Path("experiments/algorithms/topic_identification/evaluation.jsonl"))
    parser.add_argument("--allow-unreviewed", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    def locate(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else root / path).resolve()
    config_path, run = locate(args.config), locate(args.run_dir)
    from experiments.algorithms.shared.a0_adapter import file_hash
    hashes = {str(config_path): file_hash(config_path)}
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    keys = {"model", "bge", "knowledge_index", "device", "precision", "seed", "max_new_tokens",
            "max_attention_tokens", "repeats", "warmups"}
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError(f"config requires exactly {sorted(keys)}")
    for key in ("seed", "max_new_tokens", "max_attention_tokens", "repeats", "warmups"):
        if type(config[key]) is not int or config[key] < (0 if key == "seed" else 1):
            raise ValueError(f"invalid {key}")
    if config["max_attention_tokens"] > 2048:
        raise ValueError("attention token safety cap is 2048; quadratic memory must be explicitly reviewed")
    if config["device"] not in {"cpu", "cuda"} or config["precision"] not in {"fp16", "bf16", "fp32"}:
        raise ValueError("invalid device/precision")
    if config["device"] == "cpu" and config["precision"] != "fp32":
        raise ValueError("CPU requires precision=fp32")
    if args.phase == "evaluate" and args.thresholds is None:
        raise ValueError("evaluate requires frozen --thresholds from a calibration run")
    if args.phase != "evaluate" and args.thresholds is not None:
        raise ValueError("--thresholds only belongs to evaluate; do not refit an existing artifact")
    if args.phase == "evaluate":
        completion_path = locate(args.thresholds).parent / "completed.json"
        if not completion_path.is_file():
            raise ValueError("thresholds must belong to a completed calibration run, not a partial/failed run")
        completion = json.loads(completion_path.read_text(encoding="utf-8-sig"))
        if completion.get("status") != "completed" or completion.get("source_verification") != "passed":
            raise ValueError("calibration run has no successful source verification")
    if (root / "experiments/results/algorithms").resolve() not in run.parents:
        raise ValueError("run-dir must be a NEW child of experiments/results/algorithms")
    for key in ("model", "bge", "knowledge_index"):
        if not isinstance(config[key], str) or not locate(config[key]).is_dir():
            raise ValueError(f"{key}: existing LOCAL resource directory required")
        if run == locate(config[key]) or locate(config[key]) in run.parents:
            raise ValueError("run-dir cannot be inside immutable model/index resources")
    for path in ([args.evaluation, args.thresholds] if args.phase == "evaluate" else [args.calibration]):
        hashes[str(locate(path))] = file_hash(locate(path))
    source_cases = load_cases(locate(args.evaluation if args.phase == "evaluate" else args.calibration),
                             "evaluation" if args.phase == "evaluate" else "calibration", args.allow_unreviewed)
    run.mkdir(parents=True, exist_ok=False)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    from experiments.algorithms.shared.a0_adapter import A0Adapter, measured
    write_json(run / "run.json", {"status": "running", "config": config, "phase": args.phase, "mode": args.mode,
                                  "source_hashes_initial": hashes, "purpose": "auxiliary topic/context development research",
                                  "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  "provisional": any(r["review_status"] != "approved" for r in source_cases)})
    stage, current = "loading", None
    wall = time.perf_counter()
    try:
        from .features import TopicFeatures
        from .context import ContextExperiment, POLICY_VERSION, DEPENDENCY_PATTERN
        import transformers
        if transformers.__version__ != "5.16.1":
            raise ValueError("TI implementation reviewed for Transformers 5.16.1 only; no API fallback")
        adapter, load_seconds = measured(A0Adapter, root, config)
        features = TopicFeatures(adapter, config)
        identity = {"a0": adapter.identity, "features": features.definition,
                    "max_attention_tokens": config["max_attention_tokens"],
                    "context_policy": POLICY_VERSION, "dependency_pattern": DEPENDENCY_PATTERN,
                    "effective_attention_backend_all_arms": "eager"}
        identity_id = fingerprint(identity)
        import torch
        write_json(run / "identity.json", {**identity, "identity_id": identity_id,
                                           "loading_seconds": load_seconds,
                                           "hardware": {"cuda_version": torch.version.cuda,
                                                        "gpu": torch.cuda.get_device_name(0) if config["device"] == "cuda" else None,
                                                        "actual_model_dtype": str(next(adapter.llm.model.parameters()).dtype)}})
        for i in range(config["warmups"]):
            append(run / "warmup_features.jsonl", {"warmup": True, "repeat": i,
                   "embedding": features.embedding(source_cases[0]), "attention": features.attention(source_cases[0])})

        def extract(cases, split):
            nonlocal current, stage
            rows = []
            for case in cases:
                stage, current = f"{split}_features", case["scenario_id"]
                row = {"scenario_id": case["scenario_id"], "query": case["query"], "split": split,
                       "review_status": case["review_status"], "gold_shift": case["gold_shift"],
                       "embedding": features.embedding(case), "attention": features.attention(case)}
                append(run / f"{split}_features.jsonl", row)
                rows.append(row)
                print(f"TI {split} {case['scenario_id']}", flush=True)
            return rows

        if args.phase in {"calibrate", "all"}:
            calibration_rows = extract(source_cases, "calibration")
            methods = {}
            for method in ("TI0", *features.definition["layers"]):
                scores = [(r["embedding"] if method == "TI0" else r["attention"])["scores"][method] for r in calibration_rows]
                methods[method] = fit_threshold(scores, [r["gold_shift"] for r in calibration_rows])
            payload = {"schema": "ti_thresholds_v1", "identity_id": identity_id, "methods": methods,
                       "calibration_manifest": split_manifest(source_cases),
                       "calibration_sha256": hashes[str(locate(args.calibration))],
                       "provisional": any(r["review_status"] != "approved" for r in source_cases)}
            frozen = {"payload": payload, "payload_sha256": fingerprint(payload)}
            write_json(run / "thresholds.json", frozen)
            write_json(run / "calibration.snapshot.json", source_cases)
        else:
            frozen = json.loads(locate(args.thresholds).read_text(encoding="utf-8-sig"))
            payload = frozen["payload"]
            if fingerprint(payload) != frozen["payload_sha256"]:
                raise ValueError("frozen threshold checksum changed")
            if payload["schema"] != "ti_thresholds_v1" or payload["identity_id"] != identity_id:
                raise ValueError("threshold model/code/feature settings do not match; no refitting on evaluation")
            methods = payload["methods"]
            if set(methods) != {"TI0", *features.definition["layers"]} or any(
                    not math.isfinite(value["threshold"]) for value in methods.values()):
                raise ValueError("invalid frozen threshold methods/values")
            if payload["provisional"] and not args.allow_unreviewed:
                raise ValueError("calibration labels remain unreviewed; explicit opt-in required")
            write_json(run / "thresholds.used.json", frozen)
        summary = {"calibration": methods, "evaluation": None, "context_reduction": None,
                   "answer_quality_verified": False, "causal_explanation_claim": False}
        if args.phase in {"evaluate", "all"}:
            if args.phase == "all":
                hashes[str(locate(args.evaluation))] = file_hash(locate(args.evaluation))
            evaluation = source_cases if args.phase == "evaluate" else load_cases(locate(args.evaluation), "evaluation", args.allow_unreviewed)
            check_separation(payload["calibration_manifest"], evaluation)
            write_json(run / "evaluation.snapshot.json", evaluation)
            rows = extract(evaluation, "evaluation")
            summary["evaluation"] = detection_report(run, rows, methods)
            if args.mode == "all":
                experiment = ContextExperiment(adapter, features, methods, config)
                context_rows = []
                for index, case in enumerate(evaluation):
                    stage, current = "context_retrieval", case["scenario_id"]
                    (docs, scores), retrieval_seconds = measured(adapter.pipeline.retrieve, case["query"])
                    if len(docs) != 5 or len(scores) != 5:
                        raise ValueError("context comparison requires existing A0 Top-5 documents")
                    write_json(run / f"context_retrieval_{index:04d}.json", {
                        "scenario_id": case["scenario_id"], "retrieval_seconds": retrieval_seconds,
                        "shared_across_arms_and_repeats": True,
                        "documents": [{"content": d.page_content, "metadata": d.metadata, "score": float(s)}
                                      for d, s in zip(docs, scores)]})
                    for warmup, repeats in ((True, config["warmups"]), (False, config["repeats"])):
                        for repeat in range(repeats):
                            arms = ["full", "embedding", "attention"]
                            offset = (index + repeat) % len(arms)
                            for arm in arms[offset:] + arms[:offset]:
                                stage, current = "context_generation", {"scenario_id": case["scenario_id"], "arm": arm, "repeat": repeat}
                                result = experiment.run(case, docs, retrieval_seconds, arm, repeat)
                                result["warmup"] = warmup
                                append(run / "context_results.jsonl", result)
                                context_rows.append(result)
                summary["context_reduction"] = context_report(run, context_rows)
            summary["provisional"] = payload["provisional"] or any(r["review_status"] != "approved" for r in evaluation)
        else:
            summary["provisional"] = payload["provisional"]
        stage = "verify_sources"
        adapter.verify_sources()
        if any(file_hash(Path(path)) != digest for path, digest in hashes.items()):
            raise ValueError("configuration/calibration/evaluation/threshold source changed during run")
        write_json(run / "summary.json", summary)
        write_json(run / "completed.json", {"status": "completed", "provisional": summary["provisional"],
                                            "source_hashes_final": hashes, "source_verification": "passed",
                                            "wall_seconds": time.perf_counter() - wall,
                                            "quality_preservation_verified": False})
    except BaseException as error:
        write_json(run / "failed.json", {"status": "failed", "stage": stage, "scenario": current,
                                        "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
