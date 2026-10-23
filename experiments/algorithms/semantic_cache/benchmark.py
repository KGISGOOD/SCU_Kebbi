"""Sequential development scenarios, not a frozen formal evaluation set."""
from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
import statistics
import traceback


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="NEW directory; never resume or overwrite")
    parser.add_argument("--mode", choices=("none", "exact", "semantic", "all"), default="all")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--scenarios", type=Path, default=Path("experiments/algorithms/semantic_cache/scenarios.jsonl"))
    parser.add_argument("--allow-unreviewed", action="store_true", help="development run only")
    args = parser.parse_args()
    from .cache import normalize_question, NORMALIZATION_VERSION
    root = args.project_root.resolve()
    def locate(path):
        return path.resolve() if path.is_absolute() else (root / path).resolve()
    config_path, scenarios_path = locate(args.config), locate(args.scenarios)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    expected_keys = {"model", "bge", "knowledge_index", "device", "precision", "max_new_tokens", "seed", "threshold"}
    if set(config) != expected_keys:
        raise ValueError(f"config requires exactly {sorted(expected_keys)}")
    if args.threshold is not None:
        config["threshold"] = args.threshold
    if not math.isfinite(config["threshold"]) or not -1 <= config["threshold"] <= 1:
        raise ValueError("threshold must be within [-1, 1]")
    if config["precision"] not in {"fp16", "bf16", "fp32"}:
        raise ValueError("invalid precision")
    for key in ("max_new_tokens", "seed"):
        if type(config[key]) is not int or config[key] < (1 if key == "max_new_tokens" else 0):
            raise ValueError(f"invalid {key}")
    cases = []
    for number, line in enumerate(scenarios_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank scenario line {number}")
        row = json.loads(line)
        for key in ("query_id", "query", "intent_key", "category", "review_status"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"scenario line {number}: invalid {key}")
        if row["review_status"] not in {"approved", "needs_review"}:
            raise ValueError(f"scenario line {number}: unknown review_status")
        for key in ("history", "messages", "context", "session_id", "dialogue_id"):
            if row.get(key):
                raise ValueError(f"scenario line {number}: {key} is unsupported by single-turn Cache")
        cases.append(row)
    if not cases or len({r["query_id"] for r in cases}) != len(cases):
        raise ValueError("empty scenarios or duplicate query IDs")
    labeled_keys = {}
    for case in cases:
        key = normalize_question(case["query"])
        if key in labeled_keys and labeled_keys[key] != case["intent_key"]:
            raise ValueError(f"{case['query_id']}: identical normalized queries have conflicting intent labels; review data")
        labeled_keys[key] = case["intent_key"]
    provisional = any(r["review_status"] != "approved" for r in cases)
    if provisional and not args.allow_unreviewed:
        raise ValueError("scenarios need human review; use --allow-unreviewed only for development")
    # No existing output may be reused, even when resource setup fails.
    run = locate(args.run_dir)
    results_root = (root / "experiments/results/algorithms").resolve()
    if results_root not in run.parents:
        raise ValueError("run-dir must be a NEW child of experiments/results/algorithms")
    if any(run == locate(Path(config[k])) or locate(Path(config[k])) in run.parents
           for k in ("model", "bge", "knowledge_index")):
        raise ValueError("run directory must not be inside an immutable resource")
    run.mkdir(parents=True, exist_ok=False)
    from ..shared.a0_adapter import A0Adapter, file_hash, measured
    from .pipeline import CachedA0
    initial = {"status": "running", "purpose": "development diagnostic, not formal Test",
               "provisional": provisional, "config": config, "mode": args.mode,
               "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "scenarios_sha256": file_hash(scenarios_path), "config_sha256": file_hash(config_path),
               "exact_normalization": NORMALIZATION_VERSION, "semantic_safety_verified": False,
               "scenario_count": len(cases)}
    write(run / "run.json", initial)
    write(run / "scenarios.snapshot.json", cases)
    try:
        adapter = A0Adapter(root, config)
        write(run / "identity.json", adapter.identity)
        # Warm the same A0 before each mode; warmup never populates the cache.
        modes = ("none", "exact", "semantic") if args.mode == "all" else (args.mode,)
        summary = {}
        all_records = {}
        for mode in modes:
            import torch
            torch.manual_seed(config["seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config["seed"])
            _, warmup_seconds = measured(adapter.answer, cases[0]["query"])
            engine = CachedA0(adapter, mode, config["threshold"])
            records, origins = [], {}
            output = run / mode
            output.mkdir(exist_ok=False)
            with (output / "results.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
                stream.write("")  # create a fresh result stream before the first request
            for case in cases:
                try:
                    # Only the query string enters the online pipeline.
                    result = engine.answer(case["query"])
                    result.update(query_id=case["query_id"], category=case["category"], provisional=provisional)
                    result["intent_reuse_proxy"] = None
                    if result["hit"]:
                        origin = origins[result["cache_id"]]
                        result["origin_query_id"] = origin["query_id"]
                        result["intent_reuse_proxy"] = origin["intent_key"] == case["intent_key"]
                    elif result["cache_id"]:
                        origins[result["cache_id"]] = case
                    # intent_reuse_proxy is not human answer-quality verification.
                    write(output / f"{len(records) + 1:04d}.json", result)
                    with (output / "results.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    records.append(result)
                    print(f"{mode} {case['query_id']} hit={result['hit']} "
                          f"seconds={result['timings']['end_to_end_seconds']:.4f}", flush=True)
                except Exception as error:
                    failure = getattr(error, "cache_failure", {"error": str(error)})
                    failure.update(query_id=case["query_id"], traceback=traceback.format_exc())
                    write(output / f"{len(records) + 1:04d}.failed.json", failure)
                    raise
            def average(rows):
                return statistics.mean(r["timings"]["end_to_end_seconds"] for r in rows) if rows else None
            hits = [r for r in records if r["hit"]]
            misses = [r for r in records if not r["hit"]]
            summary[mode] = {"count": len(records), "hit_count": len(hits), "miss_count": len(misses),
                             "hit_rate": len(hits) / len(records), "mean_end_to_end_seconds": average(records),
                             "mean_hit_seconds": average(hits), "mean_miss_seconds": average(misses),
                             "intent_proxy_incompatible_hits": sum(r["intent_reuse_proxy"] is False for r in hits),
                             "intent_proxy_compatible_hits": sum(r["intent_reuse_proxy"] is True for r in hits),
                             "median_end_to_end_seconds": statistics.median(
                                 r["timings"]["end_to_end_seconds"] for r in records),
                             "total_end_to_end_seconds": sum(r["timings"]["end_to_end_seconds"] for r in records),
                             "stage_mean_seconds": {
                                 key: statistics.mean(r["timings"][key] for r in records)
                                 for key in records[0]["timings"]},
                             "warmup_seconds_excluded": warmup_seconds}
            all_records[mode] = records
        adapter.verify_sources()
        if file_hash(scenarios_path) != initial["scenarios_sha256"] or file_hash(config_path) != initial["config_sha256"]:
            raise ValueError("config or scenarios changed during run")
        write(run / "summary.json", {"provisional": provisional, "modes": summary,
                                     "answer_quality_verified": False, "semantic_safety_verified": False})
        if "none" in all_records:
            comparisons = []
            for mode, rows in all_records.items():
                if mode == "none":
                    continue
                for baseline, result in zip(all_records["none"], rows):
                    if baseline["query_id"] != result["query_id"]:
                        raise RuntimeError("mode query sequence mismatch")
                    comparisons.append({
                        "query_id": result["query_id"], "mode": mode, "hit": result["hit"],
                        "answer_equal_to_none": result["answer"] == baseline["answer"],
                        "seconds_saved_vs_none": baseline["timings"]["end_to_end_seconds"]
                                                 - result["timings"]["end_to_end_seconds"],
                        "note": "answer equality is not answer correctness; one fixed-order development pass"})
            write(run / "comparisons.json", comparisons)
        write(run / "completed.json", {"status": "completed", "source_verification": "passed"})
    except Exception:
        write(run / "failed.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
