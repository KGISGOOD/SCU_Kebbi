"""SC1-3: independent live cache trajectories for each threshold and repeat."""
from __future__ import annotations

import argparse
from copy import deepcopy
import datetime
import json
import math
import os
from pathlib import Path
import traceback

from .cache import fingerprint
from .threshold_data import load_sequence, manifest, assert_independent, judge_reuse
from .threshold_reporting import write_json, append, summarize, comparison_csv


DEFAULT_THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]


def parse_thresholds(values):
    if isinstance(values, str):
        values = values.split(",")
    if not isinstance(values, list) or not values:
        raise ValueError("thresholds must be a nonempty list or comma-separated CLI values")
    if any(isinstance(x, bool) for x in values):
        raise ValueError("boolean is not a threshold")
    try:
        result = [float(x) for x in values]
    except (ValueError, TypeError) as error:
        raise ValueError("thresholds must be numeric; fill the selection example before evaluation") from error
    if any(not math.isfinite(t) or not -1 <= t <= 1 for t in result) or len(set(result)) != len(result):
        raise ValueError("thresholds must be finite unique cosine values within [-1, 1]")
    return result


class ScopedAdapter:
    """Share immutable A0 resources, never share Cache objects or entries."""
    def __init__(self, adapter, threshold, repeat):
        self.embedder, self.embed, self.answer = adapter.embedder, adapter.embed, adapter.answer
        self.identity = {**deepcopy(adapter.identity), "sc1_3": {"threshold": threshold, "repeat": repeat}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--thresholds", help="comma separated; CLI > config thresholds > default five")
    parser.add_argument("--split", choices=("functional", "calibration", "evaluation"), default="functional")
    parser.add_argument("--scenarios", type=Path)
    parser.add_argument("--selection-file", type=Path, help="required for independent evaluation; exactly one chosen threshold")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--with-baseline", action="store_true", help="run a real SC0 None pass for each repeat")
    parser.add_argument("--allow-unreviewed", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    def locate(value):
        p = Path(value).expanduser()
        return (p if p.is_absolute() else root / p).resolve()
    from ..shared.a0_adapter import A0Adapter, file_hash, measured
    config_path, run = locate(args.config), locate(args.run_dir)
    hashes = {str(config_path): file_hash(config_path)}
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    required = {"model", "bge", "knowledge_index", "device", "precision", "max_new_tokens", "seed"}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - {"threshold", "thresholds"}:
        raise ValueError("use the existing SC GPU config, optionally adding thresholds; unexpected fields rejected")
    if args.repeats < 1 or any(type(config[k]) is not int or config[k] < (1 if k == "max_new_tokens" else 0)
                              for k in ("seed", "max_new_tokens")):
        raise ValueError("invalid repeats/seed/max_new_tokens")
    if config["device"] not in {"cpu", "cuda"} or config["precision"] not in {"fp16", "bf16", "fp32"}:
        raise ValueError("invalid device/precision")
    if config["device"] == "cpu" and config["precision"] != "fp32":
        raise ValueError("CPU requires precision=fp32")
    thresholds = parse_thresholds(args.thresholds if args.thresholds is not None else config.get("thresholds", DEFAULT_THRESHOLDS))
    source = locate(args.scenarios or Path(f"experiments/algorithms/semantic_cache/threshold_{args.split}.jsonl"))
    hashes[str(source)] = file_hash(source)
    cases = load_sequence(source, args.split, args.allow_unreviewed)
    selection = None
    if args.split == "evaluation":
        if args.selection_file is None:
            raise ValueError("evaluation requires --selection-file; do not sweep the final evaluation set")
        path = locate(args.selection_file)
        hashes[str(path)] = file_hash(path)
        selection = json.loads(path.read_text(encoding="utf-8-sig"))
        if selection.get("schema") != "sc1_3_selection_v1" or not isinstance(selection.get("rationale"), str) or not selection["rationale"].strip():
            raise ValueError("selection needs schema and a human selection rationale")
        chosen = parse_thresholds([selection.get("threshold")])[0]
        if args.thresholds is not None and thresholds != [chosen]:
            raise ValueError("evaluation CLI must match the single frozen selected threshold")
        thresholds = [chosen]
        calibration_run = locate(selection["calibration_run"])
        for name in ("completed.json", "run.json"):
            hashes[str(calibration_run / name)] = file_hash(calibration_run / name)
        completed = json.loads((calibration_run / "completed.json").read_text(encoding="utf-8"))
        previous = json.loads((calibration_run / "run.json").read_text(encoding="utf-8"))
        if (completed.get("status") != "completed" or completed.get("source_verification") != "passed"
                or previous.get("split") != "calibration" or chosen not in previous["thresholds"]):
            raise ValueError("selected threshold must come from a completed SC1-3 calibration run")
        assert_independent(previous["sequence_manifest"], cases)
        if selection.get("review_status") not in {"approved", "needs_review"}:
            raise ValueError("selection review_status missing")
        if selection["review_status"] != "approved" and not args.allow_unreviewed:
            raise ValueError("selection needs human review")
        if previous.get("provisional", True) and not args.allow_unreviewed:
            raise ValueError("calibration provenance is provisional; explicit --allow-unreviewed required")
    elif args.selection_file is not None:
        raise ValueError("selection-file belongs only to evaluation")
    if (root / "experiments/results/algorithms").resolve() not in run.parents:
        raise ValueError("run-dir must be a NEW child of experiments/results/algorithms")
    for key in ("model", "bge", "knowledge_index"):
        if not isinstance(config[key], str) or not config[key].strip() or not locate(config[key]).is_dir():
            raise ValueError(f"{key}: existing LOCAL directory required")
        if run == locate(config[key]) or locate(config[key]) in run.parents:
            raise ValueError("output must not be inside immutable resources")
    run.mkdir(parents=True, exist_ok=False)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    provisional = any(r["review_status"] != "approved" or any(j["review_status"] != "approved" for j in r.get("reuse_judgments", [])) for r in cases)
    provisional = provisional or (selection is not None and
                                  (selection["review_status"] != "approved" or previous.get("provisional", True)))
    write_json(run / "run.json", {"experiment": "SC1-3", "status": "running", "split": args.split,
                                  "thresholds": thresholds, "config": config, "repeats": args.repeats,
                                  "with_baseline": args.with_baseline, "sequence_manifest": manifest(cases),
                                  "source_hashes": hashes, "provisional": provisional,
                                  "formal_research_result": False, "selection": selection,
                                  "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    write_json(run / "scenarios.snapshot.json", cases)
    stage, current = "loading", None
    try:
        from .pipeline import CachedA0
        import torch
        adapter, loading_seconds = measured(A0Adapter, root, config)
        resource_id = fingerprint(adapter.identity)
        if selection is not None:
            old_identity_path = calibration_run / "identity.json"
            hashes[str(old_identity_path)] = file_hash(old_identity_path)
            old_identity = json.loads(old_identity_path.read_text(encoding="utf-8"))
            if old_identity["a0_identity_id"] != resource_id:
                raise ValueError("evaluation model/code/Prompt/generation settings differ from selected calibration")
        write_json(run / "identity.json", {"a0": adapter.identity, "a0_identity_id": resource_id,
                                           "loading_seconds_excluded": loading_seconds,
                                           "cache_isolation": "fresh object and threshold/repeat-specific namespace",
                                           "safety": "pair labels are offline only; generated answers remain unverified"})
        grouped = {i: [] for i in range(len(thresholds))}
        baseline = []
        for repeat in range(args.repeats):
            # Rotate threshold execution order across repeats, preserving query order.
            indices = list(range(len(thresholds)))
            offset = repeat % len(indices)
            plan = ([None] if args.with_baseline else []) + indices[offset:] + indices[:offset]
            for index in plan:
                threshold = thresholds[index] if index is not None else None
                tag = f"T{index + 1:02d}" if index is not None else "SC0"
                output = run / tag / f"repeat_{repeat:03d}"
                output.mkdir(parents=True, exist_ok=False)
                stage, current = "warmup", {"threshold": threshold, "repeat": repeat}
                torch.manual_seed(config["seed"])
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(config["seed"])
                _, warmup_seconds = measured(adapter.answer, cases[0]["question"])
                scoped = ScopedAdapter(adapter, threshold, repeat)
                engine = CachedA0(scoped, "semantic" if threshold is not None else "none", threshold if threshold is not None else 0.75)
                records, origins = [], {}
                for case in cases:
                    stage, current = "query", {"threshold": threshold, "repeat": repeat, "scenario_id": case["scenario_id"]}
                    result = engine.answer(case["question"])
                    origin = origins.get(result["cache_id"]) if result["hit"] else None
                    if result["hit"] and origin is None:
                        raise RuntimeError("hit has no actual insertion origin")
                    safety = judge_reuse(case, origin)
                    labels = {cache_id: label for label, cache_id in engine.cache.labels.items()}
                    result.update(experiment="SC1-3" if threshold is not None else "SC0", scenario_id=case["scenario_id"],
                                  question=case["question"], review_status=case["review_status"],
                                  query_order=case["query_order"], intent_group=case["intent_group"], split=args.split,
                                  repeat_index=repeat, provisional=provisional, reuse_safety=safety,
                                  origin_scenario_id=origin["scenario_id"] if origin else None,
                                  expected_answer_reuse=case["expected_answer_reuse"], reference_scenario_id=case["reference_scenario_id"],
                                  candidate_faiss_label=labels.get(result["candidate_cache_id"]),
                                  cache_entry_faiss_label=labels.get(result["cache_id"]),
                                  hit_kind=("exact_text_semantic_hit" if result["hit"] and result["answer_source_query"] == case["question"] else
                                            "nonexact_semantic_hit" if result["hit"] else "miss"),
                                  answer_quality_review={"correct": None, "safe_to_reuse": None, "reviewer": None, "notes": None})
                    if not result["hit"] and result["cache_id"] and result["cache_id"] not in origins:
                        origins[result["cache_id"]] = case
                    append(output / "queries.jsonl", result)
                    records.append(result)
                    print(f"SC1-3 {tag} threshold={threshold} repeat={repeat} {case['scenario_id']} hit={result['hit']}", flush=True)
                engine.cache.check_consistency()
                write_json(output / "cache_id_mapping.json", {"namespace": engine.cache.namespace,
                           "faiss_label_to_cache_id": engine.cache.labels,
                           "cache_id_to_origin": {key: value["scenario_id"] for key, value in origins.items()},
                           "warmup_seconds_excluded": warmup_seconds})
                if index is None:
                    baseline.extend(records)
                else:
                    grouped[index].extend(records)
        stage = "verify_sources"
        adapter.verify_sources()
        if any(file_hash(Path(path)) != digest for path, digest in hashes.items()):
            raise ValueError("configuration/sequence/selection/calibration provenance changed during run")
        summaries = []
        for index, rows in grouped.items():
            summary = summarize(rows, baseline)
            summary.update(provisional=provisional, split=args.split, formal_research_result=False)
            write_json(run / f"T{index + 1:02d}" / "summary.json", summary)
            summaries.append(summary)
        comparison_csv(run / "threshold_comparison.csv", summaries)
        write_json(run / "summary.json", {"thresholds": summaries, "provisional": provisional,
                                           "formal_research_result": False, "answer_quality_verified": False})
        write_json(run / "completed.json", {"status": "completed", "source_verification": "passed",
                                             "source_hashes_final": hashes, "thresholds": thresholds,
                                             "provisional": provisional, "formal_research_result": False})
    except BaseException as error:
        failure = getattr(error, "cache_failure", {})
        write_json(run / "failed.json", {"status": "failed", "stage": stage, "measurement": current,
                                        "error": str(error), "cache_failure": failure, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
