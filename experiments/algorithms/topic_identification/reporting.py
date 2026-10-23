"""Write raw results, confusion metrics and human quality-review fields."""
from __future__ import annotations

import csv
import json
import statistics
import time
from .metrics import classification


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def append(path, value):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def detection_report(run, feature_rows, thresholds):
    output, summary = [], {}
    for raw in feature_rows:
        for method, score in {**raw["embedding"]["scores"], **raw["attention"]["scores"]}.items():
            threshold = thresholds[method]["threshold"]
            data = raw["embedding"] if method == "TI0" else raw["attention"]
            start = time.perf_counter()
            prediction = int(score < threshold)
            threshold_seconds = time.perf_counter() - start
            row = {"scenario_id": raw["scenario_id"], "split": raw["split"], "method": method,
                   "question": raw["query"], "review_status": raw["review_status"],
                   "gold_shift": raw["gold_shift"], "score": score, "threshold": threshold,
                   "prediction_shift": prediction, "threshold_decision_seconds": threshold_seconds,
                   "detection_seconds": data["detection_seconds"] + threshold_seconds,
                   "forward_seconds": data.get("forward_seconds"),
                   "timing_scope": "independent BGE call" if method == "TI0" else
                                   "one full prefill shared by all four attention layer groups; do not sum group times"}
            append(run / "classification.jsonl", row)
            output.append(row)
    for method in thresholds:
        rows = [r for r in output if r["method"] == method]
        summary[method] = classification([r["gold_shift"] for r in rows], [r["prediction_shift"] for r in rows])
        summary[method]["mean_detection_seconds"] = statistics.mean(r["detection_seconds"] for r in rows)
        summary[method]["threshold"] = thresholds[method]["threshold"]
    with (run / "classification.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    return summary


def context_report(run, rows):
    measured = [r for r in rows if not r["warmup"]]
    fields = ["scenario_id", "arm", "repeat_index", "warmup", "question", "answer", "prediction_shift",
              "original_prompt_tokens", "reduced_prompt_tokens", "prompt_token_reduction",
              "topic_detection_seconds", "retrieval_seconds", "generation_seconds", "end_to_end_seconds",
              "annotated_required_context_retained", "quality_verified"]
    with (run / "context_comparison.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for arm in ("full", "embedding", "attention"):
        values = [r for r in measured if r["arm"] == arm]
        summary[arm] = {"count": len(values), "mean_topic_detection_seconds": statistics.mean(r["topic_detection_seconds"] for r in values),
                        "mean_generation_seconds": statistics.mean(r["generation_seconds"] for r in values),
                        "mean_controlled_end_to_end_seconds": statistics.mean(r["end_to_end_seconds"] for r in values),
                        "mean_prompt_token_reduction": statistics.mean(r["prompt_token_reduction"] for r in values),
                        "annotated_required_context_removed_count": sum(not r["annotated_required_context_retained"] for r in values),
                        "answer_quality": "unreviewed; no quality-preserving speedup claim"}
    return summary
