"""SC1-3 summaries from actual sequential measurements, not score replay."""
from __future__ import annotations

import csv
import json
import statistics


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def append(path, row):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def summarize(rows, baselines):
    hits = [r for r in rows if r["hit"]]
    reviewed = [r for r in hits if r["reuse_safety"]["false_reuse"] is not None]
    false = sum(r["reuse_safety"]["false_reuse"] is True for r in reviewed)
    compatible = sum(r["reuse_safety"]["false_reuse"] is False for r in reviewed)
    means = {key: statistics.mean(r["timings"][key] for r in rows) for key in rows[0]["timings"]}
    total = sum(r["timings"]["end_to_end_seconds"] for r in rows)
    repeats = {x["repeat_index"] for x in rows}
    ref = [r for r in baselines if r["repeat_index"] in repeats]
    if ref and sorted((r["repeat_index"], r["query_order"], r["question"]) for r in ref) != sorted(
            (r["repeat_index"], r["query_order"], r["question"]) for r in rows):
        raise ValueError("SC0 and threshold query/repeat sets differ; cannot calculate speedup")
    baseline_total = sum(r["timings"]["end_to_end_seconds"] for r in ref) if ref else None
    return {"threshold": rows[0]["threshold"], "mode": rows[0]["mode"], "total_queries": len(rows),
            "unique_scenarios": len({r["scenario_id"] for r in rows}),
            "repeat_count": len({r["repeat_index"] for r in rows}),
            "cache_hit_count": len(hits), "cache_miss_count": len(rows) - len(hits),
            "hit_rate": len(hits) / len(rows),
            "exact_hit_count": sum(r["hit_kind"] == "exact_text_semantic_hit" for r in hits),
            "semantic_hit_count": len(hits),
            "nonexact_semantic_hit_count": sum(r["hit_kind"] == "nonexact_semantic_hit" for r in hits),
            "reviewed_pair_hit_count": len(reviewed), "pair_compatible_hit_count": compatible,
            "false_reuse_count": false if reviewed else None,
            "unverified_hit_count": len(hits) - len(reviewed),
            "answer_reuse_safety_status": "confirmed_pair_incompatibility_detected" if false else "unverified",
            "answer_quality_verified": False,
            "mean_latency_seconds": means["end_to_end_seconds"],
            "median_latency_seconds": statistics.median(r["timings"]["end_to_end_seconds"] for r in rows),
            "total_latency_seconds": total, "mean_stage_seconds": means,
            "sc0_total_seconds": baseline_total,
            "sequence_speedup_vs_sc0": baseline_total / total if baseline_total is not None else None,
            "speedup_definition": "sum SC0 end-to-end seconds / sum threshold end-to-end seconds; same sequences/repeats",
            "quality_preserving_speedup_verified": False}


def comparison_csv(path, summaries):
    fields = ["threshold", "total_queries", "cache_hit_count", "cache_miss_count", "hit_rate", "exact_hit_count",
              "semantic_hit_count", "nonexact_semantic_hit_count", "reviewed_pair_hit_count", "false_reuse_count",
              "unverified_hit_count", "mean_latency_seconds", "median_latency_seconds", "total_latency_seconds",
              "cache_embedding_seconds", "cache_lookup_seconds", "retrieval_seconds", "generation_seconds",
              "sequence_speedup_vs_sc0", "answer_reuse_safety_status", "answer_quality_verified"]
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for summary in summaries:
            writer.writerow({**summary, **summary["mean_stage_seconds"]})
