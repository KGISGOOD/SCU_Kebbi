"""True independent E2E only. Never mix SD controlled sums with online time."""
import csv
import statistics

from experiments.algorithms.speculative_decoding.reporting import write_json, append_jsonl


def report(run, rows, provisional):
    baselines = {(r["scenario_id"], r["repeat_index"], r["warmup"]): r
                 for r in rows if r["system_variant"] == "INT0"}
    failures = 0
    for row in rows:
        baseline = baselines.get((row["scenario_id"], row["repeat_index"], row["warmup"]))
        if row["cache_hit"]:
            pass  # A reused answer is NOT a lossless decoding comparison.
        elif baseline is None:
            row["token_comparison_status"] = "unavailable_no_paired_int0"
        elif (row["prompt_identity"], row["retrieval_identity"]) != (baseline["prompt_identity"], baseline["retrieval_identity"]):
            row.update(status="failed_input_mismatch", error="independent retrieval/prompt differs from INT0",
                       token_comparison_status="failed_input_mismatch")
        else:
            matches = row["output_token_ids"] == baseline["output_token_ids"]
            row["exact_token_match_int0"] = matches
            row["token_comparison_status"] = "passed" if matches else "failed_token_mismatch"
            if not matches:
                row.update(status="failed_token_mismatch", error="output differs from paired INT0 greedy tokens")
    # Do not mistake seed equality for Target repeatability.
    seen, unstable = {}, set()
    for row in rows:
        if row["system_variant"] == "INT0":
            key = (row["scenario_id"], row["warmup"])
            if key in seen and seen[key] != row["output_token_ids"]:
                unstable.add(key)
            seen[key] = row["output_token_ids"]
    for row in rows:
        if (row["scenario_id"], row["warmup"]) in unstable:
            row.update(status="failed_int0_not_repeatable", error="INT0 tokens changed across repeats")
        row["paired_online_speedup_int0"] = None
        baseline = baselines.get((row["scenario_id"], row["repeat_index"], row["warmup"]))
        if baseline and row["status"] == baseline["status"] == "success":
            row["paired_online_speedup_int0"] = baseline["end_to_end_seconds"] / row["end_to_end_seconds"]
        failures += row["status"] != "success"
        append_jsonl(run / "results.jsonl", row)

    fields = ["measurement_id", "scenario_id", "query_order", "question", "system_variant", "decoder",
              "repeat_index", "warmup", "status", "cache_hit", "similarity", "cache_id", "reused_from_scenario",
              "generated_token_count", "input_token_count", "end_to_end_seconds", "paired_online_speedup_int0",
              "exact_token_match_int0", "token_comparison_status", "prompt_identity", "target_identity", "draft_identity",
              "output_token_sha256", "answer", "fallback", "error"]
    timing_fields = ["cache_embedding_seconds", "cache_lookup_seconds", "cache_update_seconds", "retrieval_seconds",
                     "prompt_check_seconds", "generation_seconds"]
    with (run / "comparison.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + timing_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **row["timings"]})
    modes = {}
    measured = [r for r in rows if not r["warmup"]]
    baseline_total = sum(r["end_to_end_seconds"] for r in measured if r["system_variant"] == "INT0")
    for variant in sorted({r["system_variant"] for r in measured}):
        group = [r for r in measured if r["system_variant"] == variant]
        times = [r["end_to_end_seconds"] for r in group]
        hits = sum(r["cache_hit"] for r in group)
        comparable = all(r["paired_online_speedup_int0"] is not None for r in group)
        modes[variant] = {"decoder": group[0]["decoder"], "queries": len(group), "hits": hits,
                          "misses": len(group) - hits, "hit_rate": hits / len(group),
                          "total_online_seconds": sum(times), "mean_online_seconds": statistics.mean(times),
                          "median_online_seconds": statistics.median(times),
                          "speedup_int0_total_online": baseline_total / sum(times) if comparable else None,
                          "speedup_is_quality_conditioned": False,
                          "false_reuse_count_confirmed_pairs": sum(r["answer_reuse_safety"]["false_reuse"] is True for r in group),
                          "unverified_pair_hits": sum(r["cache_hit"] and r["answer_reuse_safety"]["status"] == "unverified" for r in group),
                          "answer_quality_unverified_hits": hits,
                          "answer_quality_verified": False,
                          "generated_tokens": sum(r["generated_token_count"] for r in group),
                          "stage_totals_seconds": {key: sum(r["timings"][key] for r in group) for key in timing_fields}}
    write_json(run / "summary.json", {
        "status": "failed" if failures else "completed", "provisional": provisional,
        "variants": modes, "failed_measurements_including_warmups": failures,
        "warmups_excluded": sum(r["warmup"] for r in rows),
        "timing_policy": "each request independently executes its own cache/RAG/generation; no shared-preparation sum",
        "speedup_definition": "sum of paired INT0 online E2E / sum of variant online E2E, measured queries only",
        "safety_note": "timing ratios are descriptive, not quality-preserving speedup claims; human answer review required",
        "standalone_note": "without INT0, token equivalence and speedups are unavailable; run all to compare"})
    return failures
