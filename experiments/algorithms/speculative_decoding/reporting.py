"""Postprocess real measurements; never synthesize missing baselines."""
from __future__ import annotations

import csv
import json
import statistics


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def append_jsonl(path, value):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def token_difference(row, ref, tokenizer=None):
    left, right = ref["output_token_ids"], row["output_token_ids"]
    index = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
    if left == right:
        return None
    start, stop = max(0, index - 5), index + 6
    def window(ids):
        return [{"index": i, "id": ids[i],
                 "token": tokenizer.convert_ids_to_tokens(ids[i]) if tokenizer else None,
                 "decoded_token": tokenizer.decode([ids[i]], skip_special_tokens=False) if tokenizer else None}
                for i in range(start, min(stop, len(ids)))]
    return {"first_different_new_token_index": index, "absolute_sequence_index": row["input_token_count"] + index,
            "sd0_length": len(left), "method_length": len(right), "sd0_sha256": ref["output_token_sha256"],
            "method_sha256": row["output_token_sha256"], "sd0_stop_reason": ref["stop_reason"],
            "method_stop_reason": row["stop_reason"], "sd0_window": window(left), "method_window": window(right),
            "sd0_measurement_id": ref["measurement_id"], "method_measurement_id": row["measurement_id"],
            "note": "Index is zero-based; end-of-sequence versus another token is a strict mismatch."}


def report(run, rows, identity, provisional, tokenizer=None):
    baseline = {(r["scenario_id"], r["repeat_index"], r["warmup"]): r
                for r in rows if r["experiment_version"] == "SD0"}
    if len(baseline) != sum(r["experiment_version"] == "SD0" for r in rows):
        raise ValueError("duplicate paired SD0 key; comparison would be ambiguous")
    for row in rows:
        ref = baseline[(row["scenario_id"], row["repeat_index"], row["warmup"])]
        if ref["prompt_identity"] != row["prompt_identity"] or ref["retrieval_identity"] != row["retrieval_identity"]:
            raise ValueError("baseline and assisted inputs differ")
        if ref["target_identity"] != row["target_identity"] or ref["seed"] != row["seed"]:
            raise ValueError("paired Target identity or seed differs")
        row["exact_token_match_sd0"] = row["output_token_ids"] == ref["output_token_ids"]
        row["status"] = "success" if row["exact_token_match_sd0"] else "failed_token_mismatch"
        row["error"] = None if row["exact_token_match_sd0"] else "token IDs differ from paired SD0"
        row["baseline_measurement_id"] = ref["measurement_id"]
        row["token_difference"] = token_difference(row, ref, tokenizer)
        row["first_different_new_token_index"] = (row["token_difference"] or {}).get("first_different_new_token_index")
        row["sd0_output_length"] = len(ref["output_token_ids"])
        row["sd0_output_token_sha256"] = ref["output_token_sha256"]
        row["sd0_stop_reason"] = ref["stop_reason"]
        row["generation_speedup_sd0"] = ref["generation_seconds"] / row["generation_seconds"]
        row["end_to_end_speedup_sd0"] = ref["end_to_end_seconds"] / row["end_to_end_seconds"]
    # Greedy target itself must also be repeatable. Separate this finding from
    # a same-repeat assisted mismatch instead of assuming a seed proves it.
    seen = {}
    nondeterministic = set()
    for row in rows:
        if row["experiment_version"] == "SD0":
            key = (row["scenario_id"], row["warmup"])
            if key in seen and seen[key] != row["output_token_ids"]:
                nondeterministic.add(key)
            seen[key] = row["output_token_ids"]
    for row in rows:
        if (row["scenario_id"], row["warmup"]) in nondeterministic:
            row["status"] = "failed_sd0_not_repeatable"
            row["error"] = "greedy SD0 token output changed across repetitions"
        row["performance_eligible"] = row["status"] == "success"
        if not row["performance_eligible"]:
            row["generation_speedup_sd0"] = None
            row["end_to_end_speedup_sd0"] = None
    output = run / "results.jsonl"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    fields = ["measurement_id", "scenario_id", "experiment_version", "question", "repeat_index", "warmup",
              "status", "exact_token_match_sd0", "performance_eligible", "input_token_count", "generated_token_count",
              "retrieval_seconds", "shared_preparation_seconds", "generation_seconds", "end_to_end_seconds",
              "tokens_per_second", "generation_speedup_sd0", "end_to_end_speedup_sd0",
              "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "assistant_candidate_limit",
              "prompt_lookup_candidate_limit", "max_matching_ngram_size",
              "prompt_identity", "retrieval_identity", "target_identity", "draft_identity", "output_text",
              "output_token_sha256", "residency", "fallback", "error", "baseline_measurement_id",
              "stop_reason", "first_different_new_token_index", "sd0_output_length",
              "sd0_output_token_sha256", "sd0_stop_reason", "token_difference"]
    with (run / "comparison.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    modes = {}
    for version in sorted({r["experiment_version"] for r in rows}):
        measured = [r for r in rows if not r["warmup"] and r["experiment_version"] == version]
        times = [r["generation_seconds"] for r in measured]
        eligible = [r for r in measured if r["performance_eligible"]]
        ordered = sorted(times)
        modes[version] = {"count": len(measured),
                          "token_match_count": sum(r["exact_token_match_sd0"] for r in measured),
                          "failed_count": sum(r["status"] != "success" for r in measured),
                          "mean_generation_seconds": statistics.mean(times),
                          "median_generation_seconds": statistics.median(times),
                          "p95_generation_seconds_nearest_rank": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)],
                          "aggregate_tokens_per_second": sum(r["generated_token_count"] for r in measured) / sum(times),
                          "speedup_eligible_count": len(eligible),
                          "mean_paired_generation_speedup": statistics.mean(r["generation_speedup_sd0"] for r in eligible) if eligible else None,
                          "mean_paired_controlled_e2e_speedup": statistics.mean(r["end_to_end_speedup_sd0"] for r in eligible) if eligible else None,
                          "acceptance_rate": None, "acceptance_status": "unavailable"}
    failures = sum(r["status"] != "success" for r in rows)
    summary = {"status": "failed" if failures else "completed", "provisional": provisional,
               "failed_measurements_including_warmup": failures, "modes": modes,
               "warmup_measurements_excluded_from_metrics": sum(r["warmup"] for r in rows),
               "unique_shared_preparation_seconds": sum({r["scenario_id"]: r["shared_preparation_seconds"]
                                                         for r in rows}.values()),
               "baseline_residency": identity["residency"],
               "speedup_definition": {"generation": "paired SD0 generation_seconds / mode generation_seconds",
                                      "end_to_end": "paired SD0 controlled e2e / mode controlled e2e"},
               "shared_preparation_policy": "one measured retrieval and prompt per scenario, not per mode/repeat",
               "answer_quality_verified": False, "lossless_scope": "exact tokens on these inputs only, not universal proof"}
    write_json(run / "summary.json", summary)
    return failures
