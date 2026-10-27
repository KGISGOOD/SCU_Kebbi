"""Evidence-only projections. Missing annotations never become correctness scores."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
import os
import statistics
import uuid

from .common import Records, read, save, sha


def stats(values):
    values = sorted(x for x in values if isinstance(x, (int, float)) and math.isfinite(x))
    return {"n": len(values), "mean_seconds": statistics.mean(values) if values else None,
            "p50_seconds": statistics.median(values) if values else None,
            "p95_seconds_nearest_rank": values[max(0, math.ceil(.95 * len(values)) - 1)] if values else None}


def expected(job, data, config):
    from .data import select
    repeats = config["repeats"]
    if job["kind"] == "qa":
        label = "FAQ比對" if job["architecture"] in {"A1", "A2-1", "A2-2"} else "LoRA回答" if job["architecture"] == "A3" else "RAG問答"
        return len({r["id"] for r in select(data, label) + select(data, "問答安全/不確定性")}) * repeats
    if job["kind"] == "sc":
        return len(select(data, "語意快取")) * 2 * repeats
    if job["kind"] == "sd":
        return len(select(data, "推測解碼候選")) * (6 if job["draft"] else 5) * repeats
    if job["kind"] == "int":
        return len(select(data, "語意快取")) * 2 * 4 * repeats
    return 0  # actual confirmed conversation population is listed separately


def derive(records):
    """Project comparisons without changing source evidence or Guard standards."""
    data = deepcopy(records)
    baseline, repeated = {}, defaultdict(set)
    for r in data:
        if r.get("experiment") == "SD0" and r.get("output_token_ids"):
            key = (r["job"], r["question_id"], r["repeat"])
            baseline[key] = r
            repeated[(r["job"], r["question_id"])].add(tuple(r["output_token_ids"]))
    for r in data:
        if not r.get("experiment", "").startswith("SD"):
            continue
        ref = baseline.get((r["job"], r["question_id"], r["repeat"]))
        if ref and r.get("output_token_ids"):
            equal = r["output_token_ids"] == ref["output_token_ids"]
            r["exact_token_match_sd0"] = equal
            r["performance_eligible"] = equal and r["status"] == "success"
            if len(repeated[(r["job"], r["question_id"])]) > 1:
                r.update(status="failed_sd0_not_repeatable", performance_eligible=False,
                         error="paired SD0 changed across repeats")
            if not equal:
                r.update(status="failed_token_mismatch", performance_eligible=False)
            r["generation_speedup_sd0"] = ref["generation_seconds"] / r["generation_seconds"] if r["performance_eligible"] else None
            r["controlled_e2e_speedup_sd0"] = ref["end_to_end_seconds"] / r["end_to_end_seconds"] if r["performance_eligible"] else None
    int_baselines = {(r["job"], r["question_id"], r["repeat"], r["phase"]): r for r in data
                     if r.get("experiment") == "INT0" and r["status"] == "success"}
    sc_baselines = {(r["question_id"], r["repeat"], r["phase"]): r for r in data
                    if r.get("experiment") == "SC-none" and r["status"] == "success"}
    for r in data:
        ref = (int_baselines.get((r["job"], r["question_id"], r["repeat"], r["phase"]))
               if r.get("experiment", "").startswith("INT") else
               sc_baselines.get((r.get("question_id"), r.get("repeat"), r.get("phase")))
               if r.get("experiment", "").startswith("SC-") else None)
        if ref and r["status"] == "success":
            r["paired_online_speedup"] = latency(ref) / latency(r) if latency(r) else None
            if r.get("experiment") in {"INT2", "INT3"} and not r.get("cache_hit"):
                equal = (r["output_token_ids"] == ref["output_token_ids"]
                         and r["prompt_identity"] == ref["prompt_identity"]
                         and r["retrieval_identity"] == ref["retrieval_identity"])
                r["exact_token_match_int0"] = equal
                if not equal:
                    r.update(status="failed_int_token_mismatch", error="uncached decoder differs from same-repeat/phase INT0")
        if r.get("cache_hit", r.get("hit")):
            r["lossless_claim"] = "not_applicable_answer_reuse_quality_pending"
    return data


def latency(r):
    return r.get("end_to_end_seconds", r.get("timings", {}).get("end_to_end_seconds"))


def generate(run, root):
    from experiments.research_tools.reporting import workbook
    config, questions = read(run / "raw/config.json"), read(run / "raw/questions.snapshot.json")
    audit = read(run / "raw/data_audit.json")
    annotations = read(run / "raw/annotations.snapshot.json")
    plan = read(run / "raw/plan.json")
    stages = read(run / "raw/stages.json") if (run / "raw/stages.json").exists() else {}
    evidence, warmups, result_hashes = [], [], {}
    for job in plan:
        directory = run / "raw/jobs" / job["id"] / "records"
        if not directory.exists():
            continue
        store = Records(directory)
        result_hashes[str(store.pointer.relative_to(run))] = sha(store.pointer)
        for r in store.all():
            r = dict(r, job=job["id"])
            if r.get("warmup") or r.get("phase") == "smoke":
                warmups.append(r)
            elif r.get("phase") in {"measured", "cold", "warm"}:
                evidence.append(r)
    evidence = derive(evidence)
    group, by_type = defaultdict(list), defaultdict(list)
    table, failures, quality = [], [], []
    cache_labels = {(p["query_id"], p["origin_id"]): p for p in annotations.get("cache_pairs", [])}
    origin_queries = defaultdict(list)
    for q in questions:
        origin_queries[q["question"]].append(q["id"])
    for r in evidence:
        group[(r["job"], r.get("experiment"), r["phase"], r.get("cohort"))].append(r)
        by_type[(r["job"], r.get("experiment"), r["phase"], r.get("type"))].append(r)
        answer = r.get("answer", r.get("output_text", r.get("output", {}).get("generated_answer")))
        annotation = annotations.get("questions", {}).get(r["question_id"], {})
        source = r.get("answer_source_query", r.get("query"))
        origins = origin_queries.get(source, [])
        pair = cache_labels.get((r["question_id"], r.get("reused_from_scenario")))
        if pair is None and len(origins) == 1:
            pair = cache_labels.get((r["question_id"], origins[0]))
        hit = r.get("cache_hit", r.get("hit"))
        r["false_reuse"] = (not pair["reusable"]) if hit and pair else None
        output = r.get("output", {})
        gold_faq = annotation.get("gold_faq_id")
        faq_correct = output.get("predicted_faq_id") == gold_faq if gold_faq else None
        top_faqs = [f["faq_id"] for f in output.get("faq_top2", [])]
        faq_rr = 1 / (top_faqs.index(gold_faq) + 1) if gold_faq in top_faqs else 0.0 if gold_faq and top_faqs else None
        retrieved = output.get("retrieved_documents", [])
        retrieved_ids = [d.get("document_id", d.get("metadata", {}).get("document_id")) for d in retrieved]
        gold_documents = annotation.get("gold_document_ids", [])
        recall = len(set(retrieved_ids) & set(gold_documents)) / len(set(gold_documents)) if gold_documents and retrieved else None
        first_rank = next((i for i, docid in enumerate(retrieved_ids, 1) if docid in gold_documents), None)
        table.append({"Question ID": r["question_id"], "問題": r.get("question"), "題型": r.get("type"),
                      "Job": r["job"], "Mode": r.get("experiment"), "Phase": r["phase"],
                      "Cohort": r.get("cohort"), "Repeat": r["repeat"], "Status": r["status"],
                      "答案": answer, "E2E seconds": latency(r),
                      "Generation seconds": r.get("generation_seconds", output.get("generation_seconds", r.get("timings", {}).get("generation_seconds"))),
                      "Tokens per second": r.get("tokens_per_second"),
                      "Generated tokens": r.get("generated_token_count", output.get("output_tokens")),
                      "Cache Hit": hit, "Similarity": r.get("similarity", output.get("similarity_score")),
                      "False reuse verified pair": r["false_reuse"],
                      "Exact tokens SD0": r.get("exact_token_match_sd0"),
                      "Exact tokens INT0": r.get("exact_token_match_int0"),
                      "Eligible generation speedup": r.get("generation_speedup_sd0"),
                      "Online speedup quality pending": r.get("paired_online_speedup"),
                      "Predicted FAQ": output.get("predicted_faq_id"), "Gold FAQ": gold_faq,
                      "FAQ correct": faq_correct, "Answer quality": "pending_human_scoring",
                      "FAQ MRR at returned Top2": faq_rr,
                      "Retrieval Recall at returned K": recall,
                      "Retrieval RR": 1 / first_rank if first_rank else 0.0 if gold_documents and retrieved else None,
                      "Reference exact text match": answer == annotation["gold_answer"] if annotation.get("gold_answer") is not None else None,
                      "Error": r.get("error"), "Raw evidence": "raw/jobs/" + r["job"] + "/records/index.json"})
        if r["status"] != "success":
            failures.append({"job": r["job"], "id": r["question_id"], "status": r["status"], "error": r.get("error")})
        if r.get("experiment") in {"A0", "A1", "A2-1", "A2-2", "A3", "A4-1", "A4-2", "A5"}:
            quality.append({"id": r["question_id"], "architecture": r["experiment"], "repeat": r["repeat"],
                            "answer": answer, "reference": annotation.get("gold_answer"),
                            "source_ids": annotation.get("source_ids"), "status": "pending_answer_quality_scoring"})
    comparison = []
    for (job, mode, phase, cohort), records in group.items():
        successful = [r for r in records if r["status"] == "success"]
        # Real measured latency remains visible even when strict token parity failed.
        # Only the lossless speedup subset excludes such measurements.
        times = [latency(r) for r in records if r["status"] == "success" or r.get("output_token_ids")]
        sd = [r for r in records if r.get("output_token_ids") and r.get("experiment", "").startswith("SD")]
        eligible = [r for r in sd if r.get("performance_eligible")]
        hits = [r for r in successful if r.get("cache_hit", r.get("hit")) is True]
        faq = [t for t in table if t["Job"] == job and t["Mode"] == mode and t["Phase"] == phase
               and t["Cohort"] == cohort and t["FAQ correct"] is not None and t["Status"] == "success"]
        comparison.append({"Job": job, "Mode": mode, "Phase": phase, "Cohort": cohort,
                           "Recorded": len(records), "Successful": len(successful), "Failed": len(records) - len(successful),
                           **stats(times), "Hit count": len(hits), "Hit rate": len(hits) / len(successful) if successful and any("hit" in r or "cache_hit" in r for r in successful) else None,
                           "Verified false reuse": sum(r["false_reuse"] is True for r in hits) if any(r["false_reuse"] is not None for r in hits) else None,
                           "Verified reuse denominator": sum(r["false_reuse"] is not None for r in hits),
                           "Token match rate generated": sum(bool(r.get("exact_token_match_sd0")) for r in sd) / len(sd) if sd else None,
                           "Lossless-eligible measured subset": len(eligible),
                           "Mean paired gen speedup eligible only": statistics.mean(r["generation_speedup_sd0"] for r in eligible) if eligible else None,
                           "Mean paired controlled E2E speedup eligible only": statistics.mean(r["controlled_e2e_speedup_sd0"] for r in eligible) if eligible else None,
                           "Mean generation seconds all measured": statistics.mean(r["generation_seconds"] for r in sd) if sd else None,
                           "Timing includes valid token-failed generations": bool(sd),
                           "Full lossless status": "not_passed" if any(not r.get("performance_eligible") for r in sd) else "measured_inputs_only" if sd else "not_applicable",
                           "FAQ accuracy labeled only": sum(r["FAQ correct"] for r in faq) / len(faq) if faq else None,
                           "FAQ labeled denominator": len(faq), "Answer correctness": "pending_manual_scoring"})
    type_stats = [{"Job": k[0], "Mode": k[1], "Phase": k[2], "題型": k[3], "records": len(v),
                   **stats([latency(r) for r in v if r["status"] == "success"])} for k, v in by_type.items()]
    coverage = []
    for q in questions:
        matches = [r for r in evidence if r["question_id"] == q["id"]]
        coverage.append({"id": q["id"], "原始題目": q["question"], "type": q["type"],
                         "allocation": q["suites"], "conversation_id": q["conversation_id"], "turn": q["turn_index"],
                         "recorded_measurements": len(matches), "source_review": q["review_status"],
                         "multi_turn_status": "pending_group_confirmation" if q["conversation_id"] not in annotations.get("conversations", {}) and q["type"] == "多輪對話" else "not_applicable_or_overlay_reviewed"})
    checks = [{"job": j["id"], "expected_full_measurements": expected(j, questions, config),
               "recorded_measured": sum(r["job"] == j["id"] for r in evidence),
               "stage_status": stages.get(j["id"], {"status": "not_executed"}),
               "formal_quality_certified": False} for j in plan]
    checks += [{"item": "training/test exposure", "detail": {k: audit[k] for k in ("exact_train_overlap", "exact_faq_overlap", "near_train_overlap")}},
               {"item": "TI/RAG integration", "status": "unavailable_no_claim_of_combined_attention_RAG"},
               {"item": "Gold", "status": "missing answers remain pending; no fabricated accuracy"},
               {"item": "timing scope", "detail": "Warmup excluded; cold/warm separate. SD shared-preparation controlled sum is not INT independent-online E2E."},
               {"item": "SD0 ordering", "detail": "SD0 establishes paired reference first; nonbaseline modes rotate. Fixed baseline-first ordering is a limitation."}]
    ti_metrics = []
    for method in {m for r in evidence for m in r.get("predictions", {})}:
        from experiments.algorithms.topic_identification.metrics import classification
        selected = [r for r in evidence if method in r.get("predictions", {}) and r["status"] == "success"]
        ti_metrics.append({"method": method, **classification([int(r["gold_shift"]) for r in selected],
                                                              [int(r["predictions"][method]) for r in selected]),
                           "caution": "repeats are repeated measurements, not independent conversations"})
    for j, status in stages.items():
        if status.get("status") in {"failed", "unavailable"}:
            failures.append({"job": j, **status})
    token_failures = sum(r["status"] in {"failed_token_mismatch", "failed_sd0_not_repeatable", "failed_int_token_mismatch"} for r in evidence)
    sheets = {"Overview": [{"run": run.name, "status": read(run / "run.json")["status"],
                            "questions": len(questions), "measured_records": len(evidence), "warmup_excluded": len(warmups),
                            "token_guard_failures": token_failures, "human_gold_certified": False}],
              "Comparison": comparison, "Per Question": table, "Question Coverage": coverage,
              "By Type": type_stats, "Pending Quality": quality, "TI Metrics": ti_metrics,
              "Configuration": [{"key": k, "value": v} for k, v in config.items()],
              "Validation": checks, "Failures": failures}
    tmp = run / ("summary_" + uuid.uuid4().hex[:10] + ".xlsx")
    workbook(tmp, sheets, presentation=True)
    os.replace(str(tmp), str(run / "summary.xlsx"))
    lines = ["# 資科萬事通804題測試結果", "", "此報告整理實際證據；技術執行不代表人工Gold或正式品質認證。",
             "", "- Run：" + run.name, "- 狀態：" + read(run / "run.json")["status"],
             "- 原始804題全部保留於Question Coverage；多輪群組未核准者不作正式評分。",
             "- 已保存測量：%d；排除Warmup：%d；Token Guard失敗：%d。" % (len(evidence), len(warmups), token_failures),
             "- 訓練原題重疊：%d；FAQ候選原題重疊：%d；高字面相似：%d（不等於語意洩漏已證實）。" %
             (audit["exact_train_overlap"], audit["exact_faq_overlap"], audit["near_train_overlap"]),
             "", "## 實測比較", "", "|方法|階段|已記錄|成功|平均E2E秒|P50|P95|", "|---|---|---:|---:|---:|---:|---:|"]
    def shown(x):
        return "無法計算" if x is None else "%.4f" % x
    for r in comparison:
        lines.append("|%s / %s|%s|%d|%d|%s|%s|%s|" % (r["Job"], r["Mode"], r["Phase"], r["Recorded"], r["Successful"],
                      shown(r["mean_seconds"]), shown(r["p50_seconds"]), shown(r["p95_seconds_nearest_rank"])))
    lines += ["", "## 限制與未執行項目", "", "- 沒有已驗證標準答案時，不計算回答正確率；FAQ只在獨立Gold FAQ ID存在時評分。",
              "- Cache命中不代表答案安全；未有實際來源配對人審的誤用率為無法計算。Warm為同一序列Cold完成後再訪，Cold成本另列。",
              "- SD只將Token完全一致且SD0重複穩定的測量納入合格子集加速比；任一不一致均不可宣稱完整Lossless通過。",
              "- FP32 SD、其他FP16 QA及不同Draft駐留條件不得混成單一速度排名。沒有可靠接受率就不填數值。",
              "- TI只在原分組、逐輪標籤及獨立校準協定完整時執行；Attention與RAG的整合不在本系統冒充完成。",
              "- 資料沒有全部族群的可靠答案或模型正式批准證據時，研究資格仍待補齊。", ""]
    for f in failures:
        lines.append("- %s：%s" % (f.get("job"), f.get("reason", f.get("error", f.get("status")))))
    report_tmp = run / ("report_" + uuid.uuid4().hex[:10] + ".md")
    report_tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(str(report_tmp), str(run / "report.md"))
    save(run / "raw/report_manifest.json", {"result_indices_sha256": result_hashes,
         "summary_sha256": sha(run / "summary.xlsx"), "report_sha256": sha(run / "report.md"),
         "measured_records": len(evidence), "token_guard_failures": token_failures,
         "formal_quality_certified": False}, replace=True)
    return {"token_guard_failures": token_failures}
