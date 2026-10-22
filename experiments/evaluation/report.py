"""Immutable report snapshots and explicit human scores; no model/judge calls."""
from __future__ import annotations

import csv
import html
import json
import statistics

from .common import (ARCHITECTURES, FAQ_ARCHITECTURES, DataError, atomic_json, digest,
                     identifier, latest_results, read_json, read_run)

FLAGS = ("事實錯誤", "答案不完整", "答非所問", "過度生成", "未依據檢索資料", "標準答案本身有問題")
GROUPS = {"train": 42, "validation": 9, "test": 9, "full": 60}


def current_reviews(run):
    result = {}
    for path in sorted((run / "reviews").glob("*.json")):
        event = read_json(path)
        for row in event["scores"]:
            key = (row["architecture"], row["faq_id"], row["attempt_id"])
            prior = result.get(key)
            if prior is None or event["created_ns"] > prior[0]:
                result[key] = (event["created_ns"], row)
    return {key: value[1] for key, value in result.items()}


def import_reviews(run, path, reviewer):
    meta = read_run(run)
    current = latest_results(run)
    accepted, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"run_id", "architecture", "faq_id", "attempt_id", "answer_sha256", "question",
                    "reference_answer", "generated_answer", "score", "notes", *FLAGS}
        if not required <= set(reader.fieldnames or []):
            raise DataError("review CSV missing required columns")
        for number, row in enumerate(reader, 2):
            score = row["score"].strip()
            if not score:
                continue  # blanks are UNRATED, never numeric zero
            if score not in ("0", "1", "2", "3", "N/A"):
                raise DataError("review row %d: score must be 0/1/2/3/N/A/blank" % number)
            key = (row["architecture"], row["faq_id"])
            actual = current.get(key)
            if key in seen or row["run_id"] != meta["run_id"] or not actual or actual["status"] != "success":
                raise DataError("review row %d: duplicate, wrong run or no successful answer" % number)
            seen.add(key)
            if (row["attempt_id"] != actual["attempt_id"] or row["answer_sha256"] != digest(actual["generated_answer"])
                    or any(row[k] != actual[k] for k in ("question", "reference_answer", "generated_answer"))):
                raise DataError("review row %d: stale/edited answer or question; export a fresh review table" % number)
            flags = {}
            for flag in FLAGS:
                value = row[flag].strip().lower()
                if value not in ("", "0", "1", "false", "true"):
                    raise DataError("review row %d: %s must be blank/0/1/false/true" % (number, flag))
                flags[flag] = value in ("1", "true")
            accepted.append({"architecture": key[0], "faq_id": key[1], "attempt_id": row["attempt_id"],
                             "answer_sha256": row["answer_sha256"], "score": score, "flags": flags,
                             "notes": row["notes"], "reviewer": reviewer})
    if not accepted:
        raise DataError("no filled scores to import")
    import time
    target = run / "reviews" / (identifier() + ".json")
    atomic_json(target, {"created_ns": time.time_ns(), "reviewer": reviewer, "scores": accepted})
    return target


def csv_file(path, rows, fields):
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * p
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def generate_report(run):
    meta = read_run(run)
    latest = latest_results(run)
    reviews = current_reviews(run)
    output = run / "reports" / identifier()
    output.mkdir(parents=True, exist_ok=False)
    invalidations = [read_json(p) for p in sorted((run / "integrity_violations").glob("*.json"))]
    rows = []
    planned = set(meta["question_ids"])
    for question in meta["dataset"]:
        for arch in ARCHITECTURES:
            row = latest.get((arch, question["faq_id"]))
            if row is None:
                row = {"run_id": meta["run_id"], "architecture": arch, **question,
                       "status": "not_run" if arch in meta["architectures"] and question["faq_id"] in planned else "not_selected",
                       "generated_answer": "", "error_message": None, "attempt_id": ""}
            row = dict(row)
            score = reviews.get((arch, question["faq_id"], row["attempt_id"]))
            row["human_review"] = score
            row["run_integrity_valid"] = not invalidations
            rows.append(row)
    with (output / "all_results.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    paired = []
    for question in meta["dataset"]:
        a0 = latest.get(("A0", question["faq_id"]))
        a5 = latest.get(("A5", question["faq_id"]))
        if a0 and a5 and a0["status"] == a5["status"] == "success":
            identity = lambda row: [(d["rank"], d["document_id"], d["content"]) for d in row.get("retrieved_documents", [])]
            paired.append({"faq_id": question["faq_id"], "same_top5_order_and_content": identity(a0) == identity(a5)})
    atomic_json(output / "retrieval_comparability.json", {"paired_successes": len(paired), "checks": paired})
    summary = []
    for arch in ARCHITECTURES:
        for split, population in GROUPS.items():
            group = [r for r in rows if r["architecture"] == arch and (split == "full" or r["split"] == split)]
            successes = [r for r in group if r["status"] == "success"]
            failures = [r for r in group if r["status"] == "error"]
            requested = sum(r["faq_id"] in planned and arch in meta["architectures"] for r in group)
            numeric = [int(r["human_review"]["score"]) for r in successes
                       if r["human_review"] and r["human_review"]["score"] != "N/A"]
            na = sum(bool(r["human_review"] and r["human_review"]["score"] == "N/A") for r in successes)
            latencies = [r["latency_seconds"] for r in successes if r.get("latency_seconds") is not None]
            tokens = [r["output_tokens"] for r in successes if r.get("output_tokens") is not None]
            retrieval = [r["retrieval_seconds"] for r in successes if r.get("retrieval_seconds") is not None]
            generation = [r["generation_seconds"] for r in successes if r.get("generation_seconds") is not None]
            correct = sum(r.get("faq_id_correct") is True for r in successes)
            summary.append({"architecture": arch, "split": split, "population": population,
                            "requested": requested, "success": len(successes), "failed": len(failures),
                            "missing_requested": requested - len(successes) - len(failures),
                            "completion_rate_requested": (len(successes) + len(failures)) / requested if requested else None,
                            "success_rate_population": len(successes) / population,
                            "success_rate_requested": len(successes) / requested if requested else None,
                            "faq_top1_accuracy_successful": correct / len(successes) if arch in FAQ_ARCHITECTURES and successes else None,
                            "faq_correct": correct if arch in FAQ_ARCHITECTURES else None,
                            "human_numeric_scored": len(numeric), "human_na": na,
                            "human_unrated_successful": len(successes) - len(numeric) - na,
                            "human_mean_0_to_3": statistics.mean(numeric) if numeric else None,
                            "human_status": "已評分" if numeric else "N/A" if na else "尚未評分",
                            "mean_latency_seconds": statistics.mean(latencies) if latencies else None,
                            "median_latency_seconds": statistics.median(latencies) if latencies else None,
                            "p95_latency_seconds": percentile(latencies, .95),
                            "mean_retrieval_seconds": statistics.mean(retrieval) if retrieval else None,
                            "mean_generation_seconds": statistics.mean(generation) if generation else None,
                            "mean_output_tokens": statistics.mean(tokens) if tokens else None,
                            "model_provisional": (True if any(r.get("model_provisional") is True for r in group)
                                                  else False if any(r.get("model_provisional") is False for r in group) else None),
                            "integrity_valid": not invalidations})
    csv_file(output / "summary.csv", summary, list(summary[0]))
    comparison = []
    for question in meta["dataset"]:
        record = dict(question)
        for arch in ARCHITECTURES:
            row = next(r for r in rows if r["architecture"] == arch and r["faq_id"] == question["faq_id"])
            record[arch] = row["generated_answer"] if row["status"] == "success" else "[%s] %s" % (row["status"], row.get("error_message") or "")
        comparison.append(record)
    csv_file(output / "comparison.csv", comparison, list(comparison[0]))
    review_rows = []
    for row in rows:
        score = row["human_review"]
        review_rows.append({**{k: row.get(k, "") for k in ("run_id", "architecture", "faq_id", "split", "attempt_id", "status",
                                                              "question", "reference_answer", "generated_answer")},
                            "answer_sha256": digest(row["generated_answer"]), "score": score["score"] if score else "",
                            "notes": score["notes"] if score else "",
                            **{flag: int(score["flags"][flag]) if score else "" for flag in FLAGS}})
    csv_file(output / "review.csv", review_rows, list(review_rows[0]))
    esc = lambda value: html.escape(str(value), quote=True)
    parts = ['<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>FAQ 統一評估</title>',
             '<style>body{font:16px system-ui;margin:2rem;line-height:1.6}table{border-collapse:collapse;width:100%}'
             'td,th{border:1px solid #ccc;padding:.4rem;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
             '.error{color:#a00}.muted{color:#666}details{margin:1rem 0}article{border-left:3px solid #aaa;padding:1rem;margin:.5rem}</style>',
             '<h1>第一輪 FAQ 統一評估</h1><p>Run: %s；mode: %s</p>' % (esc(meta["run_id"]), esc(meta["mode"])),
             '<p>Smoke 權重、原題匹配與 RAG 可讀 FAQ 的結果，不能當作未見改寫泛化或充分訓練結論。'
             '人工空白為尚未評分；N/A 不計平均；延遲只統計成功題，不含載入和磁碟保存，未預先暖機。</p>',
             '<nav>' + ' | '.join('<a href="#%s">%s</a>' % (k, k) for k in GROUPS if k != "full") + '</nav>']
    if invalidations:
        parts.append('<h2 class="error">輸入完整性失敗：所有統計僅供排錯，不可用於比較</h2><pre>%s</pre>' % esc(json.dumps(invalidations, ensure_ascii=False, indent=2)))
    mismatches = [r["faq_id"] for r in paired if not r["same_top5_order_and_content"]]
    parts.append('<p>A0/A5 實際Top5文件順序/內容核對：配對成功題數 %d；不一致FAQ：%s。未配對題不宣稱通過。</p>' %
                 (len(paired), esc(", ".join(mismatches) or "無（限已配對題）")))
    for split in GROUPS:
        parts.append('<h2>%s 摘要</h2><table><tr><th>架構</th><th>成功/母體</th><th>失敗/未完成</th>'
                     '<th>FAQ ID Accuracy（成功題）</th><th>人工品質</th><th>平均/中位/P95 秒</th><th>provisional</th></tr>' % split)
        fmt = lambda v: "—" if v is None else "%.4f" % v
        for row in [s for s in summary if s["split"] == split]:
            parts.append('<tr><td>%s</td><td>%d/%d</td><td>%d/%d</td><td>%s</td><td>%s；n=%d；mean=%s</td>'
                         '<td>%s / %s / %s</td><td>%s</td></tr>' % (
                             row["architecture"], row["success"], row["population"], row["failed"], row["missing_requested"],
                             fmt(row["faq_top1_accuracy_successful"]), row["human_status"], row["human_numeric_scored"],
                             fmt(row["human_mean_0_to_3"]), fmt(row["mean_latency_seconds"]),
                             fmt(row["median_latency_seconds"]), fmt(row["p95_latency_seconds"]), row["model_provisional"]))
        parts.append('</table>')
    for split in ("train", "validation", "test"):
        parts.append('<h2 id="%s">%s 逐題比較</h2>' % (split, split))
        for question in [q for q in meta["dataset"] if q["split"] == split]:
            parts.append('<details id="%s"><summary>%s — %s</summary><h3>標準答案</h3><pre>%s</pre>' % (
                esc(question["faq_id"]), esc(question["faq_id"]), esc(question["question"]), esc(question["reference_answer"])))
            for row in [r for r in rows if r["faq_id"] == question["faq_id"]]:
                wrong = row.get("faq_id_correct") is False
                score = row["human_review"]
                parts.append('<article><h3>%s — %s %s</h3><pre>%s</pre><p class="error">%s</p>'
                             '<p>預測FAQ: %s；人工分數: %s</p><pre>%s</pre></article>' % (
                                 esc(row["architecture"]), esc(row["status"]), '【FAQ ID 錯誤】' if wrong else '',
                                 esc(row["generated_answer"]), esc(row.get("error_message") or ""),
                                 esc(row.get("predicted_faq_id") or "—"), esc(score["score"] if score else "尚未評分"),
                                 esc(json.dumps(score, ensure_ascii=False, indent=2) if score else "")))
            parts.append('</details>')
    contracts = {p.stem: read_json(p) for p in sorted((run / "contracts").glob("*.json"))}
    preflights = {p.name: read_json(p) for p in sorted((run / "preflight").glob("*.json"))}
    parts.append('<details><summary>完整設定、資源、provisional 與 Preflight</summary><pre>%s</pre></details></html>' %
                 esc(json.dumps({"run": meta, "runtime_contracts": contracts, "preflight": preflights}, ensure_ascii=False, indent=2)))
    (output / "report.html").write_text("\n".join(parts), encoding="utf-8")
    atomic_json(output / "report_complete.json", {"run_id": meta["run_id"], "integrity_valid": not invalidations,
                                                "raw_attempts_preserved": True})
    return output
