"""Collect existing full-run evidence, including failed SD, without executing models.

This is a retrospective provisional report, never a completion/certification receipt.
All input files remain untouched; output must be a new sibling directory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import traceback
import uuid

from .common import read_json, read_rows, sha, verify_bundle, fingerprint
from .reporting import generate, workbook, flatten, text


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def inside(base, relative):
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError("evidence path escapes original Run: " + str(relative))
    return path


def inventory(source):
    return {p.relative_to(source).as_posix(): sha(p) for p in sorted(source.rglob("*"))
            if p.is_file() and p.name != "active.lock"}


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def mean(values):
    return statistics.mean(values) if values else "無法計算"


def sd_projection(directory, job_id, sub, validation):
    """Strict retrospective pairing, scoped to one substudy; never modifies records."""
    path = directory / "results.jsonl"
    if not path.is_file():
        validation.append({"實驗": job_id, "子實驗": sub, "狀態": "未驗證",
                           "原因": "缺少results.jsonl，無法計算；不以raw未比較資料代替"})
        return [], [], []
    rows = read_rows(path)
    controls = defaultdict(list)
    seen = defaultdict(set)
    for row in rows:
        if row.get("experiment_version") == "SD0":
            controls[(row.get("scenario_id"), row.get("repeat_index"), row.get("warmup"))].append(row)
            tokens = row.get("output_token_ids")
            if isinstance(tokens, list):
                seen[(row.get("scenario_id"), row.get("warmup"))].add(tuple(tokens))
    details, failures = [], []
    groups = defaultdict(list)
    for row in rows:
        key = (row.get("scenario_id"), row.get("repeat_index"), row.get("warmup"))
        refs = controls[key]
        ref = refs[0] if len(refs) == 1 else None
        ids, ref_ids = row.get("output_token_ids"), ref.get("output_token_ids") if ref else None
        identity_ok = bool(ref) and all(row.get(k) is not None and row.get(k) == ref.get(k)
                       for k in ("prompt_identity", "retrieval_identity", "target_identity", "seed"))
        tokens_valid = (isinstance(ids, list) and bool(ids) and all(type(i) is int for i in ids)
                        and isinstance(ref_ids, list) and bool(ref_ids) and all(type(i) is int for i in ref_ids))
        if tokens_valid:
            tokens_valid = (fingerprint(ids) == row.get("output_token_sha256")
                            and fingerprint(ref_ids) == ref.get("output_token_sha256"))
        equal = ids == ref_ids if identity_ok and tokens_valid else None
        repeatable = len(seen[(row.get("scenario_id"), row.get("warmup"))]) == 1
        recorded_ok = (row.get("status") == "success" and row.get("performance_eligible") is True
                       and row.get("exact_token_match_sd0") is True and row.get("fallback") is False)
        baseline_ok = bool(ref) and ref.get("status") == "success" and ref.get("performance_eligible") is True
        eligible = equal is True and repeatable and recorded_ok and baseline_ok and row.get("warmup") is False
        ratio = (ref["generation_seconds"] / row["generation_seconds"]
                 if eligible and number(ref.get("generation_seconds")) and number(row.get("generation_seconds")) else None)
        reason = row.get("error") or ("配對身分／Token證據不足" if equal is None else
                 "Token與成對SD0不同" if not equal else "SD0不可重現" if not repeatable else
                 "原始Guard未批准效能統計" if not recorded_ok or not baseline_ok else None)
        detail = {"實驗": job_id, "子實驗": sub, "模式": row.get("experiment_version"),
                  "題目ID": row.get("scenario_id"), "問題": row.get("question"),
                  "Warmup": row.get("warmup"), "Repeat": row.get("repeat_index"),
                  "原始狀態": row.get("status"), "Token一致": equal,
                  "Lossless統計合格": eligible, "子實驗全體通過": False,
                  "生成秒": row.get("generation_seconds"), "端到端秒": row.get("end_to_end_seconds"),
                  "生成Token數": row.get("generated_token_count"),
                  "合格配對生成Speedup": ratio, "答案": row.get("output_text"), "錯誤": reason,
                  "來源": str(path), "measurement_id": row.get("measurement_id"),
                  "原始紀錄": row}
        details.append(detail)
        groups[row.get("experiment_version", "unknown")].append(detail)
        if reason:
            failures.append({k: v for k, v in detail.items() if k != "原始紀錄"})
    all_pass = bool(details) and all(d["Token一致"] is True and not d["錯誤"] for d in details)
    committed = False
    receipt_path = directory / "completed.json"
    if receipt_path.is_file() and not (directory / "failed.json").exists():
        receipt = read_json(receipt_path)
        committed = receipt.get("status") == "completed" and receipt.get("source_verification") == "passed"
    comparisons = []
    for mode, group in sorted(groups.items()):
        measured = [r for r in group if r["Warmup"] is False]
        known = [r for r in measured if r["Token一致"] is not None]
        ratios = [r["合格配對生成Speedup"] for r in measured if r["合格配對生成Speedup"] is not None]
        times = [r["生成秒"] for r in measured if number(r["生成秒"])]
        end_times = [r["端到端秒"] for r in measured if number(r["端到端秒"])]
        comparisons.append({"實驗": job_id, "子實驗": sub, "模式": mode,
            "狀態": "已完成／Provisional" if committed and all_pass else "失敗或未提交／Provisional",
            "正式測量數": len(measured), "Warmup數": len(group) - len(measured),
            "Token可判定數": len(known), "Token一致通過數": sum(r["Token一致"] is True for r in known),
            "Token一致通過率": sum(r["Token一致"] is True for r in known) / len(known) if known else "無法計算",
            "失敗數含Warmup": sum(bool(r["錯誤"]) for r in group),
            "正式測量失敗數": sum(bool(r["錯誤"]) for r in measured),
            "失敗題目": sorted({r["題目ID"] for r in group if r["錯誤"]}),
            "平均生成秒含不等價輸出": mean(times),
            "平均受控端到端秒含不等價輸出": mean(end_times),
            "中位生成秒含不等價輸出": statistics.median(times) if times else "無法計算",
            "合格配對Speedup樣本數": len(ratios),
            "合格子集平均Speedup非全體結論": mean(ratios),
            "完整Lossless通過": committed and all_pass,
            "限制": "Warmup排除平均；失敗輸出長度可能不同，全部耗時僅描述。合格子集有選擇偏差；不能代表整體Lossless加速。"})
    for row in details:
        row["子實驗全體通過"] = committed and all_pass
    validation.append({"實驗": job_id, "子實驗": sub, "原始完成與來源收據": committed,
                       "Token重核全通過": all_pass, "紀錄數": len(rows),
                       "證據身分": "已提交" if committed else "部分／失敗證據；新Hash僅固定目前檔案，不補認原始完整性"})
    return comparisons, details, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    def locate(p):
        return (p if p.is_absolute() else root / p).resolve()
    source = locate(args.source_run)
    base = (root / "experiments/results/algorithms").resolve()
    if not source.is_relative_to(base):
        raise ValueError("source must be an existing Run under experiments/results/algorithms")
    state, identity = read_json(source / "raw/state.json"), read_json(source / "raw/identity.json")
    output = locate(args.output_dir) if args.output_dir else base / ("collected_provisional_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex)
    if not output.is_relative_to(base) or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("output must be a NEW separate directory, never inside/above the source Run")
    before = inventory(source)
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    raw.mkdir()
    try:
        sheets = {name: [] for name in ("Overview", "QA Comparison", "SC Comparison", "SD Comparison",
                  "TI Comparison", "INT Comparison", "Per Question", "Groups", "Failures", "Configuration", "Validation")}
        validation = sheets["Validation"]
        validation.append({"檢查": "SD版本範圍", "狀態": "原Run的SD v1；獨立v2未驗證",
            "限制": "不讀取或合併其他目錄的v2單題成果；不得以v2成功覆蓋v1失敗或宣稱完整SD通過。"})
        validation.append({"檢查": "階段數", "實際": len(state["jobs"]), "預期": 11,
                           "狀態": "符合" if len(state["jobs"]) == 11 else "未驗證"})
        sheets["Configuration"].append({"來源Run": str(source), "原始身分": identity,
            "報告身分": "Provisional回溯整理；不產生實驗完成或Gold認證", "原始階段數": len(state["jobs"])})
        for job in state["jobs"]:
            stage = {"實驗": job["id"], "研究項目": job["target"], "階段": job["stage"],
                     "Cache狀態": job.get("cache_state"), "原始狀態": job["status"],
                     "報告身分": "Provisional", "原始錯誤": job.get("error"), "結果路徑": job.get("result")}
            sheets["Overview"].append(stage)
            if not job.get("result"):
                stage["證據驗證"] = "未驗證／無結果路徑"
                continue
            child = inside(source, job["result"])
            if job["target"] == "sd":
                for sub in ("benchmark", "prompt_lookup"):
                    try:
                        comparisons, details, failures = sd_projection(child / "raw" / sub, job["id"], sub, validation)
                        sheets["SD Comparison"].extend(comparisons)
                        sheets["Per Question"].extend(details)
                        sheets["Failures"].extend(failures)
                    except Exception as exc:
                        validation.append({"實驗": job["id"], "子實驗": sub, "狀態": "未驗證", "原因": str(exc)})
                stage["證據驗證"] = "見SD逐子實驗；不將失敗階段升級為完成"
                continue
            if job["status"] != "completed":
                stage["證據驗證"] = "未提交，不列入有效統計"
                continue
            try:
                receipt = verify_bundle(child / "raw", "evidence.json")
                if fingerprint(receipt) != job.get("receipt"):
                    raise ValueError("child evidence receipt differs from original state")
                for name, digest in receipt["report_sha256"].items():
                    if sha(inside(child, name)) != digest:
                        raise ValueError("original child report changed: " + name)
                context = read_json(child / "raw/context.json")
                if any(context.get(k) != job[k] for k in ("target", "stage", "cache_state")):
                    raise ValueError("child context differs from scheduled job")
                projection = generate(child, context, projection_only=True)
                stage["證據驗證"] = "已完成；原始收據與檔案Hash通過"
                sheet = {"qa": "QA Comparison", "sc": "SC Comparison", "ti": "TI Comparison", "int": "INT Comparison"}[job["target"]]
                for row in projection["comparison"]:
                    sheets[sheet].append({"實驗": job["id"], "階段": job["stage"], "Cache狀態": job.get("cache_state"),
                        "身分": "Provisional；候選品質標籤未認證", **row})
                sheets["Per Question"].extend({"實驗": job["id"], "階段": job["stage"],
                    "身分": "Provisional；答案品質未評定", **row} for row in projection["detail"])
                sheets["Groups"].extend({"實驗": job["id"], **row} for row in projection["groups"])
                validation.extend({"實驗": job["id"], "限制": w} for w in projection["warnings"])
            except Exception as exc:
                stage["證據驗證"] = "未驗證／排除有效統計"
                validation.append({"實驗": job["id"], "狀態": "未驗證", "原因": str(exc)})
            if job.get("error"):
                sheets["Failures"].append({"實驗": job["id"], "原因": job["error"]})
        if inventory(source) != before:
            raise ValueError("source evidence changed while collecting; report aborted")
        save(raw / "source_evidence.json", {"source_run": str(source), "files_sha256": before,
              "meaning": "collection-time snapshot, not retrospective completion certification"})
        save(raw / "projection.json", sheets)
        workbook(output / "summary.xlsx", {k: [flatten(r) for r in v] for k, v in sheets.items()}, presentation=True)
        lines = ["# 資科萬事通完整預備研究結果整理", "", "身分：Provisional；本報告不是正式研究認證，也不變更原Run狀態。",
                 "來源：`" + str(source) + "`", "", "## 階段成果", "", "|階段|原始狀態|證據檢查|", "|---|---|---|"]
        def clean(v):
            return text(v).replace("|", "/").replace("\n", " ")
        for row in sheets["Overview"]:
            lines.append(f"|{clean(row['實驗'])}|{clean(row['原始狀態'])}|{clean(row.get('證據驗證'))}|")
        for name in ("QA Comparison", "SC Comparison", "SD Comparison", "TI Comparison", "INT Comparison"):
            lines += ["", "## " + name, "", "以下為來源證據可支持的描述統計。Calibration與Evaluation、Cold與Prefilled分開解讀。"]
            if not sheets[name]:
                lines.append("無法計算：缺少可驗證的結果。")
            else:
                lines += ["", "|實驗／模式|指標|紀錄值|", "|---|---|---|"]
            for row in sheets[name]:
                label = " / ".join(str(row[k]) for k in ("實驗", "子實驗", "模式", "mode", "phase") if k in row)
                for key, value in flatten(row).items():
                    if key not in {"實驗", "子實驗", "模式", "mode", "phase"}:
                        lines.append(f"|{clean(label)}|{clean(key)}|{clean(value)}|")
        lines += ["", "## 失敗及未驗證項目", ""]
        for row in sheets["Failures"]:
            lines.append("- " + text({k: row[k] for k in
                ("實驗", "子實驗", "模式", "題目ID", "Warmup", "Repeat", "原始狀態", "錯誤", "原因") if k in row}))
        for row in validation:
            if row.get("狀態") == "未驗證":
                lines.append("- " + text(row))
        lines += ["", "## 研究限制與正式實驗缺口", "",
            "- SD時間可作實際執行描述；Token不等價的輸出不納入Lossless Speedup。合格子集加速比不是全體成功或無損保證。",
            "- 本報告的SD為原Run的v1；獨立v2不在本次證據範圍，標為未驗證。v2單題成功不能替代完整SD Evaluation。",
            "- SD Warmup與正式Repeat分開計數，平均延遲與加速比排除Warmup；SD與Prompt Lookup各自配對本子實驗SD0。",
            "- 缺少子實驗（例如SD失敗後未啟動的Prompt Lookup）明確標為未驗證，不補造數據。",
            "- SC答案重用安全、TI候選標籤與生成答案品質仍須真實人工審核；Token一致不代表答案正確。",
            "- INT Initial Fill與Measured沿用原始分組統計，不合併為Measured平均。",
            "- 模型仍為Provisional；正式訓練批准、Gold認證及正式Runner的全部Guard仍須獨立滿足。",
            "- FAQ60包含42個已見訓練意圖，不能整批描述為未見泛化；本報告不推論回答品質改善或統計顯著性。",
            "- 原始收據與目前檔案Hash能核對已提交成果；失敗SD只固定收集當下證據，不能補認原始執行完成。",
            "", "完整逐題投影及收集時來源檔案SHA-256位於raw；原始Run與既有報告均保持原位。"]
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if inventory(source) != before:
            raise ValueError("source changed during report output; collection not committed")
        save(raw / "collection_receipt.json", {"status": "provisional_report_collected", "formal_result_complete": False,
              "reports_sha256": {n: sha(output / n) for n in ("summary.xlsx", "report.md")},
              "source_snapshot_sha256": sha(raw / "source_evidence.json")})
        print(f"預備報告（非完整實驗通過）：{output / 'summary.xlsx'}\n{output / 'report.md'}")
    except Exception:
        save(raw / "collection_failed.json", {"traceback": traceback.format_exc(), "reports_valid": False})
        raise


if __name__ == "__main__":
    main()
