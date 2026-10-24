"""Evidence-only XLSX/Markdown projection. Standard library; no model imports."""
from __future__ import annotations

from collections import defaultdict
import json
import math
import statistics
from xml.sax.saxutils import escape
import zipfile

from .common import read_json, read_rows, sha


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + str(key)
        if isinstance(item, dict):
            result.update(flatten(item, name + "."))
        else:
            result[name] = json.dumps(item, ensure_ascii=False) if isinstance(item, list) else item
    return result


def text(value):
    if value is None:
        return "unavailable"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def column(index):
    name = ""
    while index:
        index, digit = divmod(index - 1, 26)
        name = chr(65 + digit) + name
    return name


def workbook(path, sheets):
    """Small OOXML writer: explicit text cells prevent formula injection. Raw retains full strings."""
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    declarations = ['<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
                    '<Default Extension="xml" ContentType="application/xml"/>',
                    '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>']
    names, relations = [], []
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for number, (name, records) in enumerate(sheets.items(), 1):
            fields = list(dict.fromkeys(k for row in records for k in row)) or ["status"]
            if len(records) + 1 > 1048576 or len(fields) > 16384:
                raise ValueError("Excel row/column limit exceeded; raw evidence retained")
            rows = []
            for ri, values in enumerate([fields] + [[row.get(k) for k in fields] for row in records], 1):
                cells = []
                for ci, value in enumerate(values, 1):
                    ref = column(ci) + str(ri)
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                        cells.append(f'<c r="{ref}"><v>{value}</v></c>')
                    else:
                        original = text(value)
                        cleaned = "".join(c for c in original if c in "\t\n\r" or ord(c) >= 32)
                        # Excel's limit is UTF-16 code units, not Python characters.
                        cleaned = cleaned.encode("utf-16-le")[:64000].decode("utf-16-le", errors="ignore")
                        if cleaned != original:
                            cleaned += " [完整內容見raw]"
                        cells.append(f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{escape(cleaned)}</t></is></c>')
                rows.append(f'<row r="{ri}">' + "".join(cells) + '</row>')
            document = (f'<worksheet xmlns="{ns}"><sheetViews><sheetView workbookViewId="0">'
                        '<pane ySplit="1" topLeftCell="A2" state="frozen"/></sheetView></sheetViews>'
                        '<sheetData>' + "".join(rows) + '</sheetData>'
                        f'<autoFilter ref="A1:{column(len(fields))}{len(records)+1}"/></worksheet>')
            archive.writestr(f"xl/worksheets/sheet{number}.xml", document)
            names.append(f'<sheet name="{name}" sheetId="{number}" r:id="rId{number}"/>')
            relations.append(f'<Relationship Id="rId{number}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{number}.xml"/>')
            declarations.append(f'<Override PartName="/xl/worksheets/sheet{number}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
        archive.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' + "".join(declarations) + '</Types>')
        archive.writestr("_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr("xl/workbook.xml", f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>' + "".join(names) + '</sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(relations) + '</Relationships>')
    # Structural packaging verification only; no claim of Excel application validation.
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("XLSX archive CRC verification failed")


def generate(run, context):
    raw, target = run / "raw", context["target"]
    adapter = raw / "adapter"
    manifest = read_json(adapter / "adapter_manifest.json")
    native_name = context["stage"] + ".jsonl" if target == "ti" else "measured.jsonl" if target in {"sc", "int"} else "scenarios.jsonl"
    cases = read_rows(adapter / native_name)
    case_index = {c["scenario_id"]: c for c in cases}
    if target == "int":
        case_index.update({c["scenario_id"]: c for c in read_rows(adapter / "initial_fill.jsonl")})
    detail, comparison, warnings = [], [], []
    summaries = {}
    for destination in [raw / "benchmark"] + ([raw / "prompt_lookup"] if target == "sd" else []):
        if (destination / "failed.json").exists():
            raise ValueError("refusing completed report for a failed benchmark")
        receipt = read_json(destination / "completed.json")
        if receipt.get("status") != "completed" or receipt.get("source_verification") != "passed":
            raise ValueError("benchmark did not finish verified")
        summaries[destination.name] = read_json(destination / "summary.json")
    benchmark = raw / "benchmark"
    if target == "qa":
        detail = read_rows(benchmark / "results.scorer.jsonl")
        comparison = [{"mode": k, **v} for k, v in summaries["benchmark"]["architectures"].items()]
    elif target == "sc":
        pairs = {(p["origin_question_id"], p["current_question_id"]): p
                 for row in read_rows(adapter / "scorer_only.jsonl") for p in row["pair_reviews"]}
        baseline = {}
        for mode in ("none", "exact", "semantic"):
            rows = read_rows(benchmark / mode / "results.jsonl")
            origins = {r["query_id"]: case_index[r["query_id"]]["question_id"] for r in rows}
            for r in rows:
                r.update(mode=mode, scenario_id=r["query_id"], warmup=False, phase="measured")
                if mode == "none":
                    baseline[r["query_id"]] = r
                current = case_index[r["query_id"]]["question_id"]
                origin = origins.get(r.get("origin_query_id"))
                human = pairs.get((origin, current), {}).get("human_review", {})
                allowed = (human.get("reuse_decision") if human.get("reuse_status") == "verified"
                           and human.get("reuse_decision") in {"reusable", "not_reusable"} else "unverified")
                r["reuse_review_label"] = allowed
                r["false_reuse_confirmed"] = (allowed == "not_reusable") if r["hit"] and allowed != "unverified" else None
                r["seconds_saved_vs_none"] = baseline[r["query_id"]]["timings"]["end_to_end_seconds"] - r["timings"]["end_to_end_seconds"]
            detail.extend(rows)
            group = [r for r in detail if r["mode"] == mode]
            comparison.append({"mode": mode, **summaries["benchmark"]["modes"][mode],
                               "false_reuse_confirmed": sum(r["false_reuse_confirmed"] is True for r in group) if not context["functional"] else None,
                               "unverified_hit_count": sum(r["hit"] and r["reuse_review_label"] == "unverified" for r in group),
                               "mean_seconds_saved_vs_none": statistics.mean(r["seconds_saved_vs_none"] for r in group),
                               "descriptive_total_e2e_speedup_none":
                               sum(r["timings"]["end_to_end_seconds"] for r in baseline.values()) /
                               sum(r["timings"]["end_to_end_seconds"] for r in group)})
        warnings.append("SC固定None/Exact/Semantic順序的一次序列；不是隨機重複量測，也不代表真實流量命中率。")
    elif target == "sd":
        for name, summary in summaries.items():
            for r in read_rows(raw / name / "results.jsonl"):
                r["substudy"] = name
                r["mode"] = r["experiment_version"]
                detail.append(r)
            comparison.extend({"substudy": name, "mode": k, **v} for k, v in summary["modes"].items())
        # Separate SD0 controls remain separate measurements, never averaged as one run.
        controls = {r["substudy"]: r for r in detail if r["mode"] == "SD0"}
        if controls["benchmark"]["target_identity"] != controls["prompt_lookup"]["target_identity"]:
            raise ValueError("SD/lookup Target model identity changed between substudies")
        a = read_json(benchmark / "scenarios.snapshot.json")
        b = read_json(raw / "prompt_lookup/scenarios.snapshot.json")
        if a != b:
            raise ValueError("SD and lookup did not use identical query sequence")
        for number in range(1, len(a) + 1):
            first, second = [read_json(raw / name / f"prompt_{number:04d}.json") for name in ("benchmark", "prompt_lookup")]
            for key in ("prompt_identity", "retrieval_identity"):
                if key not in first or first[key] != second.get(key):
                    raise ValueError(f"SD/lookup prompt identity differs: {key}")
        warnings.append("SD及Prompt Lookup為兩個獨立子實驗，各有成對SD0；模型駐留可能不同，跨子實驗不合併SD0或直接比較延遲。")
    elif target == "int":
        for phase in ("initial_fill", "measured"):
            path = benchmark / phase / "results.jsonl"
            if path.exists():
                detail.extend({**r, "phase": phase, "mode": r["system_variant"]} for r in read_rows(path))
            comparison.extend({"phase": phase, "mode": k, **v}
                              for k, v in summaries["benchmark"]["phases"][phase].get("variants", {}).items())
        warnings.append("Initial Fill及Warmup均不進入Measured平均；INT是獨立線上E2E，不與SD受控共享檢索時間混算。")
    else:
        thresholds_file = benchmark / ("thresholds.json" if context["stage"] == "calibration" else "thresholds.used.json")
        methods = read_json(thresholds_file)["payload"]["methods"]
        features = read_rows(benchmark / (context["stage"] + "_features.jsonl"))
        for feature in features:
            for method, chosen in methods.items():
                data = feature["embedding"] if method == "TI0" else feature["attention"]
                score = data["scores"][method]
                detail.append({"scenario_id": feature["scenario_id"], "mode": method,
                               "gold_shift": feature["gold_shift"], "prediction_shift": int(score < chosen["threshold"]),
                               "threshold": chosen["threshold"], "score": score, "warmup": False,
                               "feature_detection_seconds": data["detection_seconds"],
                               "classification_error": feature["gold_shift"] != int(score < chosen["threshold"]),
                               "threshold_timing_status": "see original classification.jsonl on Evaluation; Calibration postprocess only"})
        if context["stage"] == "calibration":
            comparison = [{"mode": k, "threshold": v["threshold"], **v["calibration_metrics"]} for k, v in methods.items()]
        else:
            comparison = [{"mode": k, "threshold": methods[k]["threshold"], **v} for k, v in summaries["benchmark"]["evaluation"].items()]
            # Retain the actual measured threshold-decision time, rather than substituting postprocessing time.
            actual = {(r["scenario_id"], r["method"]): r for r in read_rows(benchmark / "classification.jsonl")}
            for row in detail:
                measured_row = actual[(row["scenario_id"], row["mode"])]
                if (measured_row["prediction_shift"] != row["prediction_shift"]
                        or measured_row["threshold"] != row["threshold"]):
                    raise ValueError("TI frozen classifier/report mismatch")
                row.update(measured_row)
        warnings.append("Calibration分數為同資料選門檻後的擬合結果，不能當獨立Evaluation；Attention各Layer共用forward，不可加總成本。")
    for r in detail:
        case = case_index.get(r.get("scenario_id", r.get("question_id")), {})
        r["source_case"] = case  # Includes review/source/group/history, scorer-side only.
    measured = [r for r in detail if not r.get("warmup", False) and r.get("phase", "measured") == "measured"]
    warnings += ["生成答案品質尚未逐答案人審；Token一致不等於回答正確。",
                 "Excel過長儲存格會截短，完整回答、Token及來源保留raw。",
                 "題庫原有曝光及語意家族邊界仍需依人審紀錄解讀；不宣稱盲測或完全無洩漏。"]
    if context["functional"]:
        warnings.append("PROVISIONAL功能測試：非正式研究成績，未完成人工Gold認證。")
    configuration = []
    for path in sorted(raw.rglob("*.json")):
        if path.name in {"run.json", "identity.json", "context.json", "config.snapshot.json", "thresholds.json", "thresholds.used.json"}:
            configuration.append({"file": str(path.relative_to(run)), "sha256": sha(path), "content": read_json(path)})
    groups = defaultdict(list)
    for row in measured:
        event = row.get("source_case", {}).get("workload_event", {})
        if event.get("group_id"):
            groups[(row.get("mode"), row.get("repeat_index", 0), event["group_id"])].append(row)
    group_rows = []
    for (mode, repeat, group), rows in groups.items():
        positions = [r["source_case"]["workload_event"]["group_position"] for r in rows]
        if sorted(positions) != [1, 2, 3, 4]:
            raise ValueError("incomplete or duplicated four-query measured group")
        latencies = [r.get("end_to_end_seconds", r.get("timings", {}).get("end_to_end_seconds")) for r in rows]
        group_rows.append({"mode": mode, "repeat": repeat, "group": group, "count": 4,
                           "mean_end_to_end_seconds": statistics.mean(latencies)})
    effective_config = read_json(raw / "config.snapshot.json")
    overview = [{"target": target, "stage": context["stage"], "suite_revision": context["suite_revision"],
                 "status": "benchmark_verified_reporting_projection", "provisional": context["functional"],
                 "human_gold_certified": context["human_gold_certified"], "input_queries": manifest["count"],
                 "measured_result_rows": len(measured), "all_result_rows": len(detail),
                 "modes": list(dict.fromkeys(r.get("mode") for r in comparison)),
                 "target_model": effective_config.get("model", "see Configuration: architecture-specific model paths"),
                 "draft_model": effective_config.get("draft_model", "not_applicable_or_see_Configuration"),
                 "bge": effective_config.get("bge", "see Configuration"),
                 "answer_quality_verified": False}]
    sheets = {"Overview": overview, "Comparison": [flatten(r) for r in comparison],
              "Per Question": [flatten(r) for r in detail], "Configuration": configuration,
              "Validation": [{"check": "benchmark_completion_and_sources", "status": "passed"},
                             {"check": "human_gold_certified", "status": context["human_gold_certified"]},
                             {"check": "review_view_sha256", "value": context["view_sha256"]},
                             {"check": "output_answer_quality_verified", "status": False},
                             {"check": "dataset_hashes", "value": context["dataset_hashes"]}]
                            + [{"warning": w} for w in warnings], "Groups": group_rows}
    workbook(run / "summary.xlsx", sheets)
    md = [f"# {target.upper()} — {context['stage']}", "",
          "目的：以既有Benchmark量測比較；此文件只整理原始證據，不產生答案正確率結論。", "",
          "```json", json.dumps(overview[0], ensure_ascii=False, indent=2), "```", "",
          "## 各模式結果", ""]
    # Long-form table keeps every actual metric and sample denominator, including nested matrices.
    md += ["| 子實驗／階段 | 模式 | 指標 | 原始值 |", "|---|---|---|---|"]
    def safe(v):
        return text(v).replace("|", "\\|").replace("\n", " ")
    for row in comparison:
        for key, value in flatten(row).items():
            if key not in {"mode", "phase", "substudy"}:
                md.append(f"| {safe(row.get('substudy', row.get('phase', '')))} | {safe(row.get('mode'))} | {safe(key)} | {safe(value)} |")
    md += ["", "## 設定與版本", "", "完整模型、設定與版本見Excel Configuration及raw/identity.json等原始檔。", "",
           f"題庫身分：`{context['dataset_identity']}`；設定SHA-256：`{context['config_sha256']}`。", "",
           "## 驗證與限制", ""] + ["- " + w for w in warnings]
    md += ["", "客觀紀錄：", f"- 原始輸入共{manifest['count']}題／次，取得{len(measured)}筆排除Warmup及Initial Fill的結果。",
           "- 所有子Benchmark完成並通過其來源驗證；整體報告是否完整以raw/completed.json為準。",
           "- 完整錯誤、逐題答案、收據與Hash均保留raw；失敗執行不產生整體成功標記。", ""]
    with (run / "report.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(md))


def failed_report(run, failure):
    """Diagnostic projection of partial attempts; never a completion receipt or valid comparison."""
    partial, problems = [], [{"status": "failed", **failure}]
    for path in (run / "raw").rglob("*.jsonl"):
        if path.name not in {"measurements.raw.jsonl", "results.scorer.jsonl"} and not path.name.endswith(".raw.jsonl"):
            continue
        try:
            partial.extend({"evidence_file": str(path.relative_to(run)), **flatten(row),
                            "result_committed": False, "comparison_eligible": False} for row in read_rows(path))
        except Exception as exc:
            problems.append({"file": str(path.relative_to(run)), "read_error": str(exc)})
    if not (run / "summary.xlsx").exists():
        workbook(run / "summary.xlsx", {"Overview": [{"status": "failed", "formal_result_complete": False}],
                 "Comparison": [{"status": "unavailable_failed_run"}], "Per Question": partial,
                 "Configuration": [], "Validation": problems})
    if not (run / "report.md").exists():
        with (run / "report.md").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("# 執行未完成\n\n此執行不構成正式研究結果，部分逐題紀錄未提交。\n\n"
                         + "```json\n" + json.dumps(failure, ensure_ascii=False, indent=2) + "\n```\n")
