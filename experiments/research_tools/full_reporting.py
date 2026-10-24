"""One Chinese workbook over verified child runs; never invent quality labels."""
from collections import defaultdict
import math
import statistics

from .common import read_json, read_rows
from .reporting import generate, workbook, text, flatten


def percentile(values, fraction):
    if len(values) < 20:
        return None
    values = sorted(values)
    at = (len(values) - 1) * fraction
    low = math.floor(at)
    return values[low] + (values[math.ceil(at)] - values[low]) * (at - low)


def report(run, state, identity):
    sheets = {name: [] for name in ("研究總覽", "QA 架構比較", "QA 逐題比較", "Semantic Cache",
              "Speculative Decoding", "Topic Identification", "Integrated Efficiency",
              "跨實驗同題比較", "研究設定與限制")}
    wide, qa_groups, cross = {}, defaultdict(list), []
    seen = {f"{prefix}{n:02d}" for prefix in ("P", "S") for n in range(1, 22)}
    for job in state["jobs"]:
        sheets["研究總覽"].append({"實驗": job["id"], "狀態": job["status"],
            "模式": identity["mode"], "錯誤": job.get("error"), "嘗試次數": len(job["attempts"]),
            "結果位置": job.get("result"), "品質": "未評定"})
        if job["status"] != "completed":
            # Preserve a readable view of partial QA answers, without committing them.
            if job["target"] == "qa" and job.get("result"):
                path = run / job["result"] / "raw/benchmark/results.scorer.jsonl"
                if path.is_file():
                    for row in read_rows(path):
                        key = row["question_id"]
                        dst = wide.setdefault(key, {"Question ID": key, "問題": row["question"]})
                        dst[row["architecture"] + "答案"] = row.get("generated_answer")
                        dst[row["architecture"] + "狀態"] = "未提交／不計分"
            continue
        child = run / job["result"]
        context = read_json(child / "raw/context.json")
        projection = generate(child, context, projection_only=True)
        target, stage = job["target"], job["stage"]
        sheet = {"qa": "QA 架構比較", "sc": "Semantic Cache", "sd": "Speculative Decoding",
                 "ti": "Topic Identification", "int": "Integrated Efficiency"}[target]
        for row in projection["comparison"]:
            visible = {k: v for k, v in row.items() if not (identity["mode"] == "provisional"
                and any(w in k.lower() for w in ("accuracy", "precision", "recall", "f1", "confusion", "false_reuse", "correct")))}
            sheets[sheet].append({"紀錄類型": "模式統計", "階段": stage, "Cache狀態": job["cache_state"],
                                  "品質判定": "未評定", **visible})
        detail = projection["detail"]
        for row in detail:
            case = row.get("source_case", {})
            qid = case.get("question_id", row.get("question_id"))
            measured = not row.get("warmup", False) and row.get("phase", "measured") == "measured"
            if target == "qa":
                exposure = "已見意圖" if qid in seen else "保留意圖" if qid in {f"{p}{n:02d}" for p in ("P", "S") for n in range(1, 31)} else "待確認"
                dst = wide.setdefault(qid, {"Question ID": qid, "問題": row.get("question"), "曝光": exposure})
                arch = row["architecture"]
                dst.update({arch + "答案": row.get("generated_answer"), arch + "延遲秒": row.get("latency_seconds"),
                            arch + "品質": "未評定", arch + "狀態": row.get("status"),
                            arch + "模型provisional": row.get("checkpoint_provisional")})
                if row.get("result_committed") and row.get("status") == "success":
                    qa_groups[(arch, exposure)].append(row["latency_seconds"])
            elif target == "sd":
                sheets[sheet].append({"紀錄類型": "逐題", "階段": stage, "Question ID": qid,
                    "問題": case.get("question"), "模式": row.get("mode"), "子實驗": row.get("substudy"),
                    "Warmup": row.get("warmup"), "Repeat": row.get("repeat_index"),
                    "輸入Token": row.get("input_token_count"), "生成Token": row.get("generated_token_count"),
                    "生成秒": row.get("generation_seconds"), "答案": row.get("output_text"),
                    "原始紀錄": row, "品質": "未評定；Token一致不代表正確"})
            elif target == "ti":
                gold = row.get("gold_shift")
                sheets[sheet].append({"紀錄類型": "逐輪", "階段": stage, "對話": case.get("conversation_id"),
                    "回合ID": row.get("scenario_id"), "問題": case.get("question"), "歷史": case.get("history"),
                    "方法": row.get("mode"), "切換預測": row.get("prediction_shift"),
                    "標籤": gold, "標籤身分": "候選，未評定" if identity["mode"] == "provisional" else "已批准",
                    "正確與否": None if identity["mode"] == "provisional" else not row.get("classification_error"),
                    "Threshold": row.get("threshold"), "辨識秒": row.get("feature_detection_seconds")})
            elif target in {"sc", "int"}:
                event = case.get("workload_event", {})
                sheets[sheet].append({"紀錄類型": "逐題", "階段": stage, "Cache狀態": job["cache_state"],
                    "Question ID": qid, "題組": event.get("group_id"), "順序": event.get("group_position"),
                    "模式": row.get("mode"), "Phase": row.get("phase"), "Warmup": row.get("warmup"),
                    "Hit": row.get("hit", row.get("cache_hit")), "答案": row.get("answer", row.get("output_text")),
                    "延遲秒": row.get("end_to_end_seconds", row.get("timings", {}).get("end_to_end_seconds")),
                    "品質判定": "未評定", "原始紀錄": row})
            if qid and measured:
                cross.append({"Question ID": qid, "資料版本": context["suite_revision"], "實驗": target,
                    "階段": stage, "模式": row.get("mode", row.get("architecture")),
                    "Cache狀態": job["cache_state"], "子實驗": row.get("substudy"),
                    "問題": case.get("question", row.get("question")),
                    "計時意義": "QA架構E2E" if target == "qa" else "受控生成；共享檢索成本" if target == "sd" else "主題辨識" if target == "ti" else "線上E2E",
                    "限制": "只對齊題目身分，不合併延遲或計算跨實驗加速比"})
        if target in {"sc", "int"}:
            for mode in sorted({r.get("mode", "") for r in detail}):
                times = [r.get("end_to_end_seconds", r.get("timings", {}).get("end_to_end_seconds")) for r in detail
                         if r.get("mode") == mode and not r.get("warmup", False) and r.get("phase", "measured") == "measured"]
                times = [v for v in times if isinstance(v, (float, int))]
                sheets[sheet].append({"紀錄類型": "Measured延遲分布", "階段": stage, "模式": mode,
                    "Cache狀態": job["cache_state"], "樣本數": len(times),
                    "平均秒": statistics.mean(times) if times else None, "P50秒": statistics.median(times) if times else None,
                    "P95秒": percentile(times, .95), "P95限制": "少於20筆不估計"})
            sheets[sheet].extend({"紀錄類型": "完整四問群組", "階段": stage, "Cache狀態": job["cache_state"], **r}
                                 for r in projection["groups"])
        for warning in projection["warnings"]:
            sheets["研究設定與限制"].append({"實驗": job["id"], "限制": warning})
    sheets["QA 逐題比較"] = list(wide.values())
    for name in ("a2_1_model", "a2_2_model", "a3_adapter"):
        resource = identity["qa"]["resources"].get(name, {})
        value = resource.get("value", {})
        sheets["研究總覽"].append({"模型": name, "位置": value.get("path"), "狀態": resource.get("status"),
            "provisional": value.get("provenance", {}).get("provisional"), "資格": "參見原始收據及正式前置驗證"})
    for (arch, exposure), times in qa_groups.items():
        sheets["QA 架構比較"].append({"紀錄類型": "曝光分層", "架構": arch, "曝光": exposure,
            "有效樣本數": len(times), "平均延遲秒": statistics.mean(times), "品質": "未評定"})
    # Only repeated exact IDs/text/version are shown together. No fuzzy matching.
    counts = defaultdict(set)
    for row in cross:
        counts[(row["Question ID"], row["資料版本"], row["問題"])].add(row["實驗"])
    sheets["跨實驗同題比較"] = [r for r in cross if len(counts[(r["Question ID"], r["資料版本"], r["問題"])]) > 1]
    sheets["研究設定與限制"].append({"身分與設定": identity,
        "限制": "所有生成答案仍需逐答案品質評估；資料Gold不等於輸出正確。TI只量測Detection，非Context Reduction。QA包含42個已見及18個保留意圖。"})
    sheets["研究設定與限制"].extend({"項目": key, "內容": value} for key, value in flatten({
        "硬體": identity.get("hardware"), "設定": identity["configs"], "環境": identity["qa"]["environment"],
        "Gold": identity.get("gold_sha256"), "View": identity.get("view_sha256")}).items())
    workbook(run / "summary.xlsx", sheets, presentation=True)
    done = sum(j["status"] == "completed" for j in state["jobs"])
    lines = ["# 資科萬事通整體研究結果", "", f"執行模式：{identity['mode']}；完成階段：{done}/{len(state['jobs'])}。",
             "", "只有全部必要階段及報告完成才代表整體成功。Provisional是完整預備實驗，不是正式研究成果。",
             "生成答案品質尚未評定，因此不能判定是否以品質換取速度；Token一致性也不代表答案正確。", "",
             "| 實驗 | 狀態 | 錯誤 |", "|---|---|---|"]
    for j in state["jobs"]:
        lines.append(f"| {j['id']} | {j['status']} | {text(j.get('error', '')).replace('|', '/').replace(chr(10), ' ')} |")
    for name in ("QA 架構比較", "Semantic Cache", "Speculative Decoding", "Topic Identification", "Integrated Efficiency"):
        lines += ["", f"## {name}", "", "以下是已提交紀錄的描述統計；Calibration不是最終Evaluation。"]
        for row in sheets[name]:
            if row.get("紀錄類型") not in {"逐題", "逐輪"}:
                lines.append("- " + text(row))
                for metric, value in flatten(row).items():
                    if "speedup" in metric and isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                        observation = "該次描述比值顯示較快" if value > 1 else "該次描述比值顯示較慢" if value < 1 else "該次描述比值未顯示加速"
                        lines.append(f"  - {metric} = {value:.4f}：{observation}；不代表顯著性或品質提升。")
    lines += ["", "## 解讀限制", "", "加速與否請依各方法的同題、同子實驗Baseline數值判讀；不合併Cold與Prefilled、不同計時範圍或兩次SD0。",
              "分類候選標籤、未審核重用及生成品質不作正式正確率。完整原始輸出、模型來源、設定及錯誤均保留於raw。"]
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
