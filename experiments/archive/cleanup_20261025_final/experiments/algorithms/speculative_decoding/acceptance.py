"""Read-only historical technical acceptance; no Gold/identity exceptions or models."""
from __future__ import annotations

import argparse
from pathlib import Path
import uuid

from .diagnose_logits import read, config_differences
from .reporting import write_json
from experiments.research_tools.common import verify_bundle, sha, fingerprint, read_rows
from experiments.research_tools.reporting import generate, workbook


ARCHES = ("A0", "A1", "A2-1", "A2-2", "A3", "A4-1", "A4-2", "A5")


def audit(source):
    from experiments.research_tools.run_all import plan
    state, identity = read(source / "raw/state.json"), read(source / "raw/identity.json")
    checks = []
    def record(name, ok, detail=None):
        checks.append({"檢查": name, "技術狀態": "通過" if ok is True else "失敗" if ok is False else "未驗證",
                       "證據": detail, "正式研究資格": "未認證；另依Formal/Gold Guard"})
    expected = plan()
    record("11階段排程", [{k: j.get(k) for k in p} for j, p in zip(state["jobs"], expected)] == expected
           and len(state["jobs"]) == len(expected), state.get("status"))
    for job in state["jobs"]:
        try:
            child = (source / job.get("result", "__missing__")).resolve()
            if source not in child.parents:
                raise ValueError("child path escapes original run")
            receipt = verify_bundle(child / "raw", "evidence.json")
            if fingerprint(receipt) != job.get("receipt") or job["status"] != "completed":
                raise ValueError("stage not committed or receipt mismatch")
            for name, digest in receipt["report_sha256"].items():
                path = (child / name).resolve()
                if child not in path.parents or sha(path) != digest:
                    raise ValueError("child report hash/path mismatch")
            context = read(child / "raw/context.json")
            if any(context.get(k) != job[k] for k in ("target", "stage", "cache_state")):
                raise ValueError("stage/split/cache identity mismatch")
            if context["dataset_identity"] != identity["dataset_identity"]:
                raise ValueError("stage dataset differs from full-run pinned identity")
            # Existing projection checks native completion, SD/lookup prompt
            # identities, TI frozen thresholds, and complete four-query groups.
            projection = generate(child, context, projection_only=True)
            full_counts = {"qa": 60, "sc": 40 if job["stage"] == "calibration" else 80,
                           "sd": 20 if job["stage"] == "calibration" else 40,
                           "ti": 24 if job["stage"] == "calibration" else 36,
                           "int": 20 if job["stage"] == "calibration" else 40}
            actual_count = projection["overview"][0]["input_queries"]
            record(job["id"] + " 非Smoke完整輸入", actual_count == full_counts[job["target"]],
                   {"expected": full_counts[job["target"]], "actual": actual_count,
                    "TI_note": "每組四輪的首輪不作切換判斷，故8/12組對應24/36次判斷"})
            record(job["id"], True, {"dataset": context["dataset_identity"],
                   "revision": context["suite_revision"], "rows": len(projection["detail"]),
                   "input_count": projection["overview"][0]["input_queries"]})
            benchmark = child / "raw/benchmark"
            if job["target"] == "sc":
                wanted = [c["scenario_id"] for c in read_rows(child / "raw/adapter/scenarios.jsonl")]
                for mode in ("none", "exact", "semantic"):
                    actual = [r["scenario_id"] for r in projection["detail"] if r.get("mode") == mode]
                    record(job["id"] + " " + mode + " 完整查詢序列", actual == wanted, {"expected": len(wanted), "actual": len(actual)})
            if job["target"] == "ti":
                wanted = {c["scenario_id"] for c in read_rows(child / "raw/adapter" / (job["stage"] + ".jsonl"))}
                methods = read(benchmark / ("thresholds.json" if job["stage"] == "calibration" else "thresholds.used.json"))["payload"]["methods"]
                for mode in methods:
                    actual = [r["scenario_id"] for r in projection["detail"] if r.get("mode") == mode]
                    record(job["id"] + " " + mode + " 全部對話判斷", set(actual) == wanted and len(actual) == len(wanted),
                           {"expected": len(wanted), "actual": len(actual)})
            if job["target"] == "int":
                config = read(child / "raw/config.snapshot.json")
                for phase in ("initial_fill", "measured"):
                    wanted = [c["scenario_id"] for c in read_rows(child / "raw/adapter" / (phase + ".jsonl"))]
                    for mode in ("INT0", "INT1", "INT2", "INT3"):
                        actual = [(r["scenario_id"], r["repeat_index"], r["warmup"]) for r in projection["detail"]
                                  if r.get("mode") == mode and r.get("phase") == phase]
                        expected_rows = {(qid, repeat, warmup) for qid in wanted for warmup, n in
                                         ((True, config["warmups"]), (False, config["repeats"])) for repeat in range(n)}
                        record(job["id"] + " " + mode + " " + phase, set(actual) == expected_rows and len(actual) == len(expected_rows),
                               {"expected": len(expected_rows), "actual": len(actual), "timing_scope": phase})
            if job["target"] == "qa":
                cases = read_rows(child / "raw/adapter/scenarios.jsonl")
                wanted = [c["question_id"] for c in cases]
                record("QA共同題目數60", len(wanted) == 60 and len(set(wanted)) == 60, wanted)
                for arch in ARCHES:
                    rows = [r for r in projection["detail"] if r.get("architecture") == arch]
                    record("QA " + arch + " 完整測試", [r["question_id"] for r in rows] == wanted
                           and all(r.get("result_committed") is True and r.get("status") == "success" for r in rows), len(rows))
                contract = read(benchmark / "worker_contract.json")
                resources = contract["identity"]["resources"]
                for key, steps in (("a2_1_model", 54), ("a2_2_model", 54), ("a3_adapter", 33)):
                    value = resources[key]["value"]
                    provenance = value["provenance"]
                    completion = provenance.get("completed_epoch", provenance.get("completed_epochs", provenance.get("epoch")))
                    if key == "a3_adapter":
                        completion_path = Path(value["path"]).parent / "completed.json"
                        if completion_path.is_file():
                            if sha(completion_path) != value.get("training_run_sha256", {}).get("completed.json"):
                                raise ValueError("A3 completion evidence differs from the recorded QA resource identity")
                            completion = read(completion_path).get("completed_epochs")
                    record(key + " 三Epoch及實際steps", completion == 3 and provenance.get("optimizer_steps") == steps,
                           {"path": value["path"], "epochs": completion, "steps": provenance.get("optimizer_steps"),
                            "provisional": provenance.get("provisional")})
                for arch in ("A2-1", "A2-2", "A3", "A5", "A4-1", "A4-2"):
                    path = benchmark / (arch + ".model.json")
                    if not path.is_file():
                        record(arch + " 實際載入來源", None, "缺少model metadata；不能由設定推定實際載入")
                        continue
                    meta = read(path)["metadata"]
                    key = {"A2-1": "a2_1_model", "A2-2": "a2_2_model", "A3": "a3_adapter", "A5": "a3_adapter"}.get(arch)
                    if arch.startswith("A4"):
                        actual = meta.get("embedding_model_version")
                        expected_value = resources["bge"]["value"]
                    elif arch.startswith("A2"):
                        actual = meta.get("model_version_or_path")
                        expected_value = resources[key]["value"]
                    else:
                        actual, expected_value = meta.get("adapter"), resources[key]["value"]
                    record(arch + " 實際載入來源", actual == expected_value, {"actual": actual, "expected": expected_value})
            if job["target"] == "sd":
                from experiments.research_tools.collect_provisional import sd_projection
                for sub in ("benchmark", "prompt_lookup"):
                    validation = []
                    comparisons, details, failures = sd_projection(child / "raw" / sub, job["id"], sub, validation)
                    modes = {d["模式"] for d in details}
                    required = {"SD0", "SD2-1"} | {f"SD2-2-k{k}" for k in (2, 4, 6, 8)} if sub == "benchmark" else {"SD0"} | {f"SD2-4-k{k}" for k in (2, 4, 6, 8)}
                    native = child / "raw" / sub
                    config = read(native / "run.json")["config"]
                    scenarios = read(native / "scenarios.snapshot.json")
                    wanted = {(c["scenario_id"], mode, repeat, warmup) for c in scenarios for mode in required
                              for warmup, n in ((True, config["warmups"]), (False, config["repeats"])) for repeat in range(n)}
                    actual = [(r["題目ID"], r["模式"], r["Repeat"], r["Warmup"]) for r in details]
                    record(job["id"] + " " + sub + " 每題每次完整性", set(actual) == wanted and len(actual) == len(wanted),
                           {"expected": len(wanted), "actual": len(actual)})
                    record(job["id"] + " " + sub + " 全模式Token驗證", bool(details) and modes == required
                           and not failures and all(c["完整Lossless通過"] for c in comparisons), validation)
        except Exception as exc:
            record(job["id"], False, str(exc))
    record("SD v2 獨立單題（不屬原Run）", None,
           "本表僅驗收原Run；未讀取獨立v2結果，狀態未驗證。單題通過也不能取代完整SD Evaluation。")
    record("正式研究資格", None, {"mode": identity.get("mode"),
           "gold_sha256": identity.get("gold_sha256"), "policy": "技術完成不等於Gold或訓練批准；本工具不簽認"})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--check-resume", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    def locate(p):
        return (p if p.is_absolute() else root / p).resolve()
    source = locate(args.source_run)
    base = root / "experiments/results/algorithms"
    if source == base or not source.is_relative_to(base):
        raise ValueError("source must be a managed run")
    output = base / ("technical_acceptance_" + uuid.uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    checks = audit(source)
    ready = False
    if args.check_resume:
        from experiments.research_tools.run_all import inventory, defaults
        from experiments.research_tools.common import load_dataset
        settings = read(source / "raw/settings.json")
        if args.config:
            supplied = read(locate(args.config))
            current = defaults(root)
            current["configs"].update(supplied.get("configs", {}))
            for key in ("view", "gold_certification"):
                current[key] = supplied.get(key, current[key])
            settings = current
        try:
            old = read(source / "raw/identity.json")
            view = locate(Path(settings["view"])) if settings.get("view") else None
            gold = locate(Path(settings["gold_certification"])) if settings.get("gold_certification") else None
            actual = inventory(root, settings, old["mode"], load_dataset(root, "experiment_suites_v1_r4"), view, gold)
            diffs = config_differences(old, actual)
            write_json(output / "resume_identity_diff.json", diffs)
            ready = not diffs
        except Exception as exc:
            write_json(output / "resume_identity_diff.json", {"error": str(exc)})
        checks.append({"檢查": "原Run Resume完整身分", "技術狀態": "通過" if ready else "失敗",
                       "證據": "見resume_identity_diff.json；不允許忽略程式或其他身分差異"})
    write_json(output / "acceptance.json", {"checks": checks, "resume_ready": ready,
               "formal_qualification": False, "source_run": str(source)})
    workbook(output / "summary.xlsx", {"基礎驗收": checks}, presentation=True)
    lines = ["# 基礎技術驗收（非正式研究認證）", "", "|檢查|技術狀態|證據|", "|---|---|---|"]
    for row in checks:
        detail = str(row.get("證據")).replace("|", "/").replace("\n", " ")
        lines.append(f"|{row['檢查']}|{row['技術狀態']}|{detail}|")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"技術驗收：{output}")
    technical = [r for r in checks if r["檢查"] not in {"正式研究資格", "SD v2 獨立單題（不屬原Run）"}]
    return (0 if ready else 2) if args.check_resume else (2 if any(r["技術狀態"] != "通過" for r in technical) else 0)


if __name__ == "__main__":
    raise SystemExit(main())
