"""Single entry for audited test inference, isolated smoke/full workers and reports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import uuid

from .common import ROOT, INPUT, GUIDE, code_hashes, environment, fingerprint, path, read, rows, save, sha, verify
from .data import exposure, jobs, load, normalize, review


def validate_config(config):
    expected = {"models", "training_bundle", "dense_index", "device", "qa_precision", "sd_precision", "int_precision",
                "seed", "max_new_tokens", "max_input_tokens", "repeats", "warmups", "sc_threshold",
                "faq_threshold", "review_file", "ti_thresholds", "settings_origin"}
    if set(config) != expected or set(config["models"]) != {"base", "bge", "draft", "reranker", "a2_1", "a2_2", "a3"}:
        raise ValueError("use the exact config.example.json schema; no legacy/smoke defaults")
    if config["device"] != "cuda":
        raise ValueError("this benchmark profile is for the GPU computer; use --check-only on development PC")
    if any(config[k] not in {"fp16", "bf16", "fp32"} for k in ("qa_precision", "sd_precision", "int_precision")):
        raise ValueError("invalid explicit precision")
    for k in ("seed", "max_new_tokens", "max_input_tokens", "repeats", "warmups"):
        if type(config[k]) is not int or config[k] < (0 if k == "seed" else 1):
            raise ValueError("invalid " + k)
    if config["max_input_tokens"] > 4096 or not -1 <= config["sc_threshold"] <= 1:
        raise ValueError("benchmark input cap must be <=4096; cosine threshold must be in [-1, 1]")
    if config["settings_origin"] != "predeclared_before_804_evaluation_not_optimized_on_test":
        raise ValueError("settings must be frozen before test; no threshold search/calibration entry")


def dependencies(job):
    if job["kind"] == "unsupported":
        return set()
    if job["kind"] == "qa":
        return {"A0": {"base", "bge", "index"}, "A1": {"bge"}, "A2-1": {"a2_1"},
                "A2-2": {"a2_2"}, "A3": {"base", "a3"}, "A4-1": {"base", "bge", "index"},
                "A4-2": {"base", "bge", "reranker", "index"}, "A5": {"base", "bge", "a3", "index"}}[job["architecture"]]
    result = {"base", "bge", "index"}
    if job.get("draft") or job.get("decoder") == "draft_1b":
        result.add("draft")
    return result


def execute(root, run, job, smoke):
    folder = run / "raw/jobs" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / (("smoke_" if smoke else "worker_") + uuid.uuid4().hex[:12] + ".log")
    command = [sys.executable, "-u", "-B", "-m", "experiments.benchmark_804.worker", "--project-root", str(root),
               "--run-dir", str(run), "--job", job["id"]] + (["--smoke-only"] if smoke else [])
    with log.open("x", encoding="utf-8", newline="\n") as f:
        print("Starting %s %s; log=%s" % (job["id"], "smoke" if smoke else "full", log), flush=True)
        process = subprocess.run(command, cwd=str(root), stdout=f, stderr=subprocess.STDOUT,
                                 env=dict(__import__("os").environ, PYTHONIOENCODING="utf-8",
                                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1"))
    status = read(folder / "status.json") if (folder / "status.json").exists() else {
        "status": "failed", "error": "worker stopped before committing status; see log"}
    status.update(exit_code=process.returncode, log=str(log.relative_to(run)), phase="smoke" if smoke else "full")
    print("%s %s: %s%s" % (job["id"], status["phase"], status["status"],
                            "; " + str(status.get("error", "inspect raw worker log")) if process.returncode else ""), flush=True)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--config", default="experiments/benchmark_804/config.local.json")
    p.add_argument("--resume", type=Path, help="existing run, immutable settings/identities required")
    p.add_argument("--check-only", action="store_true", help="CPU data/exposure audit only, no model/pickle loads")
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--allow-provisional", action="store_true", help="explicitly permits provisional trained models; never human Gold/Formal")
    p.add_argument("--allow-overlap-for-measurement", action="store_true", help="explicitly retain/report exposed test cases for measurement; never independent generalization")
    p.add_argument("--report-only", type=Path)
    args = p.parse_args()
    root = args.project_root.resolve()
    if args.report_only:
        from .report import generate
        target = path(root, args.report_only)
        if (root / "experiments/results/benchmark_804").resolve() not in target.parents:
            raise ValueError("report-only expects this benchmark's independent project Run")
        generate(target, root)
        return 0
    config_path = path(root, args.config)
    config = read(config_path)
    validate_config(config)
    data, audit = load(root)
    annotations = review(root, config, data)
    if args.resume:
        run = path(root, args.resume)
        if (root / "experiments/results/benchmark_804").resolve() not in run.parents:
            raise ValueError("resume must be inside experiments/results/benchmark_804")
        identity = read(run / "raw/identity.json")
        if read(run / "raw/config.json") != config:
            raise ValueError("resume config differs; start NEW run, do not replace original evidence")
        if args.allow_provisional != identity["allow_provisional"] or args.allow_overlap_for_measurement != identity["allow_overlap"]:
            raise ValueError("resume must preserve explicit qualification/overlap policy flags")
        verify(identity)
        if args.check_only:
            raise ValueError("--check-only creates an independent audit, not a resume")
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12]
        run = root / "experiments/results/benchmark_804" / run_id
        run.mkdir(parents=True, exist_ok=False)
        save(run / "run.json", {"run_id": run_id, "status": "preflight", "purpose": "held-out test inference/evidence",
                                "formal_quality_certified": False}, replace=False)
        save(run / "raw/config.json", config)
        save(run / "raw/questions.snapshot.json", data)
        save(run / "raw/annotations.snapshot.json", annotations)
        canonical, findings, hashes = exposure(root, config, data)
        canonical_ids = {r["faq_id"] for r in canonical}
        for qid, annotation in annotations.get("questions", {}).items():
            if annotation.get("gold_faq_id") is not None and annotation["gold_faq_id"] not in canonical_ids:
                raise ValueError("Gold FAQ ID not in V2 candidate universe (Question ID is not FAQ ID): " + qid)
        audit.update(exposure=findings, exact_train_overlap=sum(bool(r["exact_train_question_ids"]) for r in findings),
                     exact_faq_overlap=sum(r["exact_faq_candidate_id"] is not None for r in findings),
                     near_train_overlap=sum(bool(r["near_lexical_train_ids"]) for r in findings),
                     near_policy="trigram Jaccard>=0.85 is an audit flag, not semantic leakage proof",
                     declared_test_contamination_allowed=args.allow_overlap_for_measurement)
        save(run / "raw/faq_corpus.json", canonical)
        # All smoke inputs come from existing V2 development data, not any 804 question.
        test_text = {normalize(r["question"]) for r in data}
        development = [r for r in rows(path(root, config["training_bundle"]) / "validation.jsonl")
                       if normalize(r["messages"][1]["content"]) not in test_text]
        if len(development) < 2:
            raise ValueError("need separate development smoke inputs; do not tune using test questions")
        dev = [{"id": "DEV_" + r["faq_id"], "question": r["messages"][1]["content"]} for r in development[:2]]
        save(run / "raw/development.json", dev)
        plan = jobs(data, annotations, config)
        issues, resources = {}, {}
        if not args.check_only:
            from .resources import qualify, audit_index
            resources, issues, model_hashes = qualify(root, config, args.allow_provisional)
            hashes.update(model_hashes)
            try:
                audit["rag_index"], index_hashes = audit_index(root, config, data)
                hashes.update(index_hashes)
            except Exception as exc:
                issues["index"] = str(exc)
        else:
            audit["rag_index"] = {"status": "not_loaded_in_data_only_audit"}
        save(run / "raw/resources.json", resources)
        save(run / "raw/resource_issues.json", issues)
        save(run / "raw/data_audit.json", audit)
        save(run / "raw/plan.json", plan)
        for rel in (INPUT, GUIDE):
            hashes[str(root / rel)] = sha(root / rel)
        for rel, value in code_hashes(root).items():
            hashes[str(root / rel)] = value
        for f in [config_path] + list((run / "raw").glob("*.json")):
            hashes[str(f)] = sha(f)
        for key in ("review_file", "ti_thresholds"):
            if config.get(key):
                f = path(root, config[key])
                hashes[str(f)] = sha(f)
                if key == "ti_thresholds":
                    for name in ("completed.json", "calibration.snapshot.json"):
                        sibling = f.parent / name
                        hashes[str(sibling)] = sha(sibling)
        identity = {"files_sha256": hashes, "environment": environment(),
                    "allow_provisional": args.allow_provisional, "allow_overlap": args.allow_overlap_for_measurement}
        save(run / "raw/identity.json", identity)
        save(run / "raw/host_diagnostic.json", {"platform_display_only": platform.platform(),
                                               "note": "Windows display version not an environment blocker"})
    print("Run: " + str(run), flush=True)
    from .report import generate
    if args.check_only:
        save(run / "run.json", dict(read(run / "run.json"), status="data_audit_only_not_model_verified"), replace=True)
        generate(run, root)
        return 0
    audit = read(run / "raw/data_audit.json")
    contamination = audit["exact_train_overlap"] or audit["exact_faq_overlap"] or audit.get("rag_index", {}).get("full_test_question_matches")
    if contamination and not identity["allow_overlap"]:
        save(run / "run.json", dict(read(run / "run.json"), status="blocked_overlap_review_required"), replace=True)
        generate(run, root)
        print("Blocked: review raw/data_audit.json. Exposed cases cannot be claimed unseen. No test rows removed.")
        return 2
    plan, issues = read(run / "raw/plan.json"), read(run / "raw/resource_issues.json")
    statuses = read(run / "raw/stages.json") if (run / "raw/stages.json").exists() else {}
    for job in plan:
        missing = sorted(dependencies(job) & set(issues))
        if job["kind"] == "unsupported":
            statuses[job["id"]] = {"status": "unavailable", "reason": job["reason"]}
        elif missing:
            statuses[job["id"]] = {"status": "unavailable", "reason": {k: issues[k] for k in missing}}
        else:
            previous = statuses.get(job["id"], {})
            if previous.get("status") == "completed_measurements" and not args.smoke_only:
                continue
            smoke = execute(root, run, job, True)
            if smoke["exit_code"]:
                statuses[job["id"]] = smoke
            elif args.smoke_only:
                statuses[job["id"]] = smoke
            elif job.get("available") is False:
                statuses[job["id"]] = {"status": "unavailable", "reason": job["reason"], "independent_feature_smoke": smoke}
            else:
                statuses[job["id"]] = execute(root, run, job, False)
        save(run / "raw/stages.json", statuses, replace=True)
    verify(identity)
    failed = any(s["status"] == "failed" or s.get("exit_code", 0) != 0 or
                 (name.startswith("qa_") and s["status"] == "unavailable") for name, s in statuses.items())
    overall = "failed_or_partial" if failed else "smoke_only" if args.smoke_only else "completed_available_experiments_pending_quality"
    save(run / "run.json", dict(read(run / "run.json"), status=overall, formal_quality_certified=False), replace=True)
    verdict = generate(run, root)
    if verdict["missing_completed_indices"]:
        failed = True
        save(run / "run.json", dict(read(run / "run.json"), status="failed_missing_completed_evidence"), replace=True)
        generate(run, root)
    if verdict["token_guard_failures"]:
        save(run / "run.json", dict(read(run / "run.json"), status="failed_token_guard_measurements_retained"), replace=True)
        generate(run, root)
    if not args.smoke_only and not failed and not verdict["token_guard_failures"]:
        save(run / "completed.json", {"status": overall, "formal_quality_certified": False,
                                      "unavailable": [k for k, v in statuses.items() if v["status"] == "unavailable"],
                                      "quality_scoring": "pending_verified_annotation"}, replace=True)
    return 2 if failed or verdict["token_guard_failures"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(2)
