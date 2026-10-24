"""Guarded orchestration of existing benchmarks; public reports, private raw evidence."""
from __future__ import annotations

import argparse
from argparse import Namespace
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

from .common import (ROOT, load_dataset, locate, new_output, read_json, sha,
                     tool_identity, verify_bundle, write_json)
from .adapters import build, get_views
from .suites import prepare


def completed(path):
    if (path / "failed.json").exists():
        raise ValueError(f"failed benchmark: {path}")
    receipt = read_json(path / "completed.json")
    if receipt.get("status") != "completed" or receipt.get("source_verification") != "passed":
        raise ValueError(f"missing successful source verification: {path}")
    return receipt


def certification(root, value, data, view):
    """External human attestation, never generated or upgraded by this runner."""
    if not value or view is None:
        raise ValueError("formal run blocked: supply --view AND --gold-certification after human review")
    path = locate(root, value)
    cert = read_json(path)
    if (cert.get("schema") != "research_gold_certification_v1"
            or cert.get("human_gold_certified") is not True
            or cert.get("dataset_identity") != data["identity"]
            or cert.get("review_view_sha256") != sha(view / "view_manifest.json")
            or not isinstance(cert.get("reviewer"), str) or not cert["reviewer"].strip()
            or not isinstance(cert.get("reviewed_at"), str) or not cert["reviewed_at"].strip()):
        raise ValueError("human certification missing, incomplete or belongs to another dataset/view")
    get_views(data, view)  # Existing immutable source and reviewed-view checks remain authoritative.
    return path, sha(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--dataset", choices=("experiment_suites_v1", "experiment_suites_v1_r4"), default="experiment_suites_v1")
    parser.add_argument("--target", choices=("qa", "sc", "sd", "ti", "int"), required=True)
    parser.add_argument("--stage", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--view")
    parser.add_argument("--gold-certification")
    parser.add_argument("--calibration", help="completed public Calibration run from this entry")
    parser.add_argument("--functional", action="store_true", help="explicit provisional execution, never formal")
    parser.add_argument("--limit", type=int, help="functional QA/SD only")
    parser.add_argument("--groups", type=int, help="functional SC/INT/TI complete-group prefix only")
    parser.add_argument("--cache-state", choices=("cold", "prefilled"), default="cold")
    args = parser.parse_args()
    root = args.project_root.resolve()
    if root != ROOT:
        raise ValueError("project-root must match the imported checkout")
    if args.target == "qa" and args.stage != "evaluation":
        raise ValueError("QA has one exposed fixed 60-question evaluation, no Calibration dataset")
    if not args.functional and (args.limit is not None or args.groups is not None):
        raise ValueError("formal runs cannot use Smoke prefixes")
    config = locate(root, args.config)
    data = load_dataset(root, args.dataset)
    view = locate(root, args.view) if args.view else None
    config_hash, code = sha(config), tool_identity()
    cert = certification(root, args.gold_certification, data, view) if not args.functional else None
    view_hash = get_views(data, view)[3]
    previous = None
    if args.stage == "evaluation" and args.target != "qa":
        if not args.calibration:
            raise ValueError("Evaluation needs --calibration; never refit on Evaluation")
        previous = locate(root, args.calibration)
        prior = verify_bundle(previous / "raw", "evidence.json")
        context = read_json(previous / "raw/context.json")
        if (context["target"] != args.target or context["stage"] != "calibration"
                or context["functional"] != args.functional or context["config_sha256"] != config_hash
                or context["dataset_identity"] != data["identity"] or context["view_sha256"] != view_hash
                or context["cache_state"] != args.cache_state or context["code"] != code
                or prior["dataset_identity"] != data["identity"]):
            raise ValueError("Calibration target/data/config/review/code/cache policy mismatch")
        if not (previous / "summary.xlsx").is_file() or not (previous / "report.md").is_file():
            raise ValueError("Calibration reporting was not completed")
        if any(sha(previous / name) != digest for name, digest in prior["report_sha256"].items()):
            raise ValueError("Calibration report changed after evidence was committed")
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex
    run = new_output(root, root / "experiments/results/algorithms/research" / (args.target + "_" + run_id))
    raw = run / "raw"
    raw.mkdir()
    stage = "adapter_preflight"
    try:
        context = {"dataset_name": args.dataset, "target": args.target, "stage": args.stage, "functional": args.functional,
                   "human_gold_certified": bool(cert), "dataset_identity": data["identity"],
                   "dataset_hashes": data["hashes"], "suite_revision": data["suite"]["revision"],
                   "config_sha256": config_hash, "view_sha256": view_hash, "code": code,
                   "cache_state": args.cache_state, "created_utc": run_id,
                   "calibration": str(previous) if previous else None,
                   "quality_policy": "Gold input review does not verify generated answers"}
        write_json(raw / "context.json", context)
        # Byte-exact copies: paths remain resolved against project-root, never against raw/.
        (raw / "config.snapshot.json").write_bytes(config.read_bytes())
        if cert:
            (raw / "human_certification.snapshot.json").write_bytes(cert[0].read_bytes())
        if previous:
            write_json(raw / "calibration.receipts.json", {
                str(p.relative_to(previous)): sha(p) for p in previous.rglob("*")
                if p.is_file() and (p.name in {"completed.json", "evidence.json", "context.json", "thresholds.json", "adapter_manifest.json"})})
        adapter = raw / "adapter"
        purpose = "functional" if args.functional else args.stage
        selection = Namespace(target=args.target, split="historical" if args.target == "qa" else args.stage,
                              purpose=purpose, ids=None, types=None, workload_id=None, initial_fill_ids=None,
                              groups=args.groups, limit=args.limit, suite_cache_state=args.cache_state,
                              view=str(view) if view else None, settings=str(config),
                              calibration_export=str(previous / "raw/adapter") if previous else None,
                              calibration_run=str(previous / "raw/benchmark") if previous else None)
        selected = prepare(selection, data)
        adapter.mkdir()
        build(selection, root, selected, adapter)
        write_json(adapter / "completed.json", {"status": "completed", "kind": "data_adapter_only_no_model_run"})
        manifest = verify_bundle(adapter, "adapter_manifest.json")
        expected = {"qa": 60, "sc": 40 if args.stage == "calibration" else 80,
                    "sd": 20 if args.stage == "calibration" else 40,
                    "ti": 24 if args.stage == "calibration" else 36,
                    "int": 20 if args.stage == "calibration" else 40}[args.target]
        if not args.functional and (manifest["provisional"] or manifest["count"] != expected):
            raise ValueError(f"formal input incomplete: expected {expected}, got {manifest['count']}")
        prefix = [sys.executable, "-m"]
        common = ["--project-root", str(root), "--config", str(config)]
        allow = ["--allow-unreviewed"] if args.functional else []
        benchmark = raw / "benchmark"
        if args.target in {"qa", "int"}:
            command = prefix + [f"experiments.research_tools.{args.target}_benchmark"] + common
            command += ["--adapter-dir", str(adapter), "--run-dir", str(benchmark)] + allow
            command += ["--architectures", "all"] if args.target == "qa" else ["--mode", "all"]
            if view:
                command += ["--view", str(view)]
        elif args.target == "ti":
            command = prefix + ["experiments.algorithms.topic_identification.benchmark"] + common
            command += ["--run-dir", str(benchmark), "--mode", "detection",
                        "--phase", "calibrate" if args.stage == "calibration" else "evaluate",
                        "--calibration" if args.stage == "calibration" else "--evaluation",
                        str(adapter / (args.stage + ".jsonl"))] + allow
            if previous:
                command += ["--thresholds", str(previous / "raw/benchmark/thresholds.json")]
        else:
            module = "semantic_cache" if args.target == "sc" else "speculative_decoding"
            command = prefix + [f"experiments.algorithms.{module}.benchmark"] + common
            command += ["--run-dir", str(benchmark), "--scenarios", str(adapter / "scenarios.jsonl"), "--mode", "all"] + allow
        commands = [command]
        if args.target == "sd":
            # Legacy 'all' intentionally excludes lookup. Keep its CLI stable and run BOTH paired studies.
            lookup = list(command)
            lookup[lookup.index("--run-dir") + 1] = str(raw / "prompt_lookup")
            lookup[lookup.index("--mode") + 1] = "sd2_4"
            commands.append(lookup)
        write_json(raw / "commands.json", commands)
        for index, command in enumerate(commands):
            stage = "benchmark_" + str(index + 1)
            print(f"{args.target} {args.stage}: {stage}", flush=True)
            with (raw / f"worker_{index + 1}.log").open("x", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
            if result.returncode:
                raise ValueError(f"worker exit={result.returncode}; see raw/worker_{index + 1}.log")
            destination = Path(command[command.index("--run-dir") + 1])
            completed(destination)
            if read_json(destination / "run.json").get("provisional", True) != args.functional:
                raise ValueError("benchmark provisional status differs from guarded input")
        stage = "final_source_verification"
        if (load_dataset(root, args.dataset)["identity"] != data["identity"]
                or sha(config) != config_hash or get_views(data, view)[3] != view_hash
                or tool_identity() != code or (cert and sha(cert[0]) != cert[1])):
            raise ValueError("source/config/review/certification/code changed during run")
        if previous:
            verify_bundle(previous / "raw", "evidence.json")
        stage = "excel_and_report"
        from .reporting import generate
        generate(run, context)
        stage = "evidence_receipt"
        write_json(raw / "evidence.json", {"dataset_identity": data["identity"],
                   "files_sha256": {str(p.relative_to(raw)): sha(p) for p in raw.rglob("*") if p.is_file()},
                   "report_sha256": {name: sha(run / name) for name in ("summary.xlsx", "report.md")}})
        write_json(raw / "completed.json", {"status": "completed", "source_verification": "passed",
                   "provisional": args.functional, "reports_complete": True})
        print(f"完成：{run / 'summary.xlsx'}\n報告：{run / 'report.md'}", flush=True)
    except BaseException as exc:
        failure = {"stage": stage, "error": str(exc), "traceback": traceback.format_exc(),
                   "formal_result_complete": False}
        write_json(raw / "failed.json", failure)
        try:
            from .reporting import failed_report
            failed_report(run, failure)
        except Exception as report_error:
            write_json(raw / "report_failure.json", {"error": str(report_error), "traceback": traceback.format_exc()})
        print(f"FAILED stage={stage}: {exc}\n證據保留：{raw}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
