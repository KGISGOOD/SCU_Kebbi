"""Unified offline Preflight/Smoke/Full/Resume and immutable report entry point."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from .common import (ARCHITECTURES, ROOT, SMOKE_IDS, DataError, atomic_json, code_hashes,
                     dataset, digest, environment, identifier, latest_results, lock, model_fields, pending,
                     read_json, read_run, resolve, save_result)
from .preflight import architecture_preflight, inspect_resources
from .report import generate_report, import_reviews


def load_config(root, args):
    config = read_json(resolve(root, args.config))
    expected = set(read_json(Path(__file__).with_name("config.json")))
    if set(config) != expected:
        raise DataError("config keys differ from evaluation/config.json")
    if args.allow_unreviewed:
        config["allow_unreviewed"] = True
    if args.device:
        config["device"] = args.device
    if args.precision:
        config["precision"] = args.precision
    if config["device"] not in ("cpu", "cuda") or config["precision"] not in ("fp16", "bf16", "fp32"):
        raise DataError("invalid device/precision")
    if config["device"] == "cpu" and config["precision"] != "fp32":
        raise DataError("CPU evaluation requires fp32")
    for key in ("max_new_tokens", "a5_max_input_tokens", "a2_max_length"):
        if type(config[key]) is not int or config[key] < 1:
            raise DataError("%s must be a positive integer" % key)
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise DataError("invalid seed")
    if type(config["allow_unreviewed"]) is not bool:
        raise DataError("allow_unreviewed must be bool")
    if config["bge_model"] != "BAAI/bge-m3":
        raise DataError("A0/A5 require the original BAAI/bge-m3, not a replacement embedding model")
    return config


def create_run(args):
    root = args.project_root.expanduser().resolve()
    if root != ROOT:
        raise DataError("--project-root must match the imported project; cd into the intended project first")
    config = load_config(root, args)
    run = resolve(root, args.run_dir or ("experiments/results/unified_faq/" + identifier()))
    if run.exists():
        raise DataError("run directory already exists; use resume or choose a NEW path: %s" % run)
    # No input/model directories may be used as output, even via a custom path.
    for protected in (root / "畢業專題_爬蟲", root / "experiments/models", root / "experiments/vector_stores",
                      root / "experiments/datasets", root / "robot", root / ".git",
                      *(root / "experiments" / name for name in ("shared", "baselines", "a2_faq_finetuning",
                                                                 "a3_faq_sft", "a5_rag_finetuned_llm", "evaluation"))):
        if run == protected or protected in run.parents:
            raise DataError("run directory must not be inside input/model/index directories")
    try:
        rows, source_hashes = dataset(root, config["manifest"])
    except Exception as exc:
        run.mkdir(parents=True, exist_ok=False)
        target = run / "preflight" / (identifier() + "_dataset_failed.json")
        atomic_json(target, {"dataset_status": "failed", "error": "%s: %s" % (type(exc).__name__, exc),
                             "model_loading_attempted": False})
        raise DataError("FAQ source/split validation failed; report=%s; %s" % (target, exc)) from exc
    resources = inspect_resources(root, config, rows)
    architectures = [a for a in ARCHITECTURES if a in (args.architectures or ARCHITECTURES)]
    run.mkdir(parents=True, exist_ok=False)
    meta = {"schema_version": "unified_faq_v1", "run_id": identifier(), "project_root": str(root),
            "mode": args.command, "architectures": architectures,
            "question_ids": [] if args.command == "preflight" else
                            list(SMOKE_IDS) if args.command == "smoke" else [q["faq_id"] for q in rows],
            "dataset": rows, "source_sha256": source_hashes, "config": config,
            "resources": resources, "code_sha256": code_hashes(root), "environment": environment(),
            "limitations": ["original canonical questions; not paraphrase generalization",
                            "A2/A3/A5 smoke weights are preliminary, inspect provenance",
                            "reference answers never passed to wrapper.answer",
                            "RAG may retrieve FAQ including held-out intents",
                            "prompt/template/EOS/truncation differ; inspect runtime contracts",
                            "no warmup; first answer includes first-call costs; model startup excluded"]}
    meta["run_contract_sha256"] = digest(meta)
    atomic_json(run / "run.json", meta)
    preflight = architecture_preflight(resources, architectures)
    atomic_json(run / "preflight" / (identifier() + ".json"), {
        "run_id": meta["run_id"], "source_sha256": source_hashes, "dataset_count": 60,
        "split_counts": {split: sum(q["split"] == split for q in rows) for split in ("train", "validation", "test")},
        "architectures": preflight, "resources": resources, "environment": meta["environment"],
        "resource_inventory_only": True})
    return run, meta, preflight


def launch(run, meta, architectures, retry_failed=False, load_only=False):
    """Sequential fresh Python processes, one resident architecture at a time."""
    failed = False
    for arch in architectures:
        questions = pending(run, meta, arch, retry_failed) if not load_only else []
        if not questions and not load_only:
            print("%s: no pending questions" % arch, flush=True)
            continue
        before = {key: row["attempt_id"] for key, row in latest_results(run).items()}
        log_dir = run / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / (arch + "_" + identifier() + ".log")
        command = [sys.executable, "-m", "experiments.evaluation.worker", "--run-dir", str(run), "--architecture", arch]
        if retry_failed:
            command.append("--retry-failed")
        if load_only:
            command.append("--load-only")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTHONHASHSEED=str(meta["config"]["seed"]), PYTHONUNBUFFERED="1",
                   PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        print("%s: starting isolated worker; log=%s" % (arch, log_path), flush=True)
        with log_path.open("xb") as handle:
            result = subprocess.run(command, cwd=meta["project_root"], env=env,
                                    stdout=handle, stderr=subprocess.STDOUT, check=False)
        failed = failed or result.returncode != 0
        if not load_only:
            after = latest_results(run)
            for question in questions:
                key = (arch, question["faq_id"])
                if key not in after or after[key]["attempt_id"] == before.get(key):
                    # Native crash/OOM/kill may bypass Python exception handling.
                    save_result(run, meta, arch, question, {
                        "status": "error", "error_message": "worker exited without a committed answer; code=%s; log=%s"
                                                        % (result.returncode, log_path),
                        **model_fields(meta, arch),
                    })
                    failed = True
        print("%s: worker exit=%d; GPU allocation released with process exit" % (arch, result.returncode), flush=True)
    if not load_only:
        latest = latest_results(run)
        failed = failed or any(row["status"] == "error" for (arch, faq_id), row in latest.items()
                               if arch in architectures and faq_id in meta["question_ids"])
    return failed


def runtime_preflight(run, meta):
    report = {}
    for arch in meta["architectures"]:
        jobs = [read_json(p) for p in sorted((run / "jobs" / arch).glob("*.json"))]
        report[arch] = {"model_load_verified": any(j["model_load_verified"] for j in jobs),
                        "job_statuses": [{"status": j["status"], "error": j.get("error")} for j in jobs],
                        "runtime_contract_available": (run / "contracts" / (arch + ".json")).is_file()}
    atomic_json(run / "preflight" / (identifier() + "_runtime.json"),
                {"run_id": meta["run_id"], "architectures": report, "resource_inventory_only": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for mode in ("preflight", "smoke", "full"):
        p = sub.add_parser(mode)
        p.add_argument("--project-root", type=Path, default=ROOT)
        p.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
        p.add_argument("--run-dir")
        p.add_argument("--architectures", nargs="+", choices=ARCHITECTURES)
        p.add_argument("--allow-unreviewed", action="store_true")
        p.add_argument("--device", choices=("cpu", "cuda"))
        p.add_argument("--precision", choices=("fp16", "bf16", "fp32"))
        if mode == "preflight":
            p.add_argument("--load-models", action="store_true", help="sequential model/index initialization, no answers")
    p = sub.add_parser("resume")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--architectures", nargs="+", choices=ARCHITECTURES)
    p.add_argument("--retry-failed", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--run-dir", type=Path, required=True)
    p = sub.add_parser("import-review")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--reviewer", required=True)
    args = parser.parse_args()
    try:
        if args.command in ("preflight", "smoke", "full"):
            run, meta, preflight = create_run(args)
            print("Run: %s" % run, flush=True)
            with lock(run / "run.lock"):
                if args.command == "preflight":
                    failed = any(r["status"] == "failed" for r in preflight.values())
                    if args.load_models:
                        ready = [a for a, r in preflight.items() if r["status"] != "failed"]
                        failed = launch(run, meta, ready, load_only=True) or failed
                        for arch in ready:
                            jobs = [read_json(p) for p in sorted((run / "jobs" / arch).glob("*.json"))]
                            if jobs:
                                last = jobs[-1]
                                preflight[arch].update(status=last["status"],
                                                       model_load_verified=last["model_load_verified"],
                                                       load_error=last.get("error"))
                            else:
                                preflight[arch].update(status="failed", load_error="worker crashed before load report")
                        atomic_json(run / "preflight" / (identifier() + "_loaded.json"),
                                    {"run_id": meta["run_id"], "architectures": preflight,
                                     "resource_inventory_only": False, "generation_executed": False})
                else:
                    failed = launch(run, meta, meta["architectures"])
                    runtime_preflight(run, meta)
                print("Report: %s" % generate_report(run), flush=True)
                return 2 if failed else 0
        run = args.run_dir.expanduser().resolve()
        meta = read_run(run)
        with lock(run / "run.lock"):
            if args.command == "resume":
                if meta["mode"] not in ("smoke", "full"):
                    raise DataError("preflight has no questions; create a smoke/full run")
                architectures = args.architectures or meta["architectures"]
                if not set(architectures) <= set(meta["architectures"]):
                    raise DataError("resume cannot add architectures to the original run")
                from .worker import verify_inputs
                verify_inputs(Path(meta["project_root"]), meta)
                failed = launch(run, meta, [a for a in ARCHITECTURES if a in architectures], args.retry_failed)
                runtime_preflight(run, meta)
                print("Report: %s" % generate_report(run), flush=True)
                return 2 if failed else 0
            if args.command == "import-review":
                print("Review import: %s" % import_reviews(run, args.csv.resolve(), args.reviewer))
            print("Report: %s" % generate_report(run), flush=True)
            return 0
    except KeyboardInterrupt:
        print("Interrupted. Committed attempts are retained; use resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print("ERROR [%s]: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
