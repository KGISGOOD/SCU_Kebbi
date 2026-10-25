"""Sequential full-scale research orchestration; existing guarded runners do the work."""
from __future__ import annotations

import argparse
from argparse import Namespace
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

from .common import ROOT, fingerprint, load_dataset, locate, new_output, read_json, sha, verify_bundle, write_json
from .run import certification

TARGETS = ("qa", "sc", "sd", "ti", "int")


def defaults(root):
    qa = "experiments/evaluation/config.gpu.local.json"
    return {"configs": {"qa": qa if (root / qa).is_file() else "experiments/evaluation/config.json",
        **{target: f"experiments/algorithms/{directory}/config.gpu.local.json" for target, directory in
           (("sc", "semantic_cache"), ("sd", "speculative_decoding"),
            ("ti", "topic_identification"), ("int", "integrated_efficiency"))}},
        "view": None, "gold_certification": None, "sd_fp32": False, "require_full_training": False}


def atomic(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


@contextmanager
def exclusive(path):
    # OS locks are released on crashes; an old lock file is not a stale lock.
    with path.open("a+b") as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def plan():
    jobs = [{"id": "qa_evaluation", "target": "qa", "stage": "evaluation", "cache_state": "cold", "depends": None}]
    for target, states in (("sc", ("cold",)), ("sd", ("cold",)), ("ti", ("cold",)), ("int", ("cold", "prefilled"))):
        for state in states:
            cal = f"{target}_{state}_calibration"
            for stage in ("calibration", "evaluation"):
                jobs.append({"id": f"{target}_{state}_{stage}", "target": target, "stage": stage,
                             "cache_state": state, "depends": cal if stage == "evaluation" else None})
    return jobs


def inventory(root, settings, mode, data, view, gold):
    """No model loading. Resolve/hash all existing resources and validate all input contracts first."""
    from experiments.evaluation.run import load_config
    from experiments.evaluation.preflight import architecture_preflight
    from experiments.evaluation.common import ARCHITECTURES
    from .qa_worker import identity as qa_identity
    from .common import tool_identity
    from .adapters import get_views, question_eligible, pair_label
    from .suites import prepare
    from experiments.algorithms.shared.a0_adapter import knowledge_hashes
    from experiments.evaluation.model_resources import inference_identity
    configs = {t: locate(root, settings["configs"][t]) for t in TARGETS}
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=20, check=True)
        hardware = {"gpu": query.stdout.strip(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    except (OSError, subprocess.SubprocessError) as exc:
        hardware = {"gpu": "unavailable", "diagnostic": type(exc).__name__}
    config_hashes = {t: sha(p) for t, p in configs.items()}
    qa_args = Namespace(config=str(configs["qa"]), from_run=None, allow_unreviewed=mode == "provisional",
                        bge_snapshot=None, device=None, precision=None, a5_max_input_tokens=None)
    qa_config = load_config(root, qa_args)
    if mode == "formal" and qa_config["allow_unreviewed"]:
        raise ValueError("formal QA config must disable allow_unreviewed")
    qa = qa_identity(root, qa_config)
    readiness = architecture_preflight(qa["resources"], ARCHITECTURES)
    errors = [f"{arch}: {r['errors']}" for arch, r in readiness.items() if r["status"] == "failed"]
    if settings.get("require_full_training"):
        for key, expected_steps in (("a2_1_model", 54), ("a2_2_model", 54), ("a3_adapter", 33)):
            value = qa["resources"].get(key, {}).get("value", {})
            provenance = value.get("provenance", {})
            epoch = provenance.get("completed_epoch")
            if key == "a3_adapter" and value.get("path"):
                epoch = read_json(Path(value["path"]).parent / "completed.json").get("completed_epochs")
            if epoch != 3 or provenance.get("optimizer_steps") != expected_steps or provenance.get("smoke_step_limit") is not None:
                errors.append(f"{key}: require the completed 3-Epoch/{expected_steps}-step model, not Smoke; got epoch={epoch}, steps={provenance.get('optimizer_steps')}")
    resources, signatures = {}, {}
    for target in ("sc", "sd", "ti", "int"):
        config = read_json(configs[target])
        resources[target] = {}
        for field in ("model", "bge") + (("draft_model",) if target in {"sd", "int"} else ()):
            path = locate(root, config[field])
            if not (path / "config.json").is_file():
                errors.append(f"{target}.{field}: complete LOCAL directory required: {path}")
                continue
            resources[target][field] = inference_identity(path, sentence_model=field == "bge")
        index = locate(root, config["knowledge_index"])
        resources[target]["knowledge_index"] = {"path": str(index), "files": knowledge_hashes(index)}
        signatures[target] = {key: config.get(key) for key in ("device", "precision", "seed", "max_new_tokens")}
    # All algorithm targets must use the same original Target/BGE/knowledge resources.
    for target in ("sd", "ti", "int"):
        for field in ("model", "bge", "knowledge_index"):
            if resources[target].get(field) != resources["sc"].get(field):
                errors.append(f"{target}.{field}: differs from SC; fix local paths/settings, no automatic substitution")
        if target == "sd" and settings.get("sd_fp32"):
            if signatures[target]["precision"] != "fp32" or any(signatures[target][k] != signatures["sc"][k]
                    for k in ("device", "seed", "max_new_tokens")):
                errors.append("SD FP32 control requires fp32 and identical device/seed/max_new_tokens to SC")
        elif signatures[target] != signatures["sc"]:
            errors.append(f"{target}: device/precision/seed/max_new_tokens differs from SC")
    approval_resources = {}
    if mode == "formal":
        certification(root, gold, data, view)
        questions, pairs, turns, _ = get_views(data, view)
        required = set(data["suite"]["qa"]["question_ids"])
        required.update(key for group in data["suite"]["sc"] for key in group["question_ids"])
        errors.extend("Gold question/answer not verified: " + key for key in sorted(required)
                      if not question_eligible(questions[key]))
        errors.extend("Gold turn not verified: " + key for key, turn in turns.items()
                      if turn["effective_turn_status"] != "verified")
        from .review import catalogue
        _, candidate_pairs = catalogue(data)
        for pair in candidate_pairs.values():
            if pair.get("required_for") and pair_label(pairs.get((pair["origin_question_id"], pair["current_question_id"]))) == "unverified":
                errors.append("Gold Cache pair not verified: " + pair["pair_id"])
        from .training_approval import verify
        for key in ("a2_1_model", "a2_2_model", "a3_adapter"):
            resource = qa["resources"][key]
            if resource["status"] != "ready":
                continue
            model = locate(root, qa_config[key])
            parent = model.parent
            try:
                if key.startswith("a2"):
                    config = read_json(parent / "run_config.json")["arguments"]
                    train = locate(root, config["training"])
                    manifest = locate(root, config["training_manifest"])
                    schema = "a2_training_review_v1"
                    from experiments.a2_faq_finetuning.prepare_review import validate_manifest
                    validate_manifest(root, manifest, train)
                else:
                    config = read_json(parent / "training_config.json")["arguments"]
                    directory = locate(root, config["data_dir"])
                    train, manifest = directory / "train.jsonl", directory / "dataset_manifest.json"
                    if read_json(manifest)["schema_version"] != "a3_sft_data_v2":
                        raise ValueError("A3 needs reviewed revised v2 answers")
                    schema = "a3_training_review_v1"
                    from experiments.a3_faq_sft.common import validate_data
                    validate_data(root, directory)
                verify(parent / "training_review.snapshot.json", schema, train, manifest,
                       [f"{p}{i:02d}" for p in ("P", "S") for i in range(1, 22)])
                approval_resources[key] = {"receipt": sha(parent / "training_review.snapshot.json"),
                                           "train": sha(train), "manifest": sha(manifest)}
            except Exception as exc:
                errors.append(f"{key}: formal Train approval/source validation: {exc}")
    if errors:
        raise ValueError("Preflight blocked:\n" + "\n".join(errors))
    # Check frozen input selection before starting costly workers. Actual Evaluation
    # receipts are checked later by the unmodified calibration/evaluation Guard.
    for job in plan():
        args = Namespace(target=job["target"], split="historical" if job["target"] == "qa" else job["stage"],
            purpose="functional" if mode == "provisional" else "calibration", ids=None, types=None,
            workload_id=None, initial_fill_ids=None, groups=None, limit=None, suite_cache_state=job["cache_state"],
            view=str(view) if view else None, settings=str(configs[job["target"]]),
            calibration_export=None, calibration_run=None)
        # Evaluation receipts do not exist yet: build() remains authoritative at execution.
        # Formal coverage is checked through the calibration-purpose Gold path here.
        prepare(args, data)
    return {"mode": mode, "dataset_identity": data["identity"], "dataset_hashes": data["hashes"],
            "configs": {t: {"path": str(p), "sha256": config_hashes[t]} for t, p in configs.items()},
            "view": str(view) if view else None, "view_sha256": get_views(data, view)[3],
            "gold": str(gold) if gold else None, "gold_sha256": sha(gold) if gold else None,
            "qa": qa, "algorithm_resources": resources, "settings": signatures, "code": tool_identity(),
            "readiness": readiness, "plan": plan(), "training_approvals": approval_resources, "hardware": hardware,
            "execution_policy": {"sd_fp32": bool(settings.get("sd_fp32")),
                "require_full_training": bool(settings.get("require_full_training")),
                "comparison_limit": "SD FP32 is an independent numerical control; do not pool cross-algorithm precision timings"}}


def validate_child(path, job=None):
    receipt = verify_bundle(path / "raw", "evidence.json")
    if job:
        context = read_json(path / "raw/context.json")
        if any(context.get(k) != job[k] for k in ("target", "stage", "cache_state")):
            raise ValueError("child receipt belongs to another scheduled stage")
    for name, digest in receipt["report_sha256"].items():
        if sha(path / name) != digest:
            raise ValueError("child report changed: " + name)
    return fingerprint(receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("provisional", "formal"), required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--config", help="optional machine-local run_all JSON; defaults to run_all.local.json when present")
    parser.add_argument("--view")
    parser.add_argument("--gold-certification")
    parser.add_argument("--resume", help="existing full Run directory; retries failed/interrupted stages, preserves all attempts")
    parser.add_argument("--sd-fp32", action="store_true")
    parser.add_argument("--require-full-training", action="store_true")
    parser.add_argument("--run-dir", help="NEW independent full Run; cannot combine with --resume")
    args = parser.parse_args()
    if args.resume and args.run_dir:
        parser.error("--run-dir cannot be combined with --resume")
    root = args.project_root.resolve()
    if root != ROOT:
        parser.error("cd to the intended checkout; imported project-root differs")
    settings = defaults(root)
    local = locate(root, args.config or "experiments/research_tools/run_all.local.json")
    if args.config or local.is_file():
        supplied = read_json(local)
        settings["configs"].update(supplied.get("configs", {}))
        for key in ("view", "gold_certification", "sd_fp32", "require_full_training"):
            if key in supplied:
                settings[key] = supplied[key]
    for key in ("sd_fp32", "require_full_training"):
        if getattr(args, key):
            settings[key] = True
        if type(settings[key]) is not bool:
            raise ValueError(key + " must be a JSON boolean")
    view = locate(root, args.view or settings["view"]) if args.view or settings["view"] else None
    gold = locate(root, args.gold_certification or settings["gold_certification"]) if args.gold_certification or settings["gold_certification"] else None
    base = root / "experiments/results/algorithms/research_full"
    if args.resume:
        run = locate(root, args.resume)
        if not run.is_relative_to(base) or not (run / "raw/state.json").is_file():
            parser.error("resume must name an existing managed research_full Run")
    else:
        run = new_output(root, args.run_dir or base / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex))
        (run / "raw").mkdir()
    raw = run / "raw"
    if args.resume and not args.config and not local.is_file():
        settings = read_json(raw / "settings.json")
        view = locate(root, args.view or settings["view"]) if args.view or settings["view"] else None
        gold = locate(root, args.gold_certification or settings["gold_certification"]) if args.gold_certification or settings["gold_certification"] else None
    with exclusive(raw / "active.lock"):
        session = uuid.uuid4().hex
        state = read_json(raw / "state.json") if args.resume else {
            "schema": "research_full_v1", "status": "preflight", "jobs": [dict(j, status="pending", attempts=[]) for j in plan()]}
        try:
            print("完整研究前置驗證（僅讀取／Hash；尚未載入模型）", flush=True)
            data = load_dataset(root, "experiment_suites_v1_r4")
            identity = inventory(root, settings, args.mode, data, view, gold)
            if args.resume:
                if read_json(raw / "identity.json") != identity:
                    raise ValueError("Resume identity changed: model/data/config/code/environment/review; start a NEW Run")
                if [{k: job[k] for k in definition} for job, definition in zip(state["jobs"], plan())] != plan() or len(state["jobs"]) != len(plan()):
                    raise ValueError("resume schedule was modified")
            else:
                write_json(raw / "identity.json", identity)
                write_json(raw / "settings.json", settings)
            state["status"] = "running"
            atomic(raw / "state.json", state)
            by_id = {j["id"]: j for j in state["jobs"]}
            for number, job in enumerate(state["jobs"], 1):
                if job["status"] == "completed":
                    if validate_child(run / job["result"], job) != job["receipt"]:
                        raise ValueError("successful child evidence changed: " + job["id"])
                    print(f"[{number}/{len(state['jobs'])}] 已驗證，跳過 {job['id']}", flush=True)
                    continue
                if job["depends"] and by_id[job["depends"]]["status"] != "completed":
                    job.update(status="blocked", error="Calibration not completed")
                    atomic(raw / "state.json", state)
                    continue
                prior_qa = next((run / value / "raw/benchmark" for value in reversed(job["attempts"])
                    if (run / value / "raw/benchmark/worker_contract.json").is_file()), None) if job["target"] == "qa" else None
                attempt = raw / "stages" / job["id"] / ("attempt_" + uuid.uuid4().hex)
                attempt.parent.mkdir(parents=True, exist_ok=True)
                job.update(status="running", result=attempt.relative_to(run).as_posix(), error=None)
                job["attempts"].append(job["result"])
                atomic(raw / "state.json", state)
                command = [sys.executable, "-u", "-m", "experiments.research_tools.run", "--project-root", str(root),
                    "--dataset", "experiment_suites_v1_r4", "--target", job["target"], "--stage", job["stage"],
                    "--config", str(locate(root, settings["configs"][job["target"]])),
                    "--cache-state", job["cache_state"], "--run-dir", str(attempt)]
                if args.mode == "provisional":
                    command.append("--functional")  # No limits: full workloads, explicitly provisional.
                else:
                    command += ["--view", str(view), "--gold-certification", str(gold)]
                if view and args.mode == "provisional":
                    command += ["--view", str(view)]
                if job["target"] == "sd" and settings.get("sd_fp32"):
                    command.append("--sd-fp32")
                if job["depends"]:
                    command += ["--calibration", str(run / by_id[job["depends"]]["result"])]
                if prior_qa:
                    command += ["--qa-resume-from", str(prior_qa)]
                log = attempt.parent / (attempt.name + ".log")
                print(f"[{number}/{len(state['jobs'])}] {job['id']}；完整題庫", flush=True)
                try:
                    write_json(attempt.parent / (attempt.name + ".command.json"), {"command": command})
                    with log.open("x", encoding="utf-8") as stream:
                        process = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
                    if process.returncode:
                        tail = log.read_text(encoding="utf-8", errors="strict")[-2500:]
                        raise ValueError(f"exit={process.returncode}; {log}\n{tail}")
                    job.update(receipt=validate_child(attempt, job), status="completed")
                except Exception as exc:
                    job.update(status="failed", error=str(exc), traceback=traceback.format_exc())
                    print(f"失敗 {job['id']}: {exc}", file=sys.stderr, flush=True)
                atomic(raw / "state.json", state)
            # Rehash after execution. Never commit a mixed-resource full experiment.
            if inventory(root, settings, args.mode, data, view, gold) != identity or load_dataset(root, "experiment_suites_v1_r4")["identity"] != data["identity"]:
                raise ValueError("research inputs changed during execution; full Run not committed")
            for name in ("summary.xlsx", "report.md", "raw/technical_acceptance.json"):
                if (run / name).exists():
                    archive = raw / "prior_reports" / session
                    archive.mkdir(parents=True, exist_ok=True)
                    destination = archive / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    (run / name).replace(destination)
            from .full_reporting import report
            state["technical_acceptance_passed"] = report(run, state, identity)
            state["status"] = "completed" if all(j["status"] == "completed" for j in state["jobs"]) and state["technical_acceptance_passed"] else "incomplete"
            atomic(raw / "state.json", state)
            write_json(raw / ("receipt_" + session + ".json"), {"status": state["status"],
                "technical_acceptance_passed": state["technical_acceptance_passed"],
                "technical_acceptance_sha256": sha(raw / "technical_acceptance.json"),
                "identity_sha256": sha(raw / "identity.json"), "state_sha256": sha(raw / "state.json"),
                "reports": {n: sha(run / n) for n in ("summary.xlsx", "report.md")}})
            print(f"{state['status']}：{run / 'summary.xlsx'}\n{run / 'report.md'}", flush=True)
            return 0 if state["status"] == "completed" and state["technical_acceptance_passed"] else 2
        except BaseException as exc:
            state["status"] = "failed"
            atomic(raw / "state.json", state)
            write_json(raw / ("failure_" + session + ".json"), {"error": str(exc), "traceback": traceback.format_exc()})
            if not (run / "summary.xlsx").exists() and not (run / "report.md").exists():
                from .reporting import workbook
                workbook(run / "summary.xlsx", {"研究總覽": [{"狀態": "失敗／未完成", "原因": str(exc)}],
                         "研究設定與限制": [{"模式": args.mode, "限制": "不得使用此Run宣稱完成；詳見raw錯誤證據"}]}, presentation=True)
                (run / "report.md").write_text("# 完整研究尚未完成\n\n" + str(exc) + "\n\n錯誤證據保留於raw。\n", encoding="utf-8")
            elif not (run / "report.md").exists():
                (run / "report.md").write_text("# 完整研究尚未完成\n\n報告彙整失敗：" + str(exc) + "\n詳見raw錯誤證據。\n", encoding="utf-8")
            print(f"總控失敗：{exc}\n證據：{raw}", file=sys.stderr, flush=True)
            return 130 if isinstance(exc, KeyboardInterrupt) else 2


if __name__ == "__main__":
    raise SystemExit(main())
