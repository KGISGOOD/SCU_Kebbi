"""Private one-architecture subprocess; no model co-residency across architectures."""
from __future__ import annotations

import argparse
import random
import json
import sys
import time
import traceback
from pathlib import Path

from .common import (DataError, atomic_json, code_hashes, dataset, environment, identifier,
                     differences, lock, model_fields, pending, read_json, read_run, save_result)
from .preflight import architecture_preflight, inspect_resources


class InputChanged(DataError):
    def __init__(self, message, details=None):
        self.details = details or []
        super().__init__(message + ("; differences=" + json.dumps(self.details, ensure_ascii=True)
                                   if self.details else ""))


def verify_inputs(root, meta):
    try:
        rows, hashes = dataset(root, meta["config"]["manifest"])
    except Exception as exc:
        raise InputChanged("FAQ source/split can no longer be validated: %s" % exc) from exc
    changes = (differences(meta["dataset"], rows, "dataset")
               + differences(meta["source_sha256"], hashes, "source_sha256")
               + differences(meta["code_sha256"], code_hashes(root), "code_sha256")
               + differences(meta["environment"], environment(), "environment"))
    if changes:
        raise InputChanged("input identity changed; start a NEW run", changes)
    current = inspect_resources(root, meta["config"], rows)
    if current != meta["resources"]:
        raise InputChanged("model/index/candidate/raw-data resource identity changed; start a NEW run",
                           differences(meta["resources"], current, "resources"))
    return current


def seed(value):
    import numpy as np
    import torch
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    arch = args.architecture
    print("%s stage=startup" % arch, flush=True)
    meta = read_run(run)
    if arch not in meta["architectures"]:
        parser.error("architecture is not part of this run")
    root = Path(meta["project_root"])
    job_id = identifier()
    job = {"architecture": arch, "job_id": job_id, "load_only": args.load_only,
           "verify_only": args.verify_only, "stage": "verify_inputs",
           "status": "failed", "model_load_verified": False, "error": None}
    with lock(run / "locks" / (arch + ".lock")):
        questions = pending(run, meta, arch, args.retry_failed) if not (args.load_only or args.verify_only) else []
        if not questions and not (args.load_only or args.verify_only):
            return 0
        engine = None
        try:
            resources = verify_inputs(root, meta)
            if args.verify_only:
                job["status"] = "verified_no_model_load"
                atomic_json(run / "jobs" / arch / (job_id + ".json"), job)
                print("%s stage=verify_inputs passed; no model loaded" % arch, flush=True)
                return 0
            job["stage"] = "architecture_preflight"
            preflight = architecture_preflight(resources, [arch])[arch]
            if preflight["status"] == "failed":
                raise DataError("; ".join(preflight["errors"]))
            from .wrappers import Architecture, sync
            job["stage"] = "model_initialization"
            seed(meta["config"]["seed"])
            sync()
            started = time.perf_counter()
            engine = Architecture(arch, root, meta["config"], resources)
            sync()
            job.update(model_initialization_seconds=time.perf_counter() - started,
                       model_load_verified=True, metadata=engine.metadata)
            contract_path = run / "contracts" / (arch + ".json")
            job["stage"] = "runtime_contract"
            if contract_path.exists():
                if read_json(contract_path) != engine.metadata:
                    raise InputChanged("runtime model/hardware contract changed on resume",
                                       differences(read_json(contract_path), engine.metadata, "runtime_contract"))
            else:
                atomic_json(contract_path, engine.metadata)
            if args.load_only:
                job["status"] = "loaded_no_inference"
            else:
                error_count = 0
                for question in questions:
                    job["stage"] = "inference:" + question["faq_id"]
                    # Seed per query makes retry/subset execution independent of prior calls.
                    seed(meta["config"]["seed"])
                    payload = {"status": "error", "error_message": None,
                               "job_id": job_id,
                               "model_name": engine.model_name, "model_version_or_path": engine.model_version,
                               "model_provisional": engine.provisional, "runtime_contract": str(contract_path)}
                    sync()
                    started = time.perf_counter()
                    try:
                        # Neither the gold FAQ ID nor reference_answer enters any wrapper.
                        answer = engine.answer(question["question"])
                        payload.update(answer)
                        if not isinstance(answer.get("generated_answer"), str) or not answer["generated_answer"].strip():
                            raise DataError("empty/missing answer; not counted as success")
                        payload["status"] = "success"
                    except Exception as exc:
                        error_count += 1
                        payload.update(engine.partial)
                        if engine.capture:
                            payload.update(engine.capture.last)
                        payload.update(status="error", error_message="%s: %s" % (type(exc).__name__, exc),
                                       traceback=traceback.format_exc(), failure_stage=job["stage"])
                        print("%s stage=%s: %s\n%s" % (arch, job["stage"], payload["error_message"],
                                                       payload["traceback"]), file=sys.stderr, flush=True)
                    finally:
                        try:
                            sync()
                        except Exception:
                            pass
                        payload["latency_seconds"] = time.perf_counter() - started
                    # Disk serialization/fsync is outside per-question inference latency.
                    save_result(run, meta, arch, question, payload)
                    print("%s %s %s" % (arch, question["faq_id"], payload["status"]), flush=True)
                # Detect resource changes during model execution before declaring a job complete.
                job["stage"] = "post_execution_verify_inputs"
                verify_inputs(root, meta)
                job["status"] = "completed_with_errors" if error_count else "completed"
        except BaseException as exc:
            job.update(error="%s: %s" % (type(exc).__name__, exc), traceback=traceback.format_exc())
            print("%s stage=%s: %s\n%s" % (arch, job["stage"], job["error"], job["traceback"]),
                  file=sys.stderr, flush=True)
            if isinstance(exc, InputChanged):
                atomic_json(run / "integrity_violations" / (job_id + ".json"),
                            {"architecture": arch, "error": job["error"], "job_id": job_id,
                             "stage": job["stage"], "differences": exc.details})
            if not (args.load_only or args.verify_only):
                # Commit initialization failures, retaining already completed questions.
                for question in pending(run, meta, arch, args.retry_failed):
                    save_result(run, meta, arch, question, {
                        "status": "error", "error_message": job["error"], "traceback": job["traceback"], "job_id": job_id,
                        "failure_stage": job["stage"],
                        **model_fields(meta, arch),
                    })
        atomic_json(run / "jobs" / arch / (job_id + ".json"), job)
    return 0 if job["status"] in ("completed", "loaded_no_inference") else 2


if __name__ == "__main__":
    raise SystemExit(main())
