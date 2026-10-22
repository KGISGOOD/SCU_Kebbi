"""Private one-architecture subprocess; no model co-residency across architectures."""
from __future__ import annotations

import argparse
import random
import time
import traceback
from pathlib import Path

from .common import (DataError, atomic_json, code_hashes, dataset, environment, identifier,
                     lock, model_fields, pending, read_json, read_run, save_result)
from .preflight import architecture_preflight, inspect_resources


class InputChanged(DataError):
    pass


def verify_inputs(root, meta):
    try:
        rows, hashes = dataset(root, meta["config"]["manifest"])
    except Exception as exc:
        raise InputChanged("FAQ source/split can no longer be validated: %s" % exc) from exc
    if rows != meta["dataset"] or hashes != meta["source_sha256"]:
        raise InputChanged("FAQ source/split changed; start a NEW run")
    if code_hashes(root) != meta["code_sha256"] or environment() != meta["environment"]:
        raise InputChanged("code/package/host environment changed; start a NEW run")
    current = inspect_resources(root, meta["config"], rows)
    if current != meta["resources"]:
        raise InputChanged("model/index/candidate/raw-data resource identity changed; start a NEW run")
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
    args = parser.parse_args()
    run = args.run_dir.resolve()
    meta = read_run(run)
    arch = args.architecture
    if arch not in meta["architectures"]:
        parser.error("architecture is not part of this run")
    root = Path(meta["project_root"])
    job_id = identifier()
    job = {"architecture": arch, "job_id": job_id, "load_only": args.load_only,
           "status": "failed", "model_load_verified": False, "error": None}
    with lock(run / "locks" / (arch + ".lock")):
        questions = pending(run, meta, arch, args.retry_failed) if not args.load_only else []
        if not questions and not args.load_only:
            return 0
        engine = None
        try:
            resources = verify_inputs(root, meta)
            preflight = architecture_preflight(resources, [arch])[arch]
            if preflight["status"] == "failed":
                raise DataError("; ".join(preflight["errors"]))
            from .wrappers import Architecture, sync
            seed(meta["config"]["seed"])
            sync()
            started = time.perf_counter()
            engine = Architecture(arch, root, meta["config"], resources)
            sync()
            job.update(model_initialization_seconds=time.perf_counter() - started,
                       model_load_verified=True, metadata=engine.metadata)
            contract_path = run / "contracts" / (arch + ".json")
            if contract_path.exists():
                if read_json(contract_path) != engine.metadata:
                    raise InputChanged("runtime model/hardware contract changed on resume")
            else:
                atomic_json(contract_path, engine.metadata)
            if args.load_only:
                job["status"] = "loaded_no_inference"
            else:
                error_count = 0
                for question in questions:
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
                                       traceback=traceback.format_exc())
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
                verify_inputs(root, meta)
                job["status"] = "completed_with_errors" if error_count else "completed"
        except BaseException as exc:
            job.update(error="%s: %s" % (type(exc).__name__, exc), traceback=traceback.format_exc())
            if isinstance(exc, InputChanged):
                atomic_json(run / "integrity_violations" / (job_id + ".json"),
                            {"architecture": arch, "error": job["error"], "job_id": job_id})
            if not args.load_only:
                # Commit initialization failures, retaining already completed questions.
                for question in pending(run, meta, arch, args.retry_failed):
                    save_result(run, meta, arch, question, {
                        "status": "error", "error_message": job["error"], "traceback": job["traceback"], "job_id": job_id,
                        **model_fields(meta, arch),
                    })
        atomic_json(run / "jobs" / arch / (job_id + ".json"), job)
    return 0 if job["status"] in ("completed", "loaded_no_inference") else 2


if __name__ == "__main__":
    raise SystemExit(main())
