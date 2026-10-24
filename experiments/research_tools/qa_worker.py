"""Isolated QA architecture process. Requests contain only ID and question text."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback

from .common import read_json, write_json, fingerprint, sha, tool_identity
from .runtime import append_row
from experiments.evaluation.common import (code_hashes, dataset, environment, differences,
                                          environment_diagnostics)
from experiments.evaluation.preflight import inspect_resources, architecture_preflight


def identity(root, config):
    # Original FAQ60 is used ONLY to validate existing resources, never as the
    # research query list or as labels for arbitrary research Question IDs.
    canonical, sources = dataset(root, config["manifest"])
    return {"environment": environment(), "code_sha256": code_hashes(root),
            "research_tools": tool_identity(), "original_sources": sources,
            "resources": inspect_resources(root, config, canonical)}


def verify(root, meta):
    changes = differences(meta["identity"], identity(root, meta["config"]))
    if changes:
        raise ValueError("input identity changed: " + str(changes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    args = parser.parse_args()
    run, arch = args.run_dir.resolve(), args.architecture
    stage = "verify_contract"
    try:
        meta = read_json(run / "worker_contract.json")
        if meta.pop("contract_sha256") != fingerprint(meta) or arch not in meta["architectures"]:
            raise ValueError("worker contract changed or unexpected architecture")
        root = Path(meta["project_root"])
        job_path = run / "requests.json"
        if sha(job_path) != meta["requests_sha256"]:
            raise ValueError("request file changed")
        requests = read_json(job_path)
        if any(set(r) != {"question_id", "question"} for r in requests):
            raise ValueError("worker request contains non-allowlisted fields")
        stage = "verify_inputs"
        verify(root, meta)
        stage = "architecture_preflight"
        preflight = architecture_preflight(meta["identity"]["resources"], [arch])[arch]
        if preflight["status"] == "failed":
            raise ValueError(str(preflight["errors"]))
        from experiments.evaluation.worker import seed
        from experiments.evaluation.wrappers import Architecture, sync
        seed(meta["config"]["seed"])
        stage = "model_load"
        started = time.perf_counter()
        engine = Architecture(arch, root, meta["config"], meta["identity"]["resources"])
        sync()
        if engine.provisional and not meta["provisional"]:
            raise ValueError("provisional checkpoint cannot produce formal research results")
        write_json(run / (arch + ".model.json"), {"metadata": engine.metadata,
                   "provisional": engine.provisional, "load_seconds": time.perf_counter() - started,
                   "environment_diagnostics": environment_diagnostics()})
        errors = 0
        for request in requests:
            stage = "answer"
            sync()
            started = time.perf_counter()
            try:
                # The entire call boundary is a single query string.
                result = engine.answer(request["question"])
                if not isinstance(result.get("generated_answer"), str) or not result["generated_answer"].strip():
                    raise ValueError("empty answer")
                error, trace = None, None
            except Exception as exc:
                errors += 1
                result = {**engine.partial, **(engine.capture.last if engine.capture else {})}
                error, trace = str(exc), traceback.format_exc()
                print(f"{arch} {request['question_id']} stage=answer: {error}\n{trace}", flush=True)
            sync()
            append_row(run / (arch + ".raw.jsonl"), {**request, **result, "architecture": arch,
                       "status": "error" if error else "success", "error": error, "traceback": trace,
                       "latency_seconds": time.perf_counter() - started, "provisional": meta["provisional"],
                       "checkpoint_provisional": engine.provisional})
        stage = "post_execution_verify_inputs"
        verify(root, meta)
        if sha(job_path) != meta["requests_sha256"]:
            raise ValueError("requests changed during inference")
        if errors:
            raise ValueError(f"{errors} query failures; raw attempts retained")
        write_json(run / (arch + ".completed.json"), {"status": "completed", "count": len(requests),
                   "source_verification": "passed", "raw_sha256": sha(run / (arch + ".raw.jsonl"))})
        return 0
    except BaseException as exc:
        trace = traceback.format_exc()
        write_json(run / (arch + ".failed.json"), {"architecture": arch, "stage": stage,
                   "error": str(exc), "traceback": trace})
        print(f"{arch} stage={stage}: {exc}\n{trace}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
