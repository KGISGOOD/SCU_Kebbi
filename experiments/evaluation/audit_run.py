"""Read-only committed-attempt / post-verification evidence. No model imports."""
import argparse
import json
from pathlib import Path

from .common import (ARCHITECTURES, digest, latest_results, read_attempts,
                     read_json, read_run, sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="A5")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    try:
        meta = read_run(run)
        if args.architecture not in meta["architectures"]:
            raise ValueError("architecture was not selected in this frozen run")
        records = read_attempts(run)  # Strict run ID, source fields and content hashes.
        latest = latest_results(run)
        chosen = {faq_id: row for (arch, faq_id), row in latest.items() if arch == args.architecture}
        paths = {}
        for path in sorted((run / "attempts" / args.architecture).glob("*.json")):
            record = read_json(path)
            paths[record["attempt_id"]] = {"path": str(path), "file_sha256": sha(path)}
        jobs = [read_json(p) for p in sorted((run / "jobs" / args.architecture).glob("*.json"))]
        violations = [{"path": str(p), **read_json(p)}
                      for p in sorted((run / "integrity_violations").glob("*.json"))]
        post = [v for v in violations if v.get("architecture") == args.architecture
                and v.get("stage") == "post_execution_verify_inputs"]
        fields = [d["field"] for v in post for d in v.get("differences", [])]
        a1_only = (all(v.get("differences") for v in post)
                   and all(f.startswith(("resources.a1_canonical.", "resources.a1_index.")) for f in fields)) if post else None
        base = meta["resources"]["base"].get("value")
        context = {"status": "unavailable", "source": "cached config only; no model loaded"}
        if base:
            config_path = Path(base["path"]) / "config.json"
            try:
                if sha(config_path) != base["files_sha256"]["config.json"]:
                    raise ValueError("base config differs from frozen hash")
                config = read_json(config_path)
                context.update(status="source_hash_verified", model_type=config.get("model_type"),
                               max_position_embeddings=config.get("max_position_embeddings"),
                               rope_scaling=config.get("rope_scaling"))
            except Exception as exc:
                context["error"] = str(exc)
        failures = [faq_id for faq_id, r in sorted(chosen.items()) if r["status"] == "error"]
        missing = sorted(set(meta["question_ids"]) - set(chosen))
        report = {"run_id": meta["run_id"], "architecture": args.architecture,
                  "committed_record_validation": "passed", "read_only": True,
                  "committed_attempt_count": sum(r["architecture"] == args.architecture for r in records),
                  "latest_success_count": sum(r["status"] == "success" for r in chosen.values()),
                  "latest_failure_ids": failures, "missing_ids": missing,
                  "run_integrity_status": "flagged_do_not_silently_merge" if violations else "no_recorded_violation",
                  "post_execution_changes_only_a1": a1_only, "integrity_violations": violations,
                  "jobs": [{k: j.get(k) for k in ("job_id", "status", "stage", "error")} for j in jobs],
                  "a5_max_input_tokens": meta["config"]["a5_max_input_tokens"],
                  "max_new_tokens": meta["config"]["max_new_tokens"], "model_context": context,
                  "latest_attempts": [{"faq_id": faq_id, "status": r["status"],
                                       "attempt_id": r["attempt_id"], "record_sha256": r["record_sha256"],
                                       "answer_sha256": digest(r["generated_answer"]),
                                       "latency_seconds": r["latency_seconds"], "model_name": r["model_name"],
                                       "model_identity_sha256": digest(r["model_version_or_path"]),
                                       "error_message": r.get("error_message"), **paths[r["attempt_id"]]}
                                      for faq_id, r in sorted(chosen.items())]}
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 2 if failures or missing or violations else 0
    except Exception as exc:
        parser.exit(2, "AUDIT FAILED [%s]: %s\n" % (type(exc).__name__, exc))


if __name__ == "__main__":
    raise SystemExit(main())
