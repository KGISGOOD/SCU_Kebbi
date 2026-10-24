"""Independent research QA entry; legacy evaluation.run remains unchanged."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from types import SimpleNamespace
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

from .common import (ROOT, locate, new_output, read_rows, read_json, write_json, write_rows,
                     fingerprint, sha, verify_bundle)
from .runtime import load_input, verify_input, model_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--config", required=True, help="existing evaluation config, unchanged format")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--view", help="same merged review view used at export")
    parser.add_argument("--architectures", default="all", help="all or comma-separated A0,A1,A2-1,...")
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--resume-from", help="prior QA benchmark; reuse only complete committed architectures with identical inputs")
    args = parser.parse_args()
    root = args.project_root.resolve()
    if root != ROOT:
        raise ValueError("project-root must match imported project; cd to intended checkout")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    directory, config_path = locate(root, args.adapter_dir), locate(root, args.config)
    view = locate(root, args.view) if args.view else None
    manifest, phases, questions, receipt = load_input(root, directory, "qa", config_path,
                                                      args.allow_unreviewed, view)
    from experiments.evaluation.common import ARCHITECTURES, FAQ_ARCHITECTURES
    from experiments.evaluation.run import load_config
    from .qa_worker import identity
    args.from_run = args.bge_snapshot = args.device = args.precision = args.a5_max_input_tokens = None
    config = load_config(root, args)
    if not manifest["provisional"] and config["allow_unreviewed"]:
        raise ValueError("formal QA must disable allow_unreviewed checkpoint gate")
    arches = list(ARCHITECTURES) if args.architectures == "all" else args.architectures.split(",")
    if not arches or len(set(arches)) != len(arches) or not set(arches) <= set(ARCHITECTURES):
        raise ValueError("invalid/duplicate architecture IDs")
    run = new_output(root, args.run_dir)
    try:
        cases = phases["measured"]
        write_json(run / "requests.json", [model_request(r) for r in cases])
        contract = {"project_root": str(root), "config": config, "architectures": arches,
                    "provisional": manifest["provisional"], "identity": identity(root, config),
                    "requests_sha256": sha(run / "requests.json")}
        write_json(run / "identity.json", contract["identity"])
        write_json(run / "worker_contract.json", {**contract, "contract_sha256": fingerprint(contract)})
        write_json(run / "run.json", {"runner": "research_qa_v1", "split": manifest["split"],
                   "purpose": manifest["purpose"], "provisional": manifest["provisional"],
                   "research_input": receipt, "architectures": arches, "config": config,
                   "scoring": "manual answer review; no inferred single-gold FAQ labels"})
        all_rows, summaries, failed = [], {}, []
        previous = locate(root, args.resume_from) if args.resume_from else None
        if previous:
            old_contract = read_json(previous / "worker_contract.json")
            unsigned = {k: v for k, v in old_contract.items() if k != "contract_sha256"}
            if old_contract.get("contract_sha256") != fingerprint(unsigned):
                raise ValueError("previous QA contract hash changed")
            if any(old_contract.get(k) != contract.get(k) for k in contract):
                raise ValueError("QA resume model/config/environment/requests changed")
            old_manifest = verify_bundle(previous.parent / "adapter", "adapter_manifest.json")
            for key in ("dataset_identity", "dataset_hashes", "review_view_sha256", "settings_sha256", "purpose", "split"):
                if old_manifest.get(key) != manifest.get(key):
                    raise ValueError("QA resume review/data contract changed: " + key)
        for arch in arches:
            reuse = False
            if (previous and (previous / (arch + ".completed.json")).is_file()
                    and (previous / (arch + ".raw.jsonl")).is_file()
                    and not (previous / (arch + ".failed.json")).exists()):
                done = read_json(previous / (arch + ".completed.json"))
                old_raw = previous / (arch + ".raw.jsonl")
                rows = read_rows(old_raw)
                reuse = (done.get("status") == "completed" and done.get("source_verification") == "passed"
                         and done.get("raw_sha256") == sha(old_raw) and done.get("count") == len(cases)
                         and len(rows) == len(cases) and all(r.get("status") == "success" for r in rows)
                         and [r["question_id"] for r in rows] == [r["question_id"] for r in cases])
                if reuse:
                    shutil.copyfile(old_raw, run / old_raw.name)
                    shutil.copyfile(previous / (arch + ".completed.json"), run / (arch + ".completed.json"))
                    write_json(run / (arch + ".reused.json"), {"source": str(previous), "raw_sha256": sha(old_raw),
                               "completion_sha256": sha(previous / (arch + ".completed.json"))})
            with (run / (arch + ".log")).open("x", encoding="utf-8", newline="\n") as log:
                env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
                if reuse:
                    log.write("Reused verified committed architecture from " + str(previous) + "\n")
                    process = SimpleNamespace(returncode=0)
                else:
                    process = subprocess.run([sys.executable, "-u", "-m", "experiments.research_tools.qa_worker",
                                          "--run-dir", str(run), "--architecture", arch], cwd=root,
                                         env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            completion = run / (arch + ".completed.json")
            committed = process.returncode == 0 and completion.is_file()
            raw = run / (arch + ".raw.jsonl")
            rows = read_rows(raw) if raw.is_file() else []
            if committed:
                done = read_json(completion)
                committed = (done["raw_sha256"] == sha(raw) and done["count"] == len(cases)
                             and len(rows) == len(cases)
                             and [r["question_id"] for r in rows] == [r["question_id"] for r in cases])
            if not committed:
                failed.append(arch)
            for row in rows:
                q = questions[row["question_id"]]
                case = next(c for c in cases if c["question_id"] == row["question_id"])
                row.update(research_metadata=case["research_metadata"], result_committed=committed,
                           gold_faq_id=None, gold_faq_status="unavailable_no_explicit_human_gold_mapping",
                           faq_accuracy=None, answer_quality_status="needs_review",
                           reference_answer=q["effective_reference_answer"],
                           reference_answer_review_status=q["effective_answer_status"],
                           expected_behavior=q["expected_behavior"],
                           generation_metrics_status="not_applicable" if arch in FAQ_ARCHITECTURES else "available_if_captured")
                if arch in FAQ_ARCHITECTURES:
                    row.update(generation_seconds=None, generated_tokens=None, tokens_per_second=None)
                all_rows.append(row)
            valid = [r for r in rows if committed and r["status"] == "success"]
            summaries[arch] = {"worker_exit": process.returncode, "committed": committed,
                              "queries": len(rows), "successful_committed": len(valid),
                              "mean_end_to_end_seconds": statistics.mean(r["latency_seconds"] for r in valid) if valid else None,
                              "generation_speed_comparison": "not_applicable" if arch in FAQ_ARCHITECTURES else "different_architecture_descriptive_only",
                              "answer_quality": "unverified", "faq_accuracy": "unavailable_no_human_gold_mapping"}
            detail = read_json(run / (arch + ".failed.json")) if (run / (arch + ".failed.json")).is_file() else {}
            print(f"{arch} exit={process.returncode} committed={committed} {detail.get('stage', '')} {detail.get('error', '')}", flush=True)
        # Only the parent attaches scoring information, after all model calls.
        write_rows(run / "results.scorer.jsonl", all_rows)
        with (run / "answer_review.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            fields = ["architecture", "question_id", "question", "generated_answer", "reference_answer",
                      "reference_answer_review_status", "answer_quality_status", "result_committed"]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        write_json(run / "summary.json", {"architectures": summaries, "provisional": manifest["provisional"],
                   "answer_quality_verified": False, "question_id_is_not_gold_faq_id": True})
        verify_input(root, directory, "qa", config_path, args.allow_unreviewed, view, receipt)
        from .qa_worker import verify
        verify(root, contract)
        if failed:
            raise ValueError(f"workers not committed: {failed}; inspect per-architecture logs/failed receipts")
        write_json(run / "completed.json", {"status": "completed", "source_verification": "passed",
                   "provisional": manifest["provisional"], "answer_quality_verified": False,
                   "research_input": receipt})
    except BaseException as exc:
        write_json(run / "failed.json", {"error": str(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
