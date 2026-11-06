"""Single-architecture functional questions; existing QAEngine, no 804 allocation."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback

from experiments.benchmark_804.common import ROOT, path, read, rows, save, sha, verify, environment, code_hashes
from experiments.shared.first_round import verify_evidence, export, TIMING_DEFINITION

REQUIRED = {"A0": {"base", "bge"}, "A1": {"bge"},
            "A2-1": {"bge", "a2_1"}, "A2-2": {"bge", "a2_2"},
            "A3": {"base", "a3"}, "A4-1": {"base", "bge"},
            "A4-2": {"base", "bge", "reranker"}, "A5": {"base", "bge", "a3"}}


def load_questions(filename):
    cases = rows(filename)
    if not cases or len({r["question_id"] for r in cases}) != len(cases):
        raise ValueError("empty/duplicate custom question IDs")
    for row in cases:
        for key in ("question_id", "question", "reference_answer", "category", "expected_behavior", "review_status"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError("invalid custom question field: " + key)
        if row["review_status"] not in {"needs_review", "approved"}:
            raise ValueError("invalid review_status")
        if any(row.get(k) for k in ("history", "messages", "context")):
            raise ValueError("custom QA is single-turn only")
    return cases


def main(architecture, argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path("experiments/benchmark_804/config.local.json"))
    parser.add_argument("--questions-file", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="NEW child of experiments/results")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--allow-provisional", action="store_true")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    config_path, questions, run = (path(root, p) for p in (args.config, args.questions_file, args.run_dir))
    if (root / "experiments/results").resolve() not in run.parents or run.exists():
        raise ValueError("run-dir must be a NEW child of experiments/results")
    if args.repeats < 1:
        raise ValueError("repeats must be >=1")
    config = read(config_path)
    required = REQUIRED[architecture]
    if not required <= set(config.get("models", {})):
        raise ValueError("missing selected architecture model keys: " + str(required))
    for key in ("seed", "max_input_tokens", "max_new_tokens"):
        if type(config.get(key)) is not int or config[key] < (0 if key == "seed" else 1):
            raise ValueError("invalid " + key)
    if config.get("device") not in {"cuda", "cpu"} or config.get("qa_precision") not in {"fp16", "bf16", "fp32"}:
        raise ValueError("invalid device/qa_precision")
    if config["device"] == "cpu" and config["qa_precision"] != "fp32":
        raise ValueError("CPU requires fp32")
    if config["max_input_tokens"] > 4096:
        raise ValueError("reviewed input cap is 4096; no truncation")
    threshold = config.get("faq_threshold")
    if threshold is not None and (type(threshold) not in {int, float} or not math.isfinite(threshold) or not -1 <= threshold <= 1):
        raise ValueError("invalid faq_threshold")
    config.setdefault("faq_threshold", None)
    cases = load_questions(questions)
    if any(r["review_status"] != "approved" for r in cases) and not args.allow_unreviewed:
        raise ValueError("candidate labels require --allow-unreviewed; functional only, never Formal")
    hashes = {str(config_path): sha(config_path), str(questions): sha(questions)}
    hashes.update(verify_evidence(root, cases))
    hashes.update({str(root / relative): value for relative, value in code_hashes(root).items()})
    initial_environment = environment()
    run.mkdir(parents=True, exist_ok=False)
    save(run / "run.json", {"architecture": architecture, "purpose": "functional_custom_questions",
                           "formal_quality_certified": False, "human_gold_certified": False,
                           "question_count": len(cases), "repeats": args.repeats, "warmups": 0,
                           "first_round_summary_timing": TIMING_DEFINITION,
                           "timing_scope": "CUDA-synchronized answer call; excludes load, initialization, hashing and export",
                           "config": config})
    save(run / "questions.snapshot.json", cases)
    failures = 0
    try:
        for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[variable] = "1"
        from experiments.benchmark_804.resources import qualify, audit_index
        from experiments.benchmark_804.adapters import QAEngine
        resources, issues, model_hashes = qualify(root, config, args.allow_provisional, required=required)
        if issues or not required <= set(resources):
            save(run / "resource_issues.json", issues)
            raise ValueError("selected architecture resource qualification failed: " + str(issues))
        hashes.update(model_hashes)
        canonical = []
        if architecture in {"A1", "A2-1", "A2-2", "A3", "A5"}:
            from experiments.a2_faq_finetuning.v2_data import load_bundle, read_rows
            _, manifest, data_hashes = load_bundle(root, path(root, config["training_bundle"]))
            hashes.update(data_hashes)
            if architecture in {"A1", "A2-1", "A2-2"}:
                master = read_rows(path(root, manifest["frozen_dir"]) / "qa_master.jsonl")
                canonical = [{"faq_id": r["intent_group_id"], "question": r["question"], "answer": r["answer"],
                              "source_question_id": r["question_id"], "sources": r["sources"]}
                             for r in master if r["canonical_question_id"] == r["question_id"]]
        if architecture in {"A0", "A4-1", "A4-2", "A5"}:
            audit, index_hashes = audit_index(root, config, [{"id": r["question_id"], "question": r["question"]} for r in cases])
            save(run / "index_audit.json", audit)
            hashes.update(index_hashes)
        from experiments.shared.qa_runtime import measured
        start = time.perf_counter()
        engine = QAEngine(architecture, root, config, resources, canonical, args.allow_provisional)
        save(run / "identity.json", {"model": engine.metadata, "loading_seconds_excluded": time.perf_counter() - start,
                                    "input_sha256": hashes, "environment": initial_environment})
        # Only this string enters inference: labels, sources and reference answers never enter its Prompt.
        with (run / "results.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for repeat in range(args.repeats):
                for case in cases:
                    record = {"question_id": case["question_id"], "question": case["question"],
                              "architecture": architecture, "repeat_index": repeat, "warmup": False,
                              "quality_status": "unverified_functional_only"}
                    try:
                        value = measured(engine.answer, case["question"])
                        record.update(status="success", output=value, end_to_end_seconds=measured.seconds)
                    except Exception as error:
                        failures += 1
                        record.update(status="failed", error=str(error), traceback=traceback.format_exc(),
                                      end_to_end_seconds=getattr(measured, "seconds", None))
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    print(architecture, case["question_id"], "repeat", repeat, record["status"], flush=True)
        verify({"files_sha256": hashes, "environment": initial_environment})
        export(run, architecture)
        if failures:
            raise ValueError(str(failures) + " question failures; inspect preserved results")
        save(run / "completed.json", {"status": "completed", "count": len(cases) * args.repeats,
                                      "formal_quality_certified": False, "source_verification": "passed"})
        return 0
    except BaseException as error:
        save(run / "failed.json", {"status": "failed", "error": str(error), "traceback": traceback.format_exc()})
        raise
