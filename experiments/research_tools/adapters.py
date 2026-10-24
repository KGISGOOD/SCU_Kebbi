"""Export research views to explicit benchmark contracts; never execute models."""
from __future__ import annotations

import argparse
import traceback

from .common import (ROOT, fingerprint, load_dataset, locate, new_output, read_json, read_rows,
                     sha, tool_identity, verify_bundle, write_json, write_rows)


def metadata(q):
    return {**{k: q[k] for k in ("question_id", "family_id", "split", "source_document_ids", "source_hash",
                                 "review_status", "training_overlap_status", "training_overlap_details")},
            "effective_question_status": q["effective_question_status"],
            "effective_answer_status": q["effective_answer_status"],
            "human_review_trace": {k: q.get("human_review", {}).get(k) for k in ("reviewer", "reviewed_at")}}


def get_views(data, view):
    if view is None:
        return ({q["question_id"]: {**q, "effective_question_status": "needs_review",
                  "effective_answer_status": "needs_review", "effective_reference_answer": q["reference_answer"]}
                 for q in data["questions"]}, {}, {t["turn_id"]: {**t, "effective_turn_status": "needs_review",
                    "effective_topic_label": t["same_topic_or_shift"],
                    "effective_required_history": t["required_history_turn_ids"]} for t in data["turns"]}, None)
    manifest = verify_bundle(view, "view_manifest.json")
    if manifest["dataset_identity"] != data["identity"] or manifest["dataset_hashes"] != data["hashes"]:
        raise ValueError("view belongs to a different dataset version")
    qrows = read_rows(view / "questions.view.jsonl")
    trows = read_rows(view / "turns.view.jsonl")
    prows = read_rows(view / "pairs.view.jsonl")
    if (len({r["question_id"] for r in qrows}) != len(qrows) or
            {r["question_id"] for r in qrows} != {r["question_id"] for r in data["questions"]} or
            len({r["turn_id"] for r in trows}) != len(trows) or
            {r["turn_id"] for r in trows} != {r["turn_id"] for r in data["turns"]}):
        raise ValueError("view IDs differ from source dataset")
    for original, rows, key in ((data["questions"], qrows, "question_id"), (data["turns"], trows, "turn_id")):
        index = {r[key]: r for r in rows}
        if any(any(index[r[key]].get(k) != v for k, v in r.items()) for r in original):
            raise ValueError("view changed original source fields")
    return ({r["question_id"]: r for r in qrows},
            {(r["origin_question_id"], r["current_question_id"]): r for r in prows},
            {r["turn_id"]: r for r in trows}, sha(view / "view_manifest.json"))


def select_questions(questions, split, ids, types, limit):
    if ids and not ids <= set(questions):
        raise ValueError(f"unknown question IDs: {sorted(ids - set(questions))}")
    if ids and any(questions[key]["split"] != split for key in ids):
        raise ValueError("specified IDs belong to another split")
    selected = [q for q in questions.values() if q["split"] == split and (not ids or q["question_id"] in ids)
                and (not types or q["primary_type"] in types)]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("selection is empty")
    return selected


def question_eligible(q):
    return q["effective_question_status"] == "verified" and q["effective_answer_status"] == "verified"


def pair_label(pair):
    if not pair or pair["human_review"]["reuse_status"] != "verified":
        return "unverified"
    return {"reusable": "allowed", "not_reusable": "forbidden", "unknown": "unverified"}[pair["human_review"]["reuse_decision"]]


def leaf_strings(value):
    if isinstance(value, dict):
        return set().union(*(leaf_strings(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(leaf_strings(v) for v in value))
    return {value} if isinstance(value, str) else set()


def sequence(data, questions, pairs, workload_id, split, limit, purpose):
    measured = [r for r in data["workloads"] if r["workload_id"] == workload_id]
    fills = [r for r in data["fills"] if r["workload_id"] == workload_id]
    if not measured or any(r["source_split"] != split for r in measured + fills):
        raise ValueError("unknown workload or workload source_split differs from requested split")
    measured = measured[:limit] if limit is not None else measured
    previous, initial, output = [], [], []
    for phase, source_rows in (("initial_fill", fills), ("measured", measured)):
        for index, event in enumerate(source_rows, 1):
            q = questions[event["question_id"]]
            if q["effective_question_status"] == "rejected" or q["effective_answer_status"] == "rejected":
                raise ValueError(f"workload contains rejected item {q['question_id']}; cannot silently alter sequence")
            if purpose != "functional" and not question_eligible(q):
                raise ValueError(f"{q['question_id']}: formal workload requires explicit question AND answer/behavior review")
            identifier = f"{workload_id}_{phase}_{index:03d}"
            relevant = []
            for earlier in previous:
                judgment = pairs.get((earlier["question_id"], q["question_id"]))
                label = pair_label(judgment)
                if purpose != "functional" and label == "unverified":
                    raise ValueError(f"formal cache safety needs pair review: {earlier['question_id']} -> {q['question_id']}")
                relevant.append({"reference_scenario_id": earlier["scenario_id"], "expected_answer_reuse": label,
                                 "review_status": "approved" if label != "unverified" and purpose != "functional" else "needs_review"})
            preferred = next((r for r in reversed(previous) if r["question_id"] == event.get("reference_question_id")), None)
            selected_judgment = next((r for r in relevant if preferred and r["reference_scenario_id"] == preferred["scenario_id"]), None)
            row = {"scenario_id": identifier, "query_id": identifier, "query": q["question"], "question": q["question"],
                   "question_id": q["question_id"], "intent_key": q["intent_id"] or "unresolved:" + fingerprint(q["question"]),
                   "intent_group": q["intent_id"] or "unresolved:" + fingerprint(q["question"]),
                   "category": q["primary_type"], "family_id": q["family_id"], "split": split,
                   "source_split": split, "query_order": index, "phase": phase, "workload_id": workload_id,
                   "reference_scenario_id": preferred["scenario_id"] if preferred else None,
                   "expected_answer_reuse": selected_judgment["expected_answer_reuse"] if selected_judgment else "unverified",
                   "reuse_judgments": [r for r in relevant if not preferred or r["reference_scenario_id"] != preferred["scenario_id"]],
                   "review_status": "needs_review" if purpose == "functional" else "approved",
                   "research_metadata": metadata(q), "workload_event": event,
                   "cache_state_policy": event.get("cache_state_policy", "empty_then_explicit_initial_fill")}
            (initial if phase == "initial_fill" else output).append(row)
            previous.append(row)
    # Current threshold/INT loaders require references to earlier rows in the
    # SAME file. Warm-fill is deliberately separate, never silently counted.
    return initial, output


def build(args, root, data, output):
    purpose, split, target = args.purpose, args.split, args.target
    if purpose == "calibration" and split != "calibration" or purpose == "evaluation" and split != "evaluation":
        raise ValueError("calibration/evaluation purpose must use its own split")
    if args.limit is not None and (args.limit < 1 or purpose != "functional"):
        raise ValueError("--limit is positive and functional-only; formally freeze explicit IDs instead")
    if purpose != "functional" and not args.view:
        raise ValueError("formal exports require a human-merged --view")
    questions, pairs, turns, view_hash = get_views(data, locate(root, args.view) if args.view else None)
    settings_hash = sha(locate(root, args.settings)) if args.settings else None
    parameter_source = None
    calibration_run_receipt = None
    if purpose != "functional" and settings_hash is None:
        raise ValueError("formal export requires --settings to pin model/prompt/decoder parameters")
    if purpose == "evaluation":
        if not args.calibration_export:
            raise ValueError("evaluation requires --calibration-export; no evaluation-driven parameter selection")
        prior_dir = locate(root, args.calibration_export)
        prior = verify_bundle(prior_dir, "adapter_manifest.json")
        completed = read_json(prior_dir / "completed.json")
        if (prior["purpose"] != "calibration" or prior["split"] != "calibration" or prior["target"] != target
                or prior["dataset_identity"] != data["identity"] or prior["settings_sha256"] != settings_hash
                or completed.get("status") != "completed"):
            raise ValueError("calibration export/settings/dataset/target identity mismatch")
        if prior["review_view_sha256"] != view_hash:
            raise ValueError("review view changed since calibration selection; freeze and regenerate both exports")
        if prior["tool_identity"]["code_sha256"] != tool_identity()["code_sha256"]:
            raise ValueError("adapter code changed since calibration export; review before comparing")
        parameter_source = sha(prior_dir / "adapter_manifest.json")
        if not args.calibration_run:
            raise ValueError("evaluation needs --calibration-run; an export is not a measured calibration result")
        calibration_run = locate(root, args.calibration_run)
        run_receipt = read_json(calibration_run / "completed.json")
        run_meta = read_json(calibration_run / "run.json")
        input_hashes = {v for k, v in prior["files_sha256"].items()
                        if k in {"scenarios.jsonl", "calibration.jsonl", "measured.jsonl"}}
        if (run_receipt.get("status") != "completed" or run_receipt.get("source_verification") != "passed"
                or run_receipt.get("provisional", True) or run_meta.get("provisional", True)
                or run_meta.get("phase") in {"all", "evaluate"} or run_meta.get("split") == "evaluation"
                or not input_hashes & leaf_strings({"run": run_meta, "completion": run_receipt})):
            raise ValueError("calibration run must be non-provisional, source-verified and use the matching Calibration adapter inputs")
        calibration_run_receipt = {name: sha(calibration_run / name) for name in ("run.json", "completed.json")}
    selected = select_questions(questions, split, set(filter(None, (args.ids or "").split(","))),
                                set(filter(None, (args.types or "").split(","))),
                                None if target == "ti" or args.workload_id else args.limit)
    capability = {"direct_current_cli": True, "limitation": None}
    native, scorer, initial = [], [], []
    if target in {"sc", "sc_threshold", "int"}:
        if not args.workload_id:
            if purpose != "functional" or not args.ids or args.types:
                raise ValueError("use --workload-id, or functional-only explicit ordered --ids (no type filter)")
            ordered = args.ids.split(",")
            if len(set(ordered)) != len(ordered):
                raise ValueError("explicit smoke IDs must be unique; select an R question for repeated text")
            events = []
            seen = set()
            for index, key in enumerate(ordered, 1):
                q = questions[key]
                reference = q.get("reuse_reference_question_id")
                events.append({"workload_id": "explicit_smoke_sequence", "question_id": key,
                               "question_text": q["question"], "query_order": index, "source_split": split,
                               "reference_question_id": reference if reference in seen else None,
                               "cache_state_policy": "empty_at_workload_start", "review_status": "needs_review",
                               "sequence_kind": "explicit_user_order_not_formal_workload"})
                seen.add(key)
            sequence_data = {**data, "workloads": events, "fills": []}
            workload_id = "explicit_smoke_sequence"
        else:
            if args.ids or args.types:
                raise ValueError("cannot filter a workload by IDs/types; use a functional prefix --limit")
            sequence_data, workload_id = data, args.workload_id
        initial, native = sequence(sequence_data, questions, pairs, workload_id, split, args.limit, purpose)
        if initial:
            capability = {"direct_current_cli": False, "limitation": "requires phase-aware runner: initial fill separate from measured denominator"}
        if target == "int":
            capability = {"direct_current_cli": False, "limitation": "INT loader hardcodes functional split; research split must not be relabeled"}
        if target == "sc_threshold" and split == "historical":
            capability = {"direct_current_cli": False, "limitation": "threshold CLI does not accept historical split"}
        write_rows(output / "initial_fill.jsonl", initial)
        write_rows(output / "measured.jsonl", native)
        if capability["direct_current_cli"]:
            write_rows(output / "scenarios.jsonl", native)
        for row in native:
            q = questions[row["question_id"]]
            scorer.append({"event_id": row["scenario_id"], "question": q, "pair_reviews":
                           [p for (a, b), p in pairs.items() if b == q["question_id"]]})
    elif target == "ti":
        selected_ids = {q["question_id"] for q in selected}
        eligible = [t for t in turns.values() if t["split"] == split
                    and set(t["source_question_ids"]) & selected_ids]
        conversation_ids = list(dict.fromkeys(t["conversation_id"] for t in eligible))
        if args.limit is not None:
            conversation_ids = conversation_ids[:args.limit]
        eligible = [t for t in turns.values() if t["conversation_id"] in conversation_ids]
        if any(t["effective_turn_status"] == "rejected" for t in eligible):
            raise ValueError("selected conversation contains a rejected turn; review the full conversation")
        if purpose != "functional" and any(t["effective_turn_status"] != "verified" for t in eligible):
            raise ValueError("formal TI requires all selected conversation turns explicitly reviewed")
        for t in eligible:
            if t["effective_topic_label"] == "initial":
                continue
            native.append({"scenario_id": t["turn_id"], "conversation_id": t["conversation_id"], "turn_id": t["turn_id"],
                           "query": t["question"], "history": t["history"], "split": split, "family_id": t["family_id"],
                           "category": "human_review_view" if purpose != "functional" else "candidate_script",
                           "gold_shift": int(t["effective_topic_label"] == "shift"),
                           "required_history_ids": [v for key in t["effective_required_history"] for v in (key, key + "_A")],
                           "review_status": "needs_review" if purpose == "functional" else "approved",
                           "research_metadata": t})
        if split == "historical":
            raise ValueError("no historical conversations in source dataset")
        if {r["gold_shift"] for r in native} != {0, 1}:
            raise ValueError("current TI loader requires both same-topic and shift; select another/full conversation set")
        write_rows(output / f"{split}.jsonl", native)
    else:
        for q in selected:
            if q["effective_question_status"] == "rejected" or q["effective_answer_status"] == "rejected":
                raise ValueError(f"selection contains rejected item {q['question_id']}")
            if purpose != "functional" and not question_eligible(q):
                raise ValueError(f"{q['question_id']}: explicit question/answer-or-behavior human review required")
            # No reference answer or expected behavior enters a QA/SD request.
            native.append({"question_id": q["question_id"], "scenario_id": q["question_id"],
                           "question": q["question"], "query": q["question"], "category": q["primary_type"],
                           "family_id": q["family_id"], "split": split,
                           "review_status": "needs_review" if purpose == "functional" else "approved",
                           "research_metadata": metadata(q)})
            scorer.append(q)
        write_rows(output / "scenarios.jsonl", native)
        if target == "qa":
            capability = {"direct_current_cli": False, "limitation": "eight-architecture dataset/worker/scorer use fixed FAQ60; separate evaluation ID from optional gold FAQ ID"}
    if not native:
        raise ValueError("empty adapter output")
    write_rows(output / "scorer_only.jsonl", scorer)
    plan = {"schema": "research_adapter_v1", "dataset_identity": data["identity"], "dataset_hashes": data["hashes"],
            "tool_identity": tool_identity(),
            "review_view_sha256": view_hash, "target": target, "split": split, "purpose": purpose,
            "provisional": purpose == "functional", "formal_results_produced": False,
            "settings_sha256": settings_hash, "calibration_export_manifest_sha256": parameter_source,
            "calibration_run_receipt_sha256": calibration_run_receipt,
            "count": len(native), "initial_fill_count": len(initial), "limit": args.limit,
            "family_ids": sorted({r["family_id"] for r in native}),
            "query_ids": [r["scenario_id"] for r in native], "capability": capability,
            "metrics_policy": {"qa_generation_metrics_for_A1_A2": "not_applicable",
                               "multiturn_QA_for_singleturn_architectures": "unavailable",
                               "warm_initial_fill": "exclude from measured denominator; record cost separately",
                               "quality_and_speed": "score separately; no quality-preserving speedup claim without review"},
            "generation_comparison_contract": "same Target/tokenizer/input IDs/Top5/prompt/precision/generation settings; use SD0 paired token checks",
            "files_sha256": {p.name: sha(p) for p in output.glob("*.jsonl")}}
    if purpose == "evaluation":
        prior = read_json(locate(root, args.calibration_export) / "adapter_manifest.json")
        if set(prior["family_ids"]) & set(plan["family_ids"]) or set(prior["query_ids"]) & set(plan["query_ids"]):
            raise ValueError("calibration/evaluation family or query overlap")
    write_json(output / "adapter_manifest.json", plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from pathlib import Path
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", choices=("qa", "sc", "sc_threshold", "sd", "prompt_lookup", "ti", "int"), required=True)
    parser.add_argument("--split", choices=("historical", "calibration", "evaluation"), required=True)
    parser.add_argument("--purpose", choices=("functional", "calibration", "evaluation"), default="functional")
    parser.add_argument("--view")
    parser.add_argument("--ids", help="comma-separated original Question IDs")
    parser.add_argument("--types", help="comma-separated primary type codes")
    parser.add_argument("--limit", type=int, help="functional prefix; TI uses full conversation count")
    parser.add_argument("--workload-id")
    parser.add_argument("--settings", help="chosen local config file, hashed only; no model loading")
    parser.add_argument("--calibration-export", help="completed matching Calibration adapter export")
    parser.add_argument("--calibration-run", help="non-provisional, source-verified measured Calibration run")
    args = parser.parse_args()
    root = args.project_root.resolve()
    data = load_dataset(root)
    output = new_output(root, args.output_dir)
    try:
        build(args, root, data, output)
        if load_dataset(root)["identity"] != data["identity"]:
            raise ValueError("dataset changed during adapter export")
        write_json(output / "completed.json", {"status": "completed", "kind": "data_adapter_only_no_model_run"})
    except BaseException as error:
        write_json(output / "failed.json", {"error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
