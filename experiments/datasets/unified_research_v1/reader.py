"""Portable, model-free research-data reader and explicit future field mappings.

Existing benchmark CLIs are intentionally unchanged. These mappings do not
disable their fixed-60, review, context or split gates.
"""
from pathlib import Path

from .validation import read_rows


def load_questions(project_root, split, *, include_unreviewed=False):
    if split not in {"historical", "calibration", "evaluation"}:
        raise ValueError("unknown research split")
    directory = Path(project_root).resolve() / "experiments/datasets/unified_research_v1"
    rows = read_rows(directory / f"questions_{split}.jsonl")
    if not include_unreviewed and any(r["review_status"] != "reviewed" for r in rows):
        raise ValueError("research candidates need human review; not a frozen Gold set")
    return rows


def as_qa_request(row):
    # question_id identifies the evaluation event, NOT the correct FAQ.
    # Reference answers are scorer-side only, never put into a model request.
    return {"question_id": row["question_id"], "question": row["question"],
            "research_split": row["split"], "family_id": row["family_id"]}


def load_conversation_turns(project_root, split, *, include_unreviewed=False):
    if split not in {"calibration", "evaluation"}:
        raise ValueError("unknown conversation split")
    directory = Path(project_root).resolve() / "experiments/datasets/unified_research_v1"
    rows = [r for r in read_rows(directory / "conversations.jsonl") if r["split"] == split]
    if not include_unreviewed and any(r["review_status"] != "reviewed" for r in rows):
        raise ValueError("conversation labels need human review")
    return rows


def load_cache_workload(project_root, workload_id, *, include_unreviewed=False):
    directory = Path(project_root).resolve() / "experiments/datasets/unified_research_v1"
    measured = [r for r in read_rows(directory / "cache_workloads.jsonl") if r["workload_id"] == workload_id]
    initial = [r for r in read_rows(directory / "cache_initial_fill.jsonl") if r["workload_id"] == workload_id]
    if not measured:
        raise ValueError(f"unknown workload: {workload_id}")
    if not include_unreviewed and any(r["review_status"] != "reviewed" or
            r.get("expected_reuse_review_status", "reviewed") != "reviewed" for r in initial + measured):
        raise ValueError("cache workload and reuse labels need human review")
    return {"initial_fill": initial, "measured": measured,
            "source_split": measured[0]["source_split"], "cache_reset_required": True}


def as_sd_scenario(row):
    # Ambiguous questions have no fabricated FAQ gold; every decoder still
    # receives exactly the same question/prompt in a controlled comparison.
    return {"scenario_id": row["question_id"], "query": row["question"],
            "category": row["primary_type"], "family_id": row["family_id"],
            "research_split": row["split"],
            "review_status": "approved" if row["review_status"] == "reviewed" else "needs_review"}


def as_ti_case(turn):
    if turn["same_topic_or_shift"] == "initial":
        raise ValueError("initial turns have no previous topic; exclude from shift classification")
    required = []
    for identifier in turn["required_history_turn_ids"]:
        required.extend([identifier, identifier + "_A"])
    return {"scenario_id": turn["turn_id"], "query": turn["question"],
            "history": turn["history"], "gold_shift": int(turn["same_topic_or_shift"] == "shift"),
            "required_history_ids": required, "family_id": turn["family_id"],
            "split": turn["split"], "category": "scripted_multiturn_candidate",
            "review_status": "approved" if turn["review_status"] == "reviewed" else "needs_review"}
