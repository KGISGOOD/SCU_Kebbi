"""SC1-3 sequence and pair-label validation; labels never enter the cache."""
from __future__ import annotations

import json
from .cache import normalize_question, fingerprint


def load_sequence(path, split, allow_unreviewed):
    rows, seen = [], set()
    for order, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL line {order}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {order}: expected object")
        for name in ("scenario_id", "question", "intent_group", "family_id"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"line {order}: invalid {name}")
        if row["scenario_id"] in seen or type(row.get("query_order")) is not int or row["query_order"] != order:
            raise ValueError("scenario IDs must be unique; file order must be contiguous query_order starting at 1")
        if row.get("split") != split:
            raise ValueError(f"{row['scenario_id']}: split must be {split}")
        if row.get("review_status") not in {"approved", "needs_review"}:
            raise ValueError("invalid review_status")
        if row["review_status"] != "approved" and not allow_unreviewed:
            raise ValueError("sequence needs human review; --allow-unreviewed is diagnostic only")
        if any(row.get(k) for k in ("history", "messages", "context", "session_id")):
            raise ValueError("SC1-3 supports independent single-turn questions only")
        row.setdefault("reference_scenario_id", None)
        reference = row["reference_scenario_id"]
        label = row.get("expected_answer_reuse")
        if label not in {"allowed", "forbidden", "unverified"}:
            raise ValueError("expected_answer_reuse must be allowed, forbidden or unverified")
        if reference is not None and reference not in seen:
            raise ValueError("reference_scenario_id must point to an earlier row")
        if reference is None and label != "unverified":
            raise ValueError("a reuse label requires a specific reference question")
        judgments = row.get("reuse_judgments", [])
        if not isinstance(judgments, list):
            raise ValueError("reuse_judgments must be a list")
        references = {reference} if reference is not None else set()
        for judgment in judgments:
            if not isinstance(judgment, dict):
                raise ValueError("invalid pair judgment")
            ref = judgment.get("reference_scenario_id")
            if (ref not in seen or ref in references or judgment.get("expected_answer_reuse") not in
                    {"allowed", "forbidden", "unverified"} or judgment.get("review_status") not in {"approved", "needs_review"}):
                raise ValueError("invalid or duplicate extra reference judgment")
            if judgment["review_status"] != "approved" and not allow_unreviewed:
                raise ValueError("extra pair judgments need review")
            references.add(ref)
        normalize_question(row["question"])
        seen.add(row["scenario_id"])
        rows.append(row)
    if not rows:
        raise ValueError("empty sequence")
    return rows


def manifest(rows):
    return {"scenario_ids": [r["scenario_id"] for r in rows],
            "families": sorted({r["family_id"] for r in rows}),
            "question_hashes": sorted({fingerprint(normalize_question(r["question"])) for r in rows})}


def assert_independent(calibration_manifest, rows):
    other = manifest(rows)
    for name in ("scenario_ids", "families", "question_hashes"):
        if set(calibration_manifest[name]) & set(other[name]):
            raise ValueError(f"calibration/evaluation overlap: {name}; semantic overlap still needs human review")


def judge_reuse(case, origin):
    """Assess the ACTUAL stored origin, never assume that all same-intent answers are safe."""
    if origin is None:
        return {"status": "not_applicable_miss", "label": None, "false_reuse": None}
    labels = [{"reference_scenario_id": case["reference_scenario_id"],
               "expected_answer_reuse": case["expected_answer_reuse"], "review_status": case["review_status"]}]
    labels += case.get("reuse_judgments", [])
    label = next((j for j in labels if j["reference_scenario_id"] == origin["scenario_id"]), None)
    if (label is None or label["review_status"] != "approved" or case["review_status"] != "approved" or origin["review_status"] != "approved"
            or label["expected_answer_reuse"] == "unverified"):
        return {"status": "unverified", "label": label, "false_reuse": None}
    forbidden = label["expected_answer_reuse"] == "forbidden"
    return {"status": "confirmed_pair_incompatibility" if forbidden else "pair_compatible_answer_unverified",
            "label": label, "false_reuse": forbidden}
