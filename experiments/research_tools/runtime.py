"""Verified research-export contracts shared by the independent QA/INT runners."""
from __future__ import annotations

from .common import (load_dataset, read_rows, sha, tool_identity,
                     verify_bundle)
from .adapters import get_views, metadata, pair_label, question_eligible


def load_input(root, directory, target, config_path, allow_unreviewed, view=None):
    manifest = verify_bundle(directory, "adapter_manifest.json")
    data = load_dataset(root)
    if (manifest["target"] != target or manifest["dataset_identity"] != data["identity"]
            or manifest["dataset_hashes"] != data["hashes"]):
        raise ValueError("adapter target/dataset/source identity mismatch")
    purpose, split = manifest["purpose"], manifest["split"]
    if purpose not in {"functional", "calibration", "evaluation"} or split not in {
            "historical", "calibration", "evaluation"}:
        raise ValueError("invalid purpose/split")
    formal = purpose != "functional"
    if manifest.get("settings_sha256") is not None and manifest["settings_sha256"] != sha(config_path):
        raise ValueError("settings file differs from the exported experiment settings")
    if manifest["provisional"] != (not formal) or (formal and purpose != split):
        raise ValueError("Calibration/Evaluation purpose and split must agree")
    if not formal and not allow_unreviewed:
        raise ValueError("functional data requires --allow-unreviewed; never a formal result")
    if formal:
        if view is None or manifest["settings_sha256"] != sha(config_path):
            raise ValueError("formal run requires matching --view and frozen config bytes")
        if manifest["tool_identity"]["code_sha256"] != tool_identity()["code_sha256"]:
            raise ValueError("formal adapter code changed; regenerate the export")
        if split == "evaluation" and (not manifest.get("calibration_export_manifest_sha256")
                                       or not manifest.get("calibration_run_receipt_sha256")):
            raise ValueError("Evaluation requires frozen Calibration selection receipts")
    questions, pairs, _, view_hash = get_views(data, view)
    if manifest["review_view_sha256"] != view_hash:
        raise ValueError("supply the same review --view as the adapter export")
    phases = ({"measured": read_rows(directory / "scenarios.jsonl")} if target == "qa" else
              {name: read_rows(directory / (name + ".jsonl")) for name in ("initial_fill", "measured")})
    seen, previous = set(), {}
    for phase, rows in phases.items():
        for order, row in enumerate(rows, 1):
            key = row["scenario_id"]
            q = questions[row["question_id"]]
            if target == "qa" and key != q["question_id"]:
                raise ValueError("QA scenario ID must remain the original Question ID")
            if key in seen or q["split"] != split or row["split"] != split:
                raise ValueError(f"{key}: duplicate ID or mixed split")
            if row["question"] != q["question"] or row.get("query") != q["question"]:
                raise ValueError(f"{key}: query differs from immutable source")
            if row["research_metadata"] != metadata(q) or row["family_id"] != q["family_id"]:
                raise ValueError(f"{key}: provenance/review/exposure metadata changed")
            if q["effective_question_status"] == "rejected" or q["effective_answer_status"] == "rejected":
                raise ValueError(f"{key}: rejected source")
            if formal and (not question_eligible(q) or row["review_status"] != "approved"):
                raise ValueError(f"{key}: human question and answer/behavior review required")
            if not formal and row["review_status"] != "needs_review":
                raise ValueError("functional export cannot upgrade native review status")
            if target == "int":
                if row["phase"] != phase or row["source_split"] != split or row["query_order"] != order:
                    raise ValueError(f"{key}: phase/source split/query order mismatch")
                if any(row.get(k) for k in ("history", "messages", "context", "session_id")):
                    raise ValueError("INT remains single-turn; conversation history is not silently dropped")
                workload = row["workload_id"]
                if (row["workload_event"]["question_id"] != q["question_id"]
                        or row["workload_event"]["source_split"] != split):
                    raise ValueError(f"{key}: workload event/source question identity mismatch")
                if workload == "explicit_smoke_sequence":
                    if formal:
                        raise ValueError("explicit smoke sequence is not a formal workload")
                else:
                    source = data["fills"] if phase == "initial_fill" else data["workloads"]
                    events = [e for e in source if e["workload_id"] == workload]
                    if order > len(events) or row["workload_event"] != events[order - 1]:
                        raise ValueError(f"{key}: source workload order/event changed")
                judgments = row.get("reuse_judgments", []) + ([{
                    "reference_scenario_id": row["reference_scenario_id"],
                    "expected_answer_reuse": row["expected_answer_reuse"],
                    "review_status": row["review_status"]}] if row.get("reference_scenario_id") else [])
                refs = set()
                for judgment in judgments:
                    ref = judgment["reference_scenario_id"]
                    if ref not in previous or ref in refs:
                        raise ValueError(f"{key}: invalid earlier cache reference {ref}")
                    refs.add(ref)
                    expected = pair_label(pairs.get((previous[ref], q["question_id"])))
                    if formal and (expected == "unverified" or judgment["expected_answer_reuse"] != expected
                                   or judgment["review_status"] != "approved"):
                        raise ValueError(f"{key}: missing/mismatched human cache-pair review")
                if formal and refs != set(previous):
                    raise ValueError(f"{key}: formal safety review must cover all possible earlier origins")
                if not row.get("reference_scenario_id") and row["expected_answer_reuse"] != "unverified":
                    raise ValueError(f"{key}: reuse label has no reference")
            previous[key] = q["question_id"]
            seen.add(key)
    if (not phases["measured"] or len(phases["measured"]) != manifest["count"]
            or len(phases.get("initial_fill", [])) != manifest["initial_fill_count"]):
        raise ValueError("empty input or adapter phase counts differ")
    if ([r["scenario_id"] for r in phases["measured"]] != manifest["query_ids"]
            or sorted({r["family_id"] for r in phases["measured"]}) != manifest["family_ids"]):
        raise ValueError("adapter question order/family inventory changed")
    if target == "int" and len({r["workload_id"] for rows in phases.values() for r in rows}) != 1:
        raise ValueError("a run must preserve one complete cache workload identity")
    receipt = {"adapter_manifest_sha256": sha(directory / "adapter_manifest.json"),
               "completed_sha256": sha(directory / "completed.json"), "dataset_identity": data["identity"],
               "input_files_sha256": manifest["files_sha256"], "review_view_sha256": view_hash,
               "config_sha256": sha(config_path), "tool_identity": tool_identity()}
    return manifest, phases, questions, receipt


def verify_input(root, directory, target, config_path, allow_unreviewed, view, expected):
    *_, actual = load_input(root, directory, target, config_path, allow_unreviewed, view)
    if actual != expected:
        raise ValueError("research input/config/review/tool identity changed during run")


def model_request(row):
    """Positive allowlist: no labels, scorer fields, provenance or IDs in the query."""
    return {"question_id": row["question_id"], "question": row["question"]}


def append_row(path, row):
    import json
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
