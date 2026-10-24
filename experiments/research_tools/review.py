"""Export Excel review sheets and merge explicit human decisions into NEW views."""
from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path

from .common import ROOT, fingerprint, load_dataset, locate, new_output, read_json, sha, tool_identity, write_json, write_rows

STATES = {"verified", "rejected", "needs_review"}
Q_EDIT = {"question_status", "answer_status", "answer_action", "answer_override", "reviewer", "reviewed_at", "evidence", "notes"}
P_EDIT = {"reuse_status", "reuse_decision", "reviewer", "reviewed_at", "evidence", "notes"}
T_EDIT = {"turn_status", "turn_action", "topic_label_override", "required_history_override", "reviewer", "reviewed_at", "evidence", "notes"}


def cell(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value


def catalogue(data):
    by_id = {q["question_id"]: q for q in data["questions"]}
    pairs = {}

    def pair(origin, current, reason):
        if origin not in by_id or current not in by_id:
            return
        key = "PAIR_" + fingerprint([origin, current])[:24]
        if key not in pairs:
            pairs[key] = {"pair_id": key, "origin_question_id": origin, "current_question_id": current,
                          "candidate_expected_reuse": by_id[current]["expected_reuse"]
                          if by_id[current].get("reuse_reference_question_id") == origin else "unknown",
                          "reasons": []}
        if reason not in pairs[key]["reasons"]:
            pairs[key]["reasons"].append(reason)

    for q in data["questions"]:
        for other in data["questions"]:
            if q["question_id"] != other["question_id"] and q["family_id"] == other["family_id"] and q["split"] == other["split"]:
                pair(other["question_id"], q["question_id"], "same_family_same_split_candidate_not_safe_by_default")
    for relation in data["relations"]:
        pair(relation["to_question_id"], relation["from_question_id"], relation["relation_type"])
    for event in data["workloads"]:
        for key in ("reference_question_id", "safety_reference_question_id"):
            if event.get(key):
                pair(event[key], event["question_id"], event["workload_id"] + ":" + key)
    # Any earlier stored entry can be the actual semantic origin, including
    # a different family or the SAME question ID revisited later.
    for workload_id in dict.fromkeys(r["workload_id"] for r in data["workloads"]):
        earlier_ids = set()
        events = [r for r in data["fills"] if r["workload_id"] == workload_id]
        events += [r for r in data["workloads"] if r["workload_id"] == workload_id]
        for event in events:
            for earlier in sorted(earlier_ids):
                pair(earlier, event["question_id"], workload_id + ":possible_actual_origin")
            earlier_ids.add(event["question_id"])
    # r4 INT can prefill all selected Q1s before measured requests. Those entries
    # include later groups, so cold-order pairs alone cannot satisfy its Guard.
    if data.get("dataset_name") == "experiment_suites_v1_r4":
        from .suites import execution
        filled = execution(data, "prefilled")
        for workload_id in dict.fromkeys(r["workload_id"] for r in filled["workloads"]
                                        if r["workload_id"].startswith("INT-")):
            earlier_ids = set()
            events = [r for r in filled["fills"] if r["workload_id"] == workload_id]
            events += [r for r in filled["workloads"] if r["workload_id"] == workload_id]
            for event in events:
                for earlier in sorted(earlier_ids):
                    pair(earlier, event["question_id"], workload_id + ":prefilled_possible_actual_origin")
                earlier_ids.add(event["question_id"])
    if data.get("dataset_name") == "experiment_suites_v1_r4":
        # Tag only workload-reachable pairs. Extra historical catalogue entries
        # remain reviewable but are NOT mandatory for these fixed experiments.
        from .suites import execution
        for state in ("cold", "prefilled"):
            scoped = execution(data, state)
            for wid in dict.fromkeys(r["workload_id"] for r in scoped["workloads"]):
                if state == "prefilled" and not wid.startswith("INT-"):
                    continue
                prior = set()
                events = [r for r in scoped["fills"] if r["workload_id"] == wid]
                events += [r for r in scoped["workloads"] if r["workload_id"] == wid]
                for event in events:
                    for origin in prior:
                        identifier = "PAIR_" + fingerprint([origin, event["question_id"]])[:24]
                        scopes = pairs[identifier].setdefault("required_for", [])
                        scope = wid + ":" + state
                        if scope not in scopes:
                            scopes.append(scope)
                    prior.add(event["question_id"])
        for p in pairs.values():
            p["required_for"] = sorted(p.get("required_for", []))
            p.update(candidate_pair_review(by_id[p["origin_question_id"]], by_id[p["current_question_id"]]))
    return by_id, pairs


def candidate_pair_review(a, b):
    """Five evidence classes, never a human reusable/not-reusable decision."""
    same_basis = (bool(a.get("source_hash")) and bool(a.get("source_document_ids"))
                  and a.get("source_hash") == b.get("source_hash")
                  and a.get("source_document_ids") == b.get("source_document_ids")
                  and a.get("reference_answer") == b.get("reference_answer")
                  and bool(a.get("reference_answer")))
    intents = {a.get("intent_id"), b.get("intent_id")}
    overlaps = ({"P18", "P19"}, {"P21", "S13"}, {"P24", "S11"})
    weak = any(q.get("source_audit", {}).get("insufficient_evidence", True) for q in (a, b))
    if same_basis and a["question"] == b["question"]:
        cls, reason = "1_same_question_basis", "問題與候選答案/來源Hash相同；仍須審個人化、年度及原答案真實性。"
    elif any(intents == group for group in overlaps):
        cls, reason = "4_partial_overlap", "已知完整候選答案可能同時涵蓋兩個資訊需求；須逐答案核對。"
    elif same_basis and a.get("intent_id") == b.get("intent_id"):
        cls, reason = "2_paraphrase_candidate", "來源意圖與候選答案一致的改寫候選；未證實整份答案可安全重用。"
    elif weak:
        cls, reason = "5_evidence_or_conditions_missing", "來源不足、年度/資格或逐題答案尚未核定，無法安全判斷。"
    else:
        cls, reason = "3_different_intent_candidate", "來源意圖不同的候選；不同ID不證明完整答案沒有交集，仍需人工核對。"
    return {"candidate_class": cls, "candidate_reason": reason,
            "candidate_rule_id": cls + (":source_unverified" if weak else ":partial_source_only"),
            "answer_evidence_sha256": fingerprint({"origin": a.get("reference_answer"),
                                                   "current": b.get("reference_answer"),
                                                   "origin_source": a.get("source_hash"),
                                                   "current_source": b.get("source_hash")}),
            "candidate_is_human_gold": False}


def question_sheet(q, data):
    return {"question_id": q["question_id"], "family_id": q["family_id"], "split": q["split"],
            "question": q["question"], "primary_type": q["primary_type"],
            "source_documents": cell({s: data["registry"]["sources"][s] for s in q["source_document_ids"]}),
            "candidate_answer": q["reference_answer"] or "", "candidate_answer_status": q["reference_answer_status"],
            "related_question_ids": cell(q["related_question_ids"]),
            "train_exposure": cell(q["training_overlap_details"]), "expected_behavior": q["expected_behavior"],
            "record_sha256": fingerprint(q), "question_status": "needs_review", "answer_status": "needs_review",
            "answer_action": "", "answer_override": "", "reviewer": "", "reviewed_at": "", "evidence": "", "notes": ""}


def pair_sheet(p, by_id):
    a, b = by_id[p["origin_question_id"]], by_id[p["current_question_id"]]
    return {**{k: cell(v) for k, v in p.items()}, "origin_question": a["question"], "current_question": b["question"],
            "origin_candidate_answer": a["reference_answer"] or "", "current_candidate_answer": b["reference_answer"] or "",
            "origin_family": a["family_id"], "current_family": b["family_id"],
            "origin_split": a["split"], "current_split": b["split"],
            "origin_sources": cell(a["source_document_ids"]), "current_sources": cell(b["source_document_ids"]),
            "train_exposure": cell({"origin": a["training_overlap_details"], "current": b["training_overlap_details"]}),
            "record_sha256": fingerprint([p, a, b]), "reuse_status": "needs_review", "reuse_decision": "unknown",
            "reviewer": "", "reviewed_at": "", "evidence": "", "notes": ""}


def turn_sheet(t):
    return {"turn_id": t["turn_id"], "conversation_id": t["conversation_id"], "family_id": t["family_id"],
            "split": t["split"], "question": t["question"], "history": cell(t["history"]),
            "candidate_topic_label": t["same_topic_or_shift"],
            "candidate_required_history": cell(t["required_history_turn_ids"]),
            "source_question_ids": cell(t["source_question_ids"]), "record_sha256": fingerprint(t),
            "turn_status": "needs_review", "turn_action": "", "topic_label_override": "", "required_history_override": "",
            "reviewer": "", "reviewed_at": "", "evidence": "", "notes": "",
            "candidate_relation_reason": turn_reason(t)}


def turn_reason(t):
    # Compare against preserved history queries; topic identity comes from the
    # source annotation and is not inferred from model performance.
    label = t["same_topic_or_shift"]
    reason = {"initial": "首輪，無前輪主題可比較。", "same_topic": "候選定義為同一對話任務/對象的延續。",
              "shift": "候選定義為相對上一輪切換資訊任務；回到更早主題也屬相對上一輪切換。"}[label]
    conversation = t.get("suite_conversation_id", "")
    if conversation in {"TI-10", "TI-19"}:
        reason += "含課程/年度指代，需核對是否應澄清及必要歷史，不能只按字面主題核准。"
    if conversation in {"TI-12", "TI-20"}:
        reason += "開始、找老師、分組與選題仍採專題規劃同主題粒度，待人工確認。"
    return reason + "必要歷史候選：" + cell(t["required_history_turn_ids"]) + "；未經人工認證。"


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def overview(root, data, output, view=None):
    """Read-only source projection, not a second dataset or a review decision.

    Rebuild only to a NEW CSV; manual review remains in separate merged views.
    An absent human view is blank, never inferred verified from source status.
    """
    allowed = (root / "experiments/datasets").resolve()
    if (not output.is_relative_to(allowed) or output.parent != allowed
            or output.suffix.lower() != ".csv" or output.exists()):
        raise ValueError("overview must be a NEW CSV directly under experiments/datasets")
    descriptions = {"A": "直接", "B": "改寫", "C": "近意圖", "D": "多資訊",
                    "U": "資訊不足", "O": "領域外", "T": "時效", "R": "重複",
                    "E": "條件／否定", "F": "數字／計算", "G": "模糊",
                    "H": "口語／錯字", "I": "來源衝突", "J": "干擾／注入"}
    order = {"historical": 0, "calibration": 1, "evaluation": 2}
    questions = data["questions"]
    if (len(questions) != 230 or len({q["question_id"] for q in questions}) != 230
            or {s: sum(q["split"] == s for q in questions) for s in order}
            != {"historical": 60, "calibration": 60, "evaluation": 110}):
        raise ValueError("overview requires exactly the original 60/60/110 unique main questions")
    from .adapters import get_views
    human, _, _, view_hash = get_views(data, view)
    rows = []
    for q in sorted(questions, key=lambda q: (order[q["split"]], q["question_id"])):
        sheet = question_sheet(q, data)
        if q["primary_type"] not in descriptions:
            raise ValueError(f"undefined category: {q['primary_type']}; do not guess a label")
        h = human[q["question_id"]].get("human_review", {}) if view else {}
        rows.append({"question_id": q["question_id"], "split": q["split"],
                     "category": q["primary_type"], "category_description": descriptions[q["primary_type"]],
                     "family_id": q["family_id"], "question": sheet["question"],
                     "candidate_answer": sheet["candidate_answer"], "expected_behavior": sheet["expected_behavior"],
                     "source_document_ids": cell(q["source_document_ids"]),
                     "source_reference": sheet["source_documents"],
                     "training_overlap_status": q["training_overlap_status"],
                     "training_overlap_details": sheet["train_exposure"],
                     "question_review_status": q["review_status"],
                     "answer_review_status": q["reference_answer_status"],
                     "human_question_review_status": h.get("question_status", ""),
                     "human_answer_review_status": h.get("answer_status", ""),
                     "human_answer_action": h.get("answer_action", ""),
                     "human_answer_override": h.get("answer_override", ""),
                     "human_reviewer": h.get("reviewer", ""), "human_reviewed_at": h.get("reviewed_at", ""),
                     "notes": cell({"usage": "derived_reading_view_not_gold_or_benchmark_input",
                                    "source_support_level": q.get("source_support_level"),
                                    "provenance": q.get("provenance"), "human_notes": h.get("notes", "")}),
                     "source_record_sha256": sheet["record_sha256"],
                     "dataset_identity": data["identity"], "review_view_sha256": view_hash or ""})
    # Verify both source and optional review snapshot before writing any CSV.
    if load_dataset(root)["identity"] != data["identity"]:
        raise ValueError("dataset changed while preparing overview")
    if view and get_views(data, view)[3] != view_hash:
        raise ValueError("human view changed while preparing overview")
    write_csv(output, rows)
    # Parse our projection only: exact per-cell check, including embedded commas/newlines.
    with output.open(encoding="utf-8-sig", newline="") as stream:
        actual = list(csv.DictReader(stream, strict=True))
    if actual != rows or load_dataset(root)["identity"] != data["identity"]:
        raise ValueError("overview/source consistency check failed; CSV is not certified")
    print(cell({"status": "derived_overview_written_semantics_unverified", "rows": len(rows),
                "path": str(output), "sha256": sha(output), "dataset_identity": data["identity"],
                "review_view_sha256": view_hash, "encoding": "UTF-8 BOM", "gold_created": False}))


def export(data, output):
    by_id, pairs = catalogue(data)
    questions = sorted(by_id.values(), key=lambda q: (q["split"], q["family_id"], q["question_id"]))
    pair_rows = sorted(pairs.values(), key=lambda p: (by_id[p["current_question_id"]]["family_id"], p["current_question_id"], p["origin_question_id"]))
    write_csv(output / "questions.csv", [question_sheet(q, data) for q in questions])
    write_csv(output / "cache_pairs.csv", [pair_sheet(p, by_id) for p in pair_rows])
    write_csv(output / "turns.csv", [turn_sheet(t) for t in data["turns"]])
    write_json(output / "export_manifest.json", {"schema": "research_review_export_v1", "dataset_identity": data["identity"],
               "tool_identity": tool_identity(),
               "source_hashes": data["hashes"], "counts": {"questions": len(questions), "pairs": len(pairs), "turns": len(data["turns"])},
               "encoding": "UTF-8 BOM CSV; status defaults are not human approval",
               "original_csv_sha256": {p.name: sha(p) for p in output.glob("*.csv")}})


def checked_csv(path, templates, key, editable):
    expected_fields = set(next(iter(templates.values())))
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if len(reader.fieldnames or []) != len(expected_fields) or set(reader.fieldnames or []) != expected_fields:
            raise ValueError(f"{path}: headers changed/duplicated; only edit review columns")
        for line, row in enumerate(reader, 2):
            identifier = row.get(key)
            if identifier not in templates or identifier in result or None in row or any(v is None for v in row.values()):
                raise ValueError(f"{path}:{line}: unknown/duplicate ID or malformed row {identifier}")
            for field in expected_fields - editable:
                if row[field] != templates[identifier][field]:
                    raise ValueError(f"{identifier}.{field}: immutable display/source changed; use override fields")
            result[identifier] = row
    return result


def audit(row, statuses):
    for field in statuses:
        if row[field] not in STATES:
            raise ValueError(f"invalid {field}: {row[field]}")
    if any(row[f] != "needs_review" for f in statuses):
        if not all(row[f].strip() for f in ("reviewer", "reviewed_at", "evidence")):
            raise ValueError("verified/rejected decisions need reviewer, ISO timestamp and evidence/reason")
        timestamp = datetime.datetime.fromisoformat(row["reviewed_at"].replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("reviewed_at needs timezone, e.g. 2026-10-24T14:00:00+08:00")


def merge(data, bundle, output, questions_csv=None, pairs_csv=None, turns_csv=None):
    if (bundle / "failed.json").exists() or read_json(bundle / "completed.json").get("status") != "completed":
        raise ValueError("review export did not complete successfully")
    exported = read_json(bundle / "export_manifest.json")
    if exported["dataset_identity"] != data["identity"] or exported["source_hashes"] != data["hashes"]:
        raise ValueError("review bundle belongs to another dataset version")
    by_id, pairs = catalogue(data)
    paths = {"questions": questions_csv or bundle / "questions.csv", "pairs": pairs_csv or bundle / "cache_pairs.csv",
             "turns": turns_csv or bundle / "turns.csv"}
    input_hashes = {str(p): sha(p) for p in paths.values()}
    reviews = checked_csv(paths["questions"], {k: question_sheet(q, data) for k, q in by_id.items()}, "question_id", Q_EDIT)
    pair_reviews = checked_csv(paths["pairs"], {k: pair_sheet(p, by_id) for k, p in pairs.items()}, "pair_id", P_EDIT)
    turn_reviews = checked_csv(paths["turns"], {t["turn_id"]: turn_sheet(t) for t in data["turns"]}, "turn_id", T_EDIT)
    qview, pview, tview = [], [], []
    for q in data["questions"]:
        row = reviews.get(q["question_id"], question_sheet(q, data))
        audit(row, ("question_status", "answer_status"))
        answer = q["reference_answer"]
        if row["answer_status"] == "verified":
            if row["question_status"] != "verified":
                raise ValueError(f"{q['question_id']}: verified answer needs verified question")
            if row["answer_action"] == "keep_candidate" and answer:
                pass  # Explicit human choice, never inferred from a nonempty AI answer.
            elif row["answer_action"] == "override" and row["answer_override"].strip():
                answer = row["answer_override"]
            elif row["answer_action"] == "unavailable":
                answer = None
            else:
                raise ValueError(f"{q['question_id']}: choose keep_candidate/override/unavailable explicitly")
        qview.append({**q, "human_review": {k: row[k] for k in sorted(Q_EDIT)}, "effective_reference_answer": answer,
                      "effective_answer_status": row["answer_status"], "effective_question_status": row["question_status"]})
    for key, p in pairs.items():
        row = pair_reviews.get(key, pair_sheet(p, by_id))
        audit(row, ("reuse_status",))
        if row["reuse_decision"] not in {"reusable", "not_reusable", "unknown"}:
            raise ValueError(f"{key}: invalid reuse decision")
        if row["reuse_status"] == "verified" and row["reuse_decision"] == "unknown":
            raise ValueError(f"{key}: unknown reuse cannot be verified")
        pview.append({**p, "human_review": {k: row[k] for k in sorted(P_EDIT)}})
    for t in data["turns"]:
        row = turn_reviews.get(t["turn_id"], turn_sheet(t))
        audit(row, ("turn_status",))
        label, required = t["same_topic_or_shift"], t["required_history_turn_ids"]
        if row["turn_status"] == "verified":
            if row["turn_action"] == "override":
                label = row["topic_label_override"]
                required = json.loads(row["required_history_override"])
            elif row["turn_action"] != "keep_candidate":
                raise ValueError("verified turn needs explicit keep_candidate/override")
        prior = {m["turn_id"] for m in t["history"] if m["role"] == "user"}
        if label not in {"initial", "same_topic", "shift"} or (t["turn_index"] == 1) != (label == "initial"):
            raise ValueError(f"{t['turn_id']}: invalid initial/topic label")
        if not isinstance(required, list) or any(not isinstance(v, str) or v not in prior for v in required) or len(set(required)) != len(required):
            raise ValueError(f"{t['turn_id']}: invalid required history override")
        tview.append({**t, "human_review": {k: row[k] for k in sorted(T_EDIT)}, "effective_topic_label": label,
                      "effective_required_history": required,
                      "effective_removable_history": [m["turn_id"] for m in t["history"]
                                                       if m["role"] == "user" and m["turn_id"] not in required],
                      "effective_turn_status": row["turn_status"]})
    if input_hashes != {str(p): sha(p) for p in paths.values()}:
        raise ValueError("review CSV changed while merging; close Excel and use a saved snapshot")
    write_rows(output / "questions.view.jsonl", qview)
    write_rows(output / "pairs.view.jsonl", pview)
    write_rows(output / "turns.view.jsonl", tview)
    write_json(output / "view_manifest.json", {"schema": "research_review_view_v1", "dataset_identity": data["identity"],
               "tool_identity": tool_identity(),
               "dataset_hashes": data["hashes"], "review_export_manifest_sha256": sha(bundle / "export_manifest.json"),
               "review_input_sha256": input_hashes,
               "files_sha256": {p.name: sha(p) for p in output.glob("*.jsonl")}, "automatically_verified": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "merge", "overview"))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--dataset", choices=("unified_research_v1", "experiment_suites_v1", "experiment_suites_v1_r4"), default="unified_research_v1")
    parser.add_argument("--output-dir", help="required for export/merge; NEW result directory")
    parser.add_argument("--output", default="experiments/datasets/題庫總覽.csv", help="overview-only NEW CSV")
    parser.add_argument("--view", help="overview-only completed human-merged view")
    parser.add_argument("--bundle")
    parser.add_argument("--questions-csv")
    parser.add_argument("--pairs-csv")
    parser.add_argument("--turns-csv")
    args = parser.parse_args()
    root = args.project_root.resolve()
    data = load_dataset(root, args.dataset)
    if args.action == "overview":
        if args.dataset != "unified_research_v1":
            parser.error("suite uses experiments.research_tools.suites export-review for its six-column reading sheet")
        if args.output_dir or args.bundle or args.questions_csv or args.pairs_csv or args.turns_csv:
            parser.error("overview uses --output and optional --view, not review merge/export arguments")
        overview(root, data, locate(root, args.output), locate(root, args.view) if args.view else None)
        return
    if not args.output_dir or args.view:
        parser.error("export/merge require --output-dir; --view belongs to overview")
    if args.action == "merge" and not args.bundle:
        parser.error("merge requires --bundle")
    output = new_output(root, args.output_dir)
    try:
        if args.action == "export":
            export(data, output)
        else:
            merge(data, locate(root, args.bundle), output,
                  **{k: locate(root, getattr(args, k)) if getattr(args, k) else None
                     for k in ("questions_csv", "pairs_csv", "turns_csv")})
        if load_dataset(root, args.dataset)["identity"] != data["identity"]:
            raise ValueError("dataset changed during review export/merge")
        write_json(output / "completed.json", {"status": "completed", "action": args.action, "dataset_identity": data["identity"]})
    except BaseException as error:
        import traceback
        write_json(output / "failed.json", {"error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
