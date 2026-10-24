"""Integrated r4 review projection; reuse the existing XLSX writer on the GPU PC."""
from __future__ import annotations

from .common import sha
from .reporting import workbook
from .review import catalogue, turn_reason


def tables(data):
    audit = data.get("integrated_review")
    if not audit or audit["revision"] != "spoken_candidate_r4":
        raise ValueError("integrated export requires the explicitly selected r4 candidate")
    plan = data["suite"]
    index = {q["question_id"]: q["question"] for q in data["questions"]}
    for c in plan["ti"]:
        index.update({c["conversation_id"] + f"-T{i + 1}": q for i, q in enumerate(c["queries"])})
    questions = []
    for key, source in audit["questions"].items():
        decisions = [d for d in audit["decisions"] if d["question_id"] == key]
        questions.append({"Question ID": key, "r3原文": decisions[0]["original"],
                          "Excel建議": list(dict.fromkeys(d["suggested"] for d in decisions)),
                          "r4最終候選": index[key], "實際修改": index[key] != decisions[0]["original"],
                          "決策": list(dict.fromkeys(d["decision"] for d in decisions)),
                          "原因": list(dict.fromkeys(d["reason"] for d in decisions)),
                          "所有引用": [d["reference_id"] for d in decisions],
                          "審核狀態": "needs_review", "來源審核": source})
    sources = []
    for row in audit["answer_sources"]:
        sources.append({**row, "r4_question_texts": list(dict.fromkeys(index[key] for key in row["question_ids"])),
                        "note": "source_supported only means partial local evidence; not complete official verification"})
    sc = []
    for group in plan["sc"]:
        sc.append({"group": group["group_id"], "split": group["split"],
                   **{f"Q{i + 1}": index[key] for i, key in enumerate(group["question_ids"])},
                   "question_ids": group["question_ids"], "relations": group["relations"],
                   "reuse_status": "needs_review", "boundary": group["semantic_boundary_review"],
                   "source_review": [audit["questions"][key] for key in dict.fromkeys(group["question_ids"])]})
    ti = [{"conversation": c["conversation_id"], "split": c["split"],
           **{f"T{i + 1}": q for i, q in enumerate(c["queries"])},
           "topic_ids": c["topic_ids"], "shift_labels": c["shift_labels"],
           "required_history": c["required_history_turn_indices"], "prior_exposure": c["prior_exposure"],
           "annotation_basis": c["topic_annotation_basis"], "status": "needs_review"} for c in plan["ti"]]
    by_id, pairs = catalogue(data)
    required = [p for p in pairs.values() if p.get("required_for")]
    rules = []
    for rule in sorted({p["candidate_rule_id"] for p in required}):
        members = [p for p in required if p["candidate_rule_id"] == rule]
        rules.append({"candidate_rule_id": rule, "count": len(members),
                      "candidate_reason": members[0]["candidate_reason"],
                      "scope": "workload-reachable pairs only; cold SC and cold/prefilled INT",
                      "status": "needs_review", "human_gold": False,
                      "limitation": "Grouping is navigation only, NOT a shared safety proof. Review exact answers, conditions and source hashes for every applicable pair."})
    trace = [{**p,
              "origin_question": by_id[p["origin_question_id"]]["question"],
              "current_question": by_id[p["current_question_id"]]["question"],
              "origin_candidate_answer": by_id[p["origin_question_id"]].get("reference_answer"),
              "current_candidate_answer": by_id[p["current_question_id"]].get("reference_answer"),
              "origin_source_hash": by_id[p["origin_question_id"]].get("source_hash"),
              "current_source_hash": by_id[p["current_question_id"]].get("source_hash"),
              "human_status": "needs_review"} for p in required]
    reasons = [{"turn_id": t["turn_id"], "conversation": t.get("suite_conversation_id"),
                "question": t["question"], "history": t["history"],
                "candidate_label": t["same_topic_or_shift"], "reason": turn_reason(t),
                "status": "needs_review"} for t in data["turns"]]
    return {"Overview": [{"version": plan["revision"], "human_gold_certified": False,
                          "counts": plan["counts"], **audit["counts"], "input_files": audit["inputs"]}],
            "Questions": questions, "Answer Sources": sources, "SC Review": sc, "TI Review": ti,
            "SC Pair Rules": rules, "SC Pair Trace": trace, "TI Turn Reasons": reasons,
            "Blocking Issues": audit["blocking_issues"]}


def export(data, path):
    if path.exists() or path.suffix.lower() != ".xlsx":
        raise ValueError("integrated workbook must be a NEW .xlsx; never overwrite review edits")
    workbook(path, tables(data))
    return sha(path)
