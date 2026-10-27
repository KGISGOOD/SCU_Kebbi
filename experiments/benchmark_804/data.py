"""Allocation validation, traceable exposure audit, no rewriting of test questions."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import unicodedata

from .common import INPUT, GUIDE, path, read, rows, sha

LABELS = {"FAQ比對", "RAG問答", "LoRA回答", "推測解碼候選", "語意快取",
          "問答安全/不確定性", "提示注入防護", "多輪問答", "主題辨識"}


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def load(root):
    data = rows(root / INPUT)
    if len(data) != 804 or [r["id"] for r in data] != ["Q%04d" % i for i in range(1, 805)]:
        raise ValueError("804 original ordered IDs required; never renumber/filter the source")
    groups = defaultdict(list)
    for row in data:
        required = {"id", "question", "type", "suites", "conversation_id", "turn_index",
                    "review_status", "original_line", "gold_answer", "gold_source_ids"}
        if set(row) != required or any(not isinstance(row[k], str) or not row[k].strip()
                                       for k in ("id", "question", "type", "review_status")):
            raise ValueError("invalid source schema: " + row["id"])
        if (not isinstance(row["suites"], list) or len(set(row["suites"])) != len(row["suites"])
                or not set(row["suites"]) <= LABELS or type(row["original_line"]) is not int
                or not isinstance(row["gold_source_ids"], list)
                or row["gold_answer"] is not None and not isinstance(row["gold_answer"], str)):
            raise ValueError("invalid allocation/answer fields: " + row["id"])
        if row["conversation_id"]:
            if row["type"] != "多輪對話" or type(row["turn_index"]) is not int:
                raise ValueError("invalid conversation: " + row["id"])
            groups[row["conversation_id"]].append(row)
        elif row["turn_index"] is not None:
            raise ValueError("turn without group: " + row["id"])
    for cid, turns in groups.items():
        if len(turns) != 4 or [r["turn_index"] for r in turns] != [1, 2, 3, 4]:
            raise ValueError("candidate conversation is not ordered four turns: " + cid)
    if [r["original_line"] for r in data] != sorted({r["original_line"] for r in data}):
        raise ValueError("original line order changed")
    return data, {"questions": len(data), "input_sha256": sha(root / INPUT), "guide_sha256": sha(root / GUIDE),
                  "types": dict(Counter(r["type"] for r in data)),
                  "labels": dict(Counter(s for r in data for s in r["suites"])),
                  "candidate_conversations": len(groups), "candidate_grouped_turns": sum(map(len, groups.values())),
                  "ungrouped_turns": sum(r["type"] == "多輪對話" and not r["conversation_id"] for r in data),
                  "gold_status": "pending_independent_annotation", "groups_verified_by_source": False}


def review(root, config, data):
    if not config.get("review_file"):
        return {"questions": {}, "conversations": {}, "cache_pairs": []}
    value = read(path(root, config["review_file"]))
    if value["dataset_sha256"] != sha(root / INPUT):
        raise ValueError("annotation overlay belongs to another test version")
    by_id = {r["id"]: r for r in data}
    for qid, annotation in value.get("questions", {}).items():
        if qid not in by_id or annotation.get("status") != "verified" or not annotation.get("evidence"):
            raise ValueError("invalid human question annotation: " + qid)
    for cid, annotation in value.get("conversations", {}).items():
        expected = [r["id"] for r in data if r["conversation_id"] == cid]
        if (not expected or annotation.get("question_ids") != expected
                or annotation.get("status") != "verified" or not annotation.get("evidence")
                or annotation.get("history_scope") not in {"previous_user_questions_only", "full_reviewed_messages"}):
            raise ValueError("conversation review must confirm original grouping and history scope: " + cid)
        labels = annotation.get("shift_labels", {})
        if set(labels) != set(expected[1:]) or any(type(v) is not bool for v in labels.values()):
            raise ValueError("missing genuine turn-shift labels: " + cid)
        if annotation["history_scope"] == "full_reviewed_messages":
            histories = annotation.get("histories", {})
            if set(histories) != set(expected[1:]):
                raise ValueError("full reviewed histories missing: " + cid)
            for index, qid in enumerate(expected[1:], 1):
                history = histories[qid]
                if (len(history) != index * 2 or [m.get("role") for m in history] != ["user", "assistant"] * index
                        or [m.get("content") for m in history[::2]] != [by_id[k]["question"] for k in expected[:index]]
                        or any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in history)):
                    raise ValueError("reviewed history changes original turns or lacks assistant replies: " + qid)
    for p in value.get("cache_pairs", []):
        if (p.get("origin_id") not in by_id or p.get("query_id") not in by_id
                or type(p.get("reusable")) is not bool or p.get("status") != "verified" or not p.get("evidence")):
            raise ValueError("invalid reviewed reuse pair")
    if not value.get("reviewer") or not value.get("reviewed_at"):
        raise ValueError("annotation overlay needs real reviewer/time; never autogenerated")
    if datetime.fromisoformat(value["reviewed_at"].replace("Z", "+00:00")).tzinfo is None:
        raise ValueError("review timestamp needs timezone")
    return value


def exposure(root, config, data):
    bundle = path(root, config["training_bundle"])
    from experiments.a2_faq_finetuning.v2_data import load_bundle
    converted, manifest, hashes = load_bundle(root, bundle)
    master = rows(path(root, manifest["frozen_dir"]) / "qa_master.jsonl")
    by_norm = defaultdict(list)
    for r in converted["train.jsonl"]:
        by_norm[normalize(r["messages"][1]["content"])].append(r["faq_id"])
    canonical = [{"faq_id": r["intent_group_id"], "question": r["question"], "answer": r["answer"],
                  "source_question_id": r["question_id"], "sources": r["sources"]}
                 for r in master if r["question_id"] == r["canonical_question_id"]]
    faq_norm = {normalize(r["question"]): r["faq_id"] for r in canonical}
    findings = []
    train_grams = [(qid, set(q[i:i + 3] for i in range(max(0, len(q) - 2))))
                   for q, ids in by_norm.items() for qid in ids]
    for r in data:
        q = normalize(r["question"])
        grams = set(q[i:i + 3] for i in range(max(0, len(q) - 2)))
        near = [qid for qid, other in train_grams if grams and other
                and len(grams & other) / len(grams | other) >= .85]
        findings.append({"question_id": r["id"], "exact_train_question_ids": by_norm.get(q, []),
                         "exact_faq_candidate_id": faq_norm.get(q), "near_lexical_train_ids": near,
                         "semantic_overlap": "unverified; common knowledge exposure not inferred from string distance"})
    return canonical, findings, hashes


def select(data, label):
    return [r for r in data if label in r["suites"] and r["type"] != "多輪對話"]


def jobs(data, annotations, config):
    result = [{"id": "qa_" + a, "kind": "qa", "architecture": a} for a in
              ("A0", "A1", "A2-1", "A2-2", "A3", "A4-1", "A4-2", "A5")]
    result += [{"id": "sc_" + m, "kind": "sc", "mode": m} for m in ("none", "exact", "semantic")]
    result += [{"id": "sd", "kind": "sd", "draft": True},
               {"id": "prompt_lookup", "kind": "sd", "draft": False}]
    result += [{"id": "ti", "kind": "ti", "available": bool(annotations.get("conversations")) and bool(config.get("ti_thresholds")),
                "reason": "requires confirmed original groups/turn labels and independently frozen thresholds; never calibrate on 804"}]
    for decoder in ("draft_1b", "prompt_lookup"):
        result.append({"id": "int_" + decoder, "kind": "int", "decoder": decoder})
    result += [{"id": "int_attention", "kind": "unsupported", "available": False,
                "reason": "no reviewed Attention+SC+SD online integration; do not claim existing TI context experiment completes this"},
               {"id": "multi_turn_qa", "kind": "unsupported", "available": False,
                "reason": "804 groups/assistant histories unverified; original eight QA methods are single-turn"}]
    return result
