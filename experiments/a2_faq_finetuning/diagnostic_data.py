"""Read-only checks for the separate, Train-intent hard diagnostic draft.

Existing final-query text is used ONLY as a collision reference. It is never
returned as a query, model input, gold-selection hint or semantic example.
"""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import re

from .common import (EXPECTED_SPLITS, assert_input_hashes, load_source_faqs,
                     read_jsonl, sha256_file)
from .validate_a2_data import (add, collect_validation, normalized, scan_text,
                              validate_corpus)

DIFFICULTY_TYPES = ("indirect_semantic", "intent_discrimination", "natural_colloquial")
FIELDS = {"query_id", "query", "gold_faq_id", "difficulty_type",
          "confusable_faq_ids", "rationale", "review_status"}


def collect_diagnostic_validation(root, paths, diagnostic_path):
    # Do not run the formal query validator or inspect final-test performance.
    report = collect_validation(root, paths, include_eval=False)
    if not report["structural_ok"]:
        return report
    issues = report["issues"]
    extra_paths = (paths["candidates"], paths["queries"], diagnostic_path)
    try:
        extra_hashes = {str(path): sha256_file(path) for path in extra_paths}
        faqs = load_source_faqs(root)
        by_id = {row["faq_id"]: row for row in faqs}
        train_ids = set(EXPECTED_SPLITS["train"])
        corpus = read_jsonl(paths["candidates"], issues)
        validate_corpus(corpus, by_id, issues)
        training = read_jsonl(paths["training"], issues)
        existing = read_jsonl(paths["queries"], issues)
        historical = read_jsonl(paths["diagnostics"], issues)
        rows = read_jsonl(diagnostic_path, issues)
        # No old query IDs/gold labels/splits enter diagnostic examples.
        refs = [(row["question"], "source_question") for row in faqs]
        for row in training:
            for field in ("positive", "negative", "hard_negative"):
                refs.extend((text, "training_" + field) for text in row.get(field, [])
                            if isinstance(text, str))
        for records, field, label in ((existing, "query", "existing_evaluation_text"),
                                      (historical, "question", "historical_diagnostic_text")):
            for row in records:
                if not isinstance(row.get(field), str) or not row[field].strip():
                    add(issues, "error", "collision_reference_schema", label,
                        "missing comparison text; cannot establish collision coverage")
                else:
                    refs.append((row[field], label))
        if len(existing) != 138:
            add(issues, "error", "collision_reference_count", "existing_evaluation_text",
                "expected the unchanged 138-query collision reference")
        normalized_refs = {normalized(text) for text, _ in refs}
        ids, texts, types, golds, pending = [], [], Counter(), Counter(), []
        valid = []
        for number, row in enumerate(rows, 1):
            location = "hard diagnostic row %d" % number
            if set(row) != FIELDS:
                add(issues, "error", "diagnostic_fields", location,
                    "expected exactly seven diagnostic fields")
                continue
            strings = FIELDS - {"confusable_faq_ids"}
            if any(not isinstance(row[key], str) or not row[key].strip() for key in strings):
                add(issues, "error", "diagnostic_string_type", location,
                    "all scalar fields must be nonempty strings")
                continue
            query_id, text, gold = row["query_id"], row["query"], row["gold_faq_id"]
            valid.append(row)
            ids.append(query_id)
            texts.append(normalized(text))
            types[row["difficulty_type"]] += 1
            golds[gold] += 1
            prefix = {"indirect_semantic": "I", "intent_discrimination": "C",
                      "natural_colloquial": "N"}.get(row["difficulty_type"])
            if prefix is None or not re.fullmatch("HD_" + (prefix or "?") + r"(?:0[1-9]|10)", query_id):
                add(issues, "error", "diagnostic_id_type", query_id,
                    "expected HD_I01..10 / HD_C01..10 / HD_N01..10 with matching type")
            if gold not in train_ids:
                add(issues, "error", "diagnostic_gold", query_id,
                    "diagnostic gold must be an existing Train intent, never a held-out ID")
            confusable = row["confusable_faq_ids"]
            if (not isinstance(confusable, list) or not confusable or
                    any(not isinstance(item, str) or item not in train_ids or item == gold
                        for item in confusable)):
                add(issues, "error", "diagnostic_confusable", query_id,
                    "expected nonempty list of other Train IDs")
            elif len(confusable) != len(set(confusable)):
                add(issues, "error", "diagnostic_confusable_duplicate", query_id,
                    "confusable IDs must not repeat")
            if row["review_status"] not in ("needs_review", "approved"):
                add(issues, "error", "diagnostic_review_status", query_id,
                    "expected needs_review or approved; checker never approves")
            if row["review_status"] != "approved":
                pending.append(query_id)
            scan_text(text, query_id + ".query", issues)
            if normalized(text) in normalized_refs:
                add(issues, "error", "diagnostic_copy", query_id,
                    "query copies a source/training/existing-evaluation/historical text after normalization")
            near_labels = sorted({label for reference, label in refs
                                  if SequenceMatcher(None, normalized(text), normalized(reference),
                                                     autojunk=False).ratio() >= 0.82})
            if near_labels:
                # No protected final-test reference content/IDs in the report.
                add(issues, "review", "diagnostic_near_reference", query_id,
                    "lexical similarity only; independently review wording and intent",
                    reference_families=near_labels)
        for label, values in (("query_id", ids), ("normalized_query", texts)):
            if len(values) != len(set(values)):
                add(issues, "error", "diagnostic_duplicate", label, "duplicate diagnostic value")
        for index, first in enumerate(valid):
            for second in valid[index + 1:]:
                if SequenceMatcher(None, normalized(first["query"]), normalized(second["query"]),
                                   autojunk=False).ratio() >= 0.82:
                    add(issues, "review", "diagnostic_near_pair", first["query_id"],
                        "check near-duplicate diagnostic wording", other=second["query_id"])
        if len(rows) != 30 or any(types[kind] != 10 for kind in DIFFICULTY_TYPES):
            add(issues, "error", "diagnostic_coverage", "diagnostic",
                "expected 30 rows, exactly 10 of each difficulty_type")
        report["statistics"]["hard_diagnostic"] = {
            "query_count": len(rows), "difficulty_counts": dict(types),
            "gold_intent_count": len(golds), "gold_counts": dict(golds),
            "unreviewed_query_ids": pending,
        }
        report["input_sha256"].update(extra_hashes)
        assert_input_hashes(report["input_sha256"])
    except (OSError, ValueError, UnicodeError) as exc:
        add(issues, "error", "diagnostic_input", "inputs", str(exc))
    report["error_count"] = sum(item["severity"] == "error" for item in issues)
    report["review_count"] = sum(item["severity"] == "review" for item in issues)
    report["structural_ok"] = report["error_count"] == 0
    report["semantic_approval"] = False
    report["purpose"] = "provisional_development_diagnostic_not_final_test"
    return report
