"""Standard-library validation only. Read-only inputs; reports require a NEW directory.

This program has not been executed on the development computer. String checks
can identify review candidates, never certify semantic independence or truth.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import unicodedata


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def decode(text):
    def bad(value):
        raise ValueError(f"nonstandard JSON constant: {value}")
    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=bad)


def read_json(path):
    return decode(path.read_text(encoding="utf-8-sig"))


def read_rows(path):
    rows = []
    for line, text in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        try:
            row = decode(text)
            if not isinstance(row, dict):
                raise ValueError("expected object")
            rows.append(row)
        except ValueError as error:
            raise ValueError(f"{path}:{line}: {error}") from error
    return rows


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def schema_errors(value, schema, where):
    """Evaluate the simple JSON Schema subset used by this dataset, no dependencies."""
    errors = []
    kinds = schema.get("type", [])
    kinds = kinds if isinstance(kinds, list) else [kinds]
    checks = {"object": isinstance(value, dict), "array": isinstance(value, list),
              "string": isinstance(value, str), "integer": type(value) is int,
              "number": type(value) in (int, float), "boolean": type(value) is bool,
              "null": value is None}
    if kinds and not any(checks[k] for k in kinds):
        return [f"{where}: expected {kinds}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{where}: invalid enum {value!r}")
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        errors.append(f"{where}: string too short")
    if isinstance(value, dict):
        errors += [f"{where}: missing {key}" for key in schema.get("required", []) if key not in value]
        for key, subschema in schema.get("properties", {}).items():
            if key in value:
                errors += schema_errors(value[key], subschema, f"{where}.{key}")
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            errors += schema_errors(item, schema["items"], f"{where}[{index}]")
    return errors


def validate(root, directory, require_reviewed=False):
    errors, warnings, leakage = [], [], []
    manifest = read_json(directory / "dataset_manifest.json")
    registry = read_json(directory / "source_registry.json")
    schema = read_json(directory / "schema.json")["$defs"]
    groups = {split: read_rows(directory / f"questions_{split}.jsonl")
              for split in ("historical", "calibration", "evaluation")}
    main = sum(groups.values(), [])
    legacy = read_rows(directory / "questions_legacy_fixed60.jsonl")
    all_questions = main + legacy
    by_id = {}
    for row in all_questions:
        key = row.get("question_id", "<missing>")
        errors += schema_errors(row, schema["question"], key)
        if key in by_id:
            errors.append(f"duplicate question_id: {key}")
        by_id[key] = row
    if errors:
        return {"status": "failed", "stage": "question_schema", "errors": errors,
                "warnings": warnings, "human_gold_certified": False}
    for split, rows in groups.items():
        if any(r["split"] != split for r in rows):
            errors.append(f"file/split mismatch: {split}")
        if len(rows) != manifest["main_counts"][split]:
            errors.append(f"manifest count mismatch: {split}")
    for filename, expected in manifest.get("artifact_sha256", {}).items():
        path = directory / filename
        if not path.is_file() or sha(path) != expected:
            errors.append(f"dataset artifact changed: {filename}")
    sources = registry["sources"]
    source_cache = {}
    for recorded in registry["inventory"]:
        path = (root / recorded["path"]).resolve()
        if not path.is_relative_to(root):
            errors.append(f"inventory source outside project: {recorded['path']}")
            continue
        if not path.is_file():
            warnings.append(f"audit-only inventory source absent on this machine: {recorded['path']}")
            continue
        source_cache[path] = sha(path)
        if source_cache[path] != recorded["sha256"]:
            warnings.append(f"audit-only inventory source changed: {recorded['path']}")
    for source_id, source in sources.items():
        path = (root / source["path"]).resolve()
        if not path.is_relative_to(root):
            errors.append(f"source escapes project: {source_id}")
            continue
        if not path.is_file():
            errors.append(f"missing source: {source_id}: {source['path']}")
            continue
        if path not in source_cache:
            source_cache[path] = sha(path)
        if source_cache[path] != source["sha256"]:
            errors.append(f"source SHA256 mismatch: {source_id}")
    for row in all_questions:
        qid = row["question_id"]
        for ref in row["related_question_ids"]:
            if ref not in by_id:
                errors.append(f"{qid}: unresolved related_question_id {ref}")
        for ref in row["source_document_ids"]:
            if ref not in sources or row["source_hash"].get(ref) != sources[ref]["sha256"]:
                errors.append(f"{qid}: unresolved source/hash {ref}")
        if row["reference_answer_status"] == "unavailable" and row["reference_answer"] is not None:
            errors.append(f"{qid}: unavailable answer must be null")
        if row["reference_answer_status"] != "unavailable" and not row["reference_answer"]:
            errors.append(f"{qid}: answer status requires nonempty text")
        if row["reference_answer_status"] == "verified" and row["review_status"] != "reviewed":
            errors.append(f"{qid}: unreviewed answer cannot be verified")
        if row["reference_answer"] and not row["source_document_ids"]:
            errors.append(f"{qid}: answer without source")
        if require_reviewed and row["review_status"] != "reviewed":
            errors.append(f"{qid}: human review required")
        if require_reviewed and row["reference_answer_status"] == "needs_review":
            errors.append(f"{qid}: reference answer still needs verification")
        for field in ("question", "reference_answer"):
            for char in row[field] or "":
                if char == "\ufffd" or 0xE000 <= ord(char) <= 0xF8FF or (ord(char) < 32 and char not in "\n\r\t"):
                    errors.append(f"{qid}.{field}: suspicious Unicode U+{ord(char):04X}")
                    break
    # Match original CSV fields byte-for-Unicode-character, without rewriting.
    original = {}
    for relative in sorted({s["path"] for k, s in sources.items() if k.startswith("FAQ_")}):
        with (root / relative).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = row["編號"]
                if key in original:
                    errors.append(f"duplicate source FAQ ID {key}")
                original[key] = (row["問題內容"], row["ai答覆"])
    if set(original) != {f"{p}{n:02d}" for p in "PS" for n in range(1, 31)}:
        errors.append("source CSV FAQ IDs must be P01-P30/S01-S30")
    if set(r["question_id"] for r in groups["historical"]) != set(original):
        errors.append("historical IDs differ from source CSV")
    for row in groups["historical"]:
        if original.get(row["question_id"]) != (row["question"], row["reference_answer"]):
            errors.append(f"{row['question_id']}: historical Q/A differs from source")
    legacy_source = root / sources["LEGACY_FIXED60"]["path"]
    source_legacy = {r["question_id"]: r["question"] for r in read_rows(legacy_source)}
    if {r["question_id"]: r["question"] for r in legacy} != source_legacy:
        errors.append("legacy fixed60 IDs/text changed")
    relations = read_rows(directory / "question_relations.jsonl")
    if len({r["relation_id"] for r in relations}) != len(relations):
        errors.append("duplicate relation_id")
    declared_repeats = set()
    for row in relations:
        if require_reviewed and row["review_status"] != "reviewed":
            errors.append(f"{row['relation_id']}: relation review required")
        a, b = row["from_question_id"], row["to_question_id"]
        if a not in by_id or b not in by_id:
            errors.append(f"unresolved relation {row['relation_id']}")
        elif row["relation_type"] == "exact_repeat":
            if by_id[a]["question"] != by_id[b]["question"]:
                errors.append(f"not an exact repeat: {a}/{b}")
            declared_repeats.add(frozenset((a, b)))
    # Semantic risk candidates, NOT a semantic leakage certificate.
    text_groups, family_groups = defaultdict(list), defaultdict(list)
    for row in main:
        text_groups[normalized(row["question"])].append(row)
        family_groups[row["family_id"]].append(row)
    for same in text_groups.values():
        if len(same) < 2:
            continue
        splits = {r["split"] for r in same}
        entry = {"kind": "exact_question_duplicate", "ids": [r["question_id"] for r in same], "splits": sorted(splits)}
        leakage.append(entry)
        if {"calibration", "evaluation"} <= splits:
            errors.append(f"cross-split exact query: {entry['ids']}")
        # Same-split repeats need an explicit source relation; history duplicates
        # across splits are reported as exposure, not erased.
        for row in same:
            if row["primary_type"] == "R" and frozenset((row["question_id"], row["reuse_reference_question_id"])) not in declared_repeats:
                errors.append(f"undeclared repeat: {row['question_id']}")
    for family, rows in family_groups.items():
        if {"calibration", "evaluation"} <= {r["split"] for r in rows}:
            errors.append(f"family split across calibration/evaluation: {family}")
    for a in groups["calibration"]:
        for b in groups["evaluation"]:
            ratio = SequenceMatcher(None, normalized(a["question"]), normalized(b["question"])).ratio()
            if ratio >= 0.72:
                leakage.append({"kind": "cross_split_lexical_near_candidate", "ids": [a["question_id"], b["question_id"]],
                                "sequence_ratio": ratio, "review_status": "needs_review"})
    manifest_path = root / "experiments/datasets/faq_finetuning/faq_split_manifest.json"
    raw_splits = read_json(manifest_path)["split_scheme"]
    intent_splits = {key: raw_splits[key] for key in ("train", "validation", "test")}
    if any(set(intent_splits[a]) & set(intent_splits[b]) for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        errors.append("original intent manifest splits overlap")
    if sorted(map(len, intent_splits.values())) != [9, 9, 42] or set(sum(intent_splits.values(), [])) != set(original):
        errors.append("original intent manifest coverage/count mismatch")
    training = root / "experiments/datasets/faq_finetuning/faq_training_candidates.jsonl"
    exposure = defaultdict(list)
    for row in read_rows(training):
        for field in ("question", "positive", "negative", "hard_negative"):
            values = [row[field]] if isinstance(row[field], str) else row[field]
            for value in values:
                exposure[normalized(value)].append({"faq_id": row["faq_id"], "field": field})
    for row in main:
        matched = exposure.get(normalized(row["question"]))
        if matched:
            leakage.append({"kind": "exact_training_text_exposure", "question_id": row["question_id"], "matches": matched})
        elif row["split"] != "historical":
            near = []
            query = normalized(row["question"])
            for prior_text, locations in exposure.items():
                ratio = SequenceMatcher(None, query, prior_text).ratio()
                if ratio >= 0.85:
                    near.append({"sequence_ratio": ratio, "matches": locations, "text": prior_text})
            if near:
                leakage.append({"kind": "training_lexical_near_candidates", "question_id": row["question_id"],
                                "matches": near, "review_status": "needs_review"})
        expected = sorted(set(row["supporting_intent_ids"]) & set(intent_splits["train"]))
        if sorted(row["training_overlap_details"].get("train_intent_ids", [])) != expected:
            errors.append(f"{row['question_id']}: train intent marker differs from manifest")
    # Existing query banks: exposure review only, never imported into training.
    prior_banks = [p for p in registry["inventory"] if p.get("lines") and
                   (p["path"].startswith("experiments/algorithms/") or p["path"].endswith(
                       ("faq_eval_queries.jsonl", "faq_hard_diagnostic.jsonl", "scu_faq_matching_test_v1.jsonl")))]
    for bank in prior_banks:
        prior_path = root / bank["path"]
        if not prior_path.is_file():
            warnings.append(f"prior-bank leakage scan unavailable: {bank['path']}")
            if require_reviewed:
                errors.append(f"formal gate needs prior-bank exposure audit: {bank['path']}")
            continue
        for old in read_rows(prior_path):
            text = old.get("question", old.get("query"))
            if not isinstance(text, str):
                continue
            for row in main:
                if normalized(row["question"]) == normalized(text):
                    leakage.append({"kind": "prior_bank_exact_query", "question_id": row["question_id"], "path": bank["path"]})
    conversations = read_rows(directory / "conversations.jsonl")
    conv = defaultdict(list)
    turn_ids = set()
    for row in conversations:
        issues = schema_errors(row, schema["conversation_turn"], row.get("turn_id", "turn"))
        if issues:
            return {"status": "failed", "stage": "conversation_schema", "errors": errors + issues,
                    "warnings": warnings, "human_gold_certified": False}
        if row["turn_id"] in turn_ids:
            errors.append(f"duplicate turn_id {row['turn_id']}")
        turn_ids.add(row["turn_id"])
        conv[row["conversation_id"]].append(row)
    for cid, turns in conv.items():
        prior, previous_questions = set(), []
        if len(turns) != 4 or [r["turn_index"] for r in turns] != [1, 2, 3, 4]:
            errors.append(f"{cid}: expected four ordered turns")
        if len({r["split"] for r in turns}) != 1:
            errors.append(f"{cid}: split changes within conversation")
        for row in turns:
            history = row["history"]
            if len(history) != 2 * len(prior) or any(
                    message["role"] != ("user" if index % 2 == 0 else "assistant")
                    for index, message in enumerate(history)):
                errors.append(f"{row['turn_id']}: complete alternating history pairs required")
            required, removable = set(row["required_history_turn_ids"]), set(row["removable_history_turn_ids"])
            if required & removable or required | removable != prior:
                errors.append(f"{row['turn_id']}: required/removable must partition prior turns")
            expected_history = [r for r in row["history"] if r["role"] == "user"]
            if [(r["turn_id"], r["content"]) for r in expected_history] != previous_questions:
                errors.append(f"{row['turn_id']}: history differs from prior user turns")
            if any(ref not in by_id for ref in row["source_question_ids"]):
                errors.append(f"{row['turn_id']}: unresolved source question")
            if row["family_id"] not in {r["family_id"] for r in groups[row["split"]]}:
                errors.append(f"{row['turn_id']}: unknown split/family")
            if require_reviewed and row["review_status"] != "reviewed":
                errors.append(f"{row['turn_id']}: review required")
            prior.add(row["turn_id"])
            previous_questions.append((row["turn_id"], row["question"]))
    work = read_rows(directory / "cache_workloads.jsonl")
    fills = read_rows(directory / "cache_initial_fill.jsonl")
    issues = [issue for number, row in enumerate(work, 1) for issue in
              schema_errors(row, schema["cache_event"], f"cache_workloads:{number}:{row.get('query_event_id', '?')}")]
    if issues:
        return {"status": "failed", "stage": "cache_event_schema", "errors": errors + issues,
                "warnings": warnings, "human_gold_certified": False}
    workload_manifest = read_json(directory / "cache_workload_manifest.json")["workloads"]
    if len({r["workload_id"] for r in workload_manifest}) != len(workload_manifest):
        errors.append("duplicate workload manifest ID")
    event_ids = [r["query_event_id"] for r in work]
    if len(set(event_ids)) != len(event_ids):
        errors.append("duplicate cache query event ID")
    if set(r["workload_id"] for r in work) != set(r["workload_id"] for r in workload_manifest):
        errors.append("workload manifest IDs differ from events")
    for meta in workload_manifest:
        rows = [r for r in work if r["workload_id"] == meta["workload_id"]]
        initial = [r for r in fills if r["workload_id"] == meta["workload_id"]]
        prior = set()
        for row in initial + rows:
            key = row["question_id"]
            if key not in by_id or row["question_text"] != by_id[key]["question"]:
                errors.append(f"{meta['workload_id']}: invalid query reference/text {key}")
                continue
            if row["source_split"] != meta["source_split"] or by_id[key]["split"] != meta["source_split"]:
                errors.append(f"{meta['workload_id']}: source split mismatch")
            if row.get("reference_question_id") and row["reference_question_id"] not in prior:
                errors.append(f"{meta['workload_id']}: revisit before reference")
            if "intended_repeat_type" in row:
                errors += schema_errors(row, schema["cache_event"], row["query_event_id"])
                if row["intended_repeat_type"] == "exact" and row["question_text"] != by_id[row["reference_question_id"]]["question"]:
                    errors.append(f"{row['query_event_id']}: exact-repeat text differs")
            if require_reviewed and (row["review_status"] != "reviewed" or
                    row.get("expected_reuse_review_status", "reviewed") != "reviewed"):
                errors.append(f"{meta['workload_id']}: workload review required")
            prior.add(key)
        if len(rows) != 20 or [r["query_order"] for r in rows] != list(range(1, 21)):
            errors.append(f"{meta['workload_id']}: requires 20 contiguous measured queries")
        revisit = sum(r["intended_revisit"] for r in rows)
        if revisit != meta["intended_revisit_count"] or revisit * 5 != meta["target_revisit_percent"]:
            errors.append(f"{meta['workload_id']}: revisit ratio mismatch")
        if len(initial) != meta["initial_fill_count"]:
            errors.append(f"{meta['workload_id']}: initial-fill count mismatch")
    if len(fills) != manifest["cache_initial_fill_queries"] or len(workload_manifest) != manifest["cache_workload_count"]:
        errors.append("workload/initial-fill counts differ from dataset manifest")
    if len(conv) != manifest["conversation_count"] or len(work) != manifest["cache_measured_queries"]:
        errors.append("conversation/workload counts differ from dataset manifest")
    warnings.append("Semantic leakage, answer truth, conditions and history necessity require human review; string checks do not certify them.")
    return {"status": "failed" if errors else "structural_checks_passed_semantics_unverified", "errors": errors,
            "warnings": warnings, "leakage_candidates": leakage,
            "counts": {k: len(v) for k, v in groups.items()}, "primary_types": dict(Counter(r["primary_type"] for r in main)),
            "supplemental_legacy_count": len(legacy), "conversations": len(conv), "turns": len(conversations),
            "cache_queries": len(work), "initial_fill_queries": len(fills),
            "human_gold_certified": False, "require_reviewed": require_reviewed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--data-dir", type=Path, default=Path("experiments/datasets/unified_research_v1"))
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--require-reviewed", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    directory = (root / args.data_dir).resolve()
    report_dir = (root / args.report_dir).resolve()
    if directory == report_dir or directory in report_dir.parents:
        parser.error("report-dir must be outside immutable dataset directory")
    report_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = validate(root, directory, args.require_reviewed)
    except (ValueError, KeyError, OSError, TypeError, csv.Error) as error:
        result = {"status": "failed", "errors": [f"{type(error).__name__}: {error}"], "human_gold_certified": False}
    with (report_dir / "validation_report.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "errors": len(result.get("errors", [])),
                      "report": str(report_dir / "validation_report.json")}, ensure_ascii=False))
    return 2 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
