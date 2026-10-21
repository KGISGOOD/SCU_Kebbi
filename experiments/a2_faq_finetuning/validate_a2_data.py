"""Read-only A2 structural validation plus separate human-review findings.

Exit 0 means no deterministic errors, never semantic approval. This command
does not load an embedding model, rewrite inputs, or mark rows approved.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from .common import (CSV_SOURCES, EXPECTED_SPLITS, GROUPS, DataError, add_data_arguments,
                     data_paths, load_source_faqs, read_json, read_jsonl, sha256_file,
                     resolve_path, write_json_exclusive)

TRAIN_FIELDS = {"faq_id", "question", "positive", "negative", "hard_negative",
                "review_status", "split"}
ARRAY_FIELDS = ("positive", "negative", "hard_negative")
BOUNDARIES = [
    (["P18", "P19"], "學雜費及獎助學金的廣義問題，與優秀入學獎學金條件／免學雜費子題可能包含；不能只靠字串判定唯一 gold。"),
    (["P21", "S13", "S12"], "雲端補助、校內伺服器、筆電 GPU 需求有答案交集；P21 答案直接提及系上 GPU 伺服器。"),
    (["P13", "P14", "S04"], "學業預警、導師制度、上機考緊張補救共享輔導措施；需確認完整條件能否區分。"),
    (["P05", "P06", "P07", "S22", "S23", "S24"], "連讀名額、就業是否必須讀研、考研管道錄取率、升學比例、頂大去向、跨考難度不同，但保留集共享升學主題。"),
    (["S11", "P24"], "選修外系課及學分採計不等於輔系／雙主修門檻；P24 原題跨修用語本身較寬。"),
    (["P04", "S26"], "畢業 3–5 年待遇成長與直接就業起薪的時間條件不同，不可在改寫時刪除。"),
    (["P08", "P09", "S27"], "知名企業實習及給薪、整年全職實習學分、是否推薦／強制實習不同。"),
    (["P16", "P17", "S28", "S29", "S30"], "面試偏好、相對國立競爭力、CS/MIS 定位、AI 選系不同；寬泛優勢題可能容納多種回答。"),
    (["P01", "P03", "S25", "P30"], "企業去向、公職國營類科、其他職務、就業輔導服務不同，但都屬職涯主題。"),
    (["S10", "S11", "S19", "S20", "P28", "P29"], "擋修／學分／專題流程與畢業檢定門檻、畢業率及專題延畢不能泛化成同一句。"),
    (["S15", "S16", "P27"], "系館學習設施、住宿租屋與周邊機能／治安／交通不同；生活環境主題有交集。"),
    (["P25", "P26"], "書審無競賽獎項與面試形式／特質分屬 Validation/Test，主題接近並非已證實同意圖。"),
    (["P23", "P29"], "交換延畢與專題延畢原因不同，但共享延期畢業詞彙。"),
    (["S06", "S07", "S08", "S10"], "數學負擔／當人、微積分差異、統計及公式、擋修後果需保留子意圖。"),
    (["S09", "S14", "S17", "S18"], "課表與空堂、熬夜作業、社團時間、系學會活動內容不同。"),
    (["S19", "S20", "S21"], "專題年級及找教授、分組選題、歷屆 APP/AI 案例不同；原答案可能部分互相涵蓋。"),
]
SUSPICIOUS = ("systemup", "系up", "go whichles", "高bine", "computinng",
              "跟notup", "找to", "上store", "knownd", "畢族生", "趇工", "锟斤拷")


def normalized(text):
    value = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in value
                   if unicodedata.category(char)[0] in ("L", "N"))


def add(issues, severity, code, location, message, **details):
    issues.append({"severity": severity, "code": code, "location": location,
                   "message": message, **details})


def scan_text(text, location, issues):
    if any(unicodedata.category(char) == "Cc" for char in text):
        add(issues, "error", "control_character", location,
            "text contains a control character", text=text)
    reasons = [token for token in SUSPICIOUS if token.casefold() in text.casefold()]
    if "\ufffd" in text:
        reasons.append("Unicode replacement character")
    if re.search(r"[\u3040-\u30ff]", text):
        reasons.append("Japanese kana; verify whether intentional")
    if re.search(r"Ã|Â|â€", text):
        reasons.append("possible encoding mojibake")
    if len(normalized(text)) < 6:
        reasons.append("very short text may omit intent")
    if reasons:
        add(issues, "review", "suspicious_text", location,
            "heuristic only; traditional Chinese and English terms are allowed",
            reasons=reasons, text=text)


def manifest_sets(manifest, issues):
    scheme = manifest.get("split_scheme")
    if not isinstance(scheme, dict):
        raise DataError("manifest.split_scheme must be an object")
    splits = {}
    for name, expected in EXPECTED_SPLITS.items():
        values = scheme.get(name)
        if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
            raise DataError("manifest.split_scheme.%s must be string[]" % name)
        counts = Counter(values)
        duplicates = sorted(key for key, count in counts.items() if count > 1)
        if duplicates:
            add(issues, "error", "split_duplicate", name, "duplicate IDs", ids=duplicates)
        splits[name] = set(values)
        if splits[name] != set(expected):
            add(issues, "error", "split_membership", name,
                "manifest differs from the task's fixed 42/9/9 split",
                missing=sorted(set(expected) - splits[name]),
                unexpected=sorted(splits[name] - set(expected)))
    names = list(splits)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            overlap = sorted(splits[first] & splits[second])
            if overlap:
                add(issues, "error", "split_overlap", first + "/" + second,
                    "FAQ IDs occur in multiple splits", ids=overlap)
    return splits


def validate_training(rows, faq_by_id, splits, diagnostic_refs, issues, near_threshold=0.82):
    question_ids = {faq["question"]: faq_id for faq_id, faq in faq_by_id.items()}
    ids, positive_owners, positive_norms, negative_refs = [], defaultdict(list), defaultdict(list), []
    empty_hard, total_samples, sample_refs = [], Counter(), []
    for number, row in enumerate(rows, 1):
        faq_id = row.get("faq_id")
        location = str(faq_id) if isinstance(faq_id, str) else "training row %d" % number
        if set(row) != TRAIN_FIELDS:
            add(issues, "error", "training_fields", location, "expected exactly seven fields",
                missing=sorted(TRAIN_FIELDS - set(row)), extra=sorted(set(row) - TRAIN_FIELDS))
        if not isinstance(faq_id, str) or not faq_id.strip():
            add(issues, "error", "field_type", location, "faq_id must be a nonempty string")
            continue
        ids.append(faq_id)
        for field in ("question", "review_status", "split"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                add(issues, "error", "field_type", location + "." + field,
                    "expected nonempty string")
        if faq_id not in splits["train"]:
            add(issues, "error", "not_train", location, "candidate ID is not in Train")
        if row.get("split") != "train":
            add(issues, "error", "training_split", location + ".split", "must equal train")
        if row.get("review_status") not in ("needs_review", "approved", "rejected"):
            add(issues, "error", "review_status", location, "unknown review_status")
        if row.get("review_status") == "rejected":
            add(issues, "error", "rejected_training", location, "rejected row remains in candidates")
        expected = faq_by_id.get(faq_id, {}).get("question")
        if row.get("question") != expected or expected is None:
            add(issues, "error", "question_mismatch", location + ".question",
                "must exactly equal source CSV question",
                actual=row.get("question"), expected=expected)
        if isinstance(row.get("question"), str):
            text = row["question"]
            scan_text(text, location + ".question", issues)
            sample_refs.append((location + ".question", text))
            if text in diagnostic_refs:
                add(issues, "review", "diagnostic_anchor", location + ".question",
                    "source anchor equals historical diagnostic query; old diagnostics are not independent",
                    diagnostic_ids=diagnostic_refs[text], text=text)
        fields = {}
        for field in ARRAY_FIELDS:
            value = row.get(field)
            if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
                add(issues, "error", "field_type", location + "." + field, "expected string[]")
                fields[field] = []
                continue
            fields[field] = value
            total_samples[field] += len(value)
            if field == "positive" and len(value) != 3:
                add(issues, "error", "positive_count", location, "expected three Positive paraphrases")
            if field == "hard_negative" and len(value) > 3:
                add(issues, "error", "hard_count", location, "expected 0-3 Hard Negative candidates")
            if field == "hard_negative" and not value:
                empty_hard.append(faq_id)
            counts = Counter(value)
            for text, count in counts.items():
                if count > 1:
                    add(issues, "error", "duplicate_sample", location + "." + field,
                        "duplicate sample in one array", text=text)
            norms = defaultdict(list)
            for sample_index, text in enumerate(value):
                item_loc = "%s.%s[%d]" % (faq_id, field, sample_index)
                sample_refs.append((item_loc, text))
                if not text.strip() or not normalized(text):
                    add(issues, "error", "empty_sample", item_loc, "empty or punctuation-only sample")
                scan_text(text, item_loc, issues)
                norms[normalized(text)].append(text)
                if text in diagnostic_refs:
                    add(issues, "error", "diagnostic_sample", item_loc,
                        "historical diagnostic query copied into training samples",
                        diagnostic_ids=diagnostic_refs[text], text=text)
                if field == "positive":
                    positive_owners[text].append(item_loc)
                    positive_norms[normalized(text)].append((faq_id, item_loc))
                    if text in question_ids:
                        add(issues, "error", "positive_copies_original", item_loc,
                            "Positive copies a source FAQ question",
                            source_faq_id=question_ids[text], text=text)
                    elif normalized(text) == normalized(expected or ""):
                        add(issues, "error", "positive_normalized_copy", item_loc,
                            "Positive differs from own question only in punctuation/case/width")
                else:
                    negative_refs.append((item_loc, text))
                    source_id = question_ids.get(text)
                    if source_id is None:
                        add(issues, "error", "untraceable_negative", item_loc,
                            "negative is not an exact original CSV question", text=text)
                    elif source_id == faq_id:
                        add(issues, "error", "self_negative", item_loc,
                            "negative points to its own FAQ", source_faq_id=source_id, text=text)
                    elif source_id not in splits["train"]:
                        add(issues, "error", "heldout_negative", item_loc,
                            "negative source belongs to Validation/Test",
                            source_faq_id=source_id, text=text)
            for texts in norms.values():
                if len(texts) > 1 and len(set(texts)) > 1:
                    add(issues, "review", "normalized_duplicate", location + "." + field,
                        "samples differ only after normalizing punctuation/width", texts=texts)
            if field == "positive":
                for i, first in enumerate(value):
                    for second in value[i + 1:]:
                        score = SequenceMatcher(None, normalized(first), normalized(second),
                                                autojunk=False).ratio()
                        if score >= near_threshold:
                            add(issues, "review", "near_positive", location + "." + field,
                                "high lexical similarity; review wording diversity, not proof of semantic identity",
                                first=first, second=second, ratio=score)
        overlap = set(fields.get("negative", [])) & set(fields.get("hard_negative", []))
        if overlap:
            add(issues, "error", "negative_hard_overlap", location,
                "Negative and Hard Negative overlap", texts=sorted(overlap))
    for faq_id, count in Counter(ids).items():
        if count > 1:
            add(issues, "error", "duplicate_faq_id", faq_id, "duplicate training FAQ ID")
    if len(rows) != 42 or set(ids) != splits["train"]:
        add(issues, "error", "training_coverage", "training",
            "expected exactly 42 unique Train rows",
            row_count=len(rows), missing=sorted(splits["train"] - set(ids)),
            unexpected=sorted(set(ids) - splits["train"]))
    for text, owners in positive_owners.items():
        if len({owner.split(".")[0] for owner in owners}) > 1:
            add(issues, "error", "positive_cross_intent", "training",
                "identical Positive assigned to different intents", locations=owners, text=text)
    for item_loc, text in negative_refs:
        if text in positive_owners:
            add(issues, "error", "positive_negative_collision", item_loc,
                "same text is both Positive and Negative/Hard Negative anywhere in Train",
                positive_locations=positive_owners[text], text=text)
        elif normalized(text) in positive_norms:
            add(issues, "review", "normalized_positive_negative_collision", item_loc,
                "normalized text collides with a Positive", text=text)
    return sample_refs, {"training_rows": len(rows), "samples": dict(total_samples),
                         "training_review_status_counts": dict(Counter(
                             str(row.get("review_status")) for row in rows)),
                         "empty_hard_negative_ids": sorted(empty_hard)}


def validate_corpus(rows, faq_by_id, issues):
    ids = []
    for number, row in enumerate(rows, 1):
        faq_id = row.get("faq_id")
        location = "corpus row %d" % number
        if not isinstance(faq_id, str):
            add(issues, "error", "corpus_id_type", location, "faq_id must be string")
            continue
        ids.append(faq_id)
        expected = faq_by_id.get(faq_id)
        for field in ("question", "source_group"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                add(issues, "error", "corpus_field_type", faq_id + "." + field,
                    "expected nonempty string")
        if expected is None or any(row.get(key) != expected[key] for key in ("question", "source_group")):
            add(issues, "error", "corpus_source_mismatch", faq_id,
                "corpus must preserve original CSV question and source_group", actual=row, expected=expected)
    if len(ids) != 60 or len(set(ids)) != 60 or set(ids) != set(faq_by_id):
        add(issues, "error", "corpus_coverage", "corpus",
            "expected all 60 unique original FAQ IDs; no Positive candidates",
            rows=len(rows), missing=sorted(set(faq_by_id) - set(ids)))


def validate_queries(rows, faq_by_id, splits, sample_refs, diagnostic_refs, issues,
                     near_threshold):
    exact_refs = defaultdict(list)
    for faq in faq_by_id.values():
        exact_refs[faq["question"]].append("CSV." + faq["faq_id"])
    for location, text in sample_refs:
        exact_refs[text].append(location)
    for text, query_ids in diagnostic_refs.items():
        exact_refs[text].extend(query_ids)
    normalized_refs = defaultdict(list)
    for text, refs in exact_refs.items():
        normalized_refs[normalized(text)].append((text, refs))
    query_ids, texts, counts, unreviewed = [], defaultdict(list), Counter(), []
    valid_queries = []
    for number, row in enumerate(rows, 1):
        location = "query row %d" % number
        required = ("query_id", "query", "gold_faq_id", "scenario", "split", "review_status")
        if any(not isinstance(row.get(field), str) or not row[field].strip() for field in required):
            add(issues, "error", "query_schema", location,
                "required fields must be nonempty strings", fields=list(required))
            continue
        query_id, text, gold, split = row["query_id"], row["query"], row["gold_faq_id"], row["split"]
        query_ids.append(query_id)
        texts[text].append(query_id)
        valid_queries.append(row)
        scan_text(text, query_id + ".query", issues)
        if not normalized(text):
            add(issues, "error", "empty_query", query_id, "query has no letters or numbers")
        if gold not in faq_by_id:
            add(issues, "error", "unknown_gold", query_id, "gold FAQ is absent from original 60", gold=gold)
        if split not in GROUPS:
            add(issues, "error", "unknown_eval_split", query_id,
                "split must be seen_test/unseen_validation/unseen_test", actual=split)
        else:
            scenario, source_split, _ = GROUPS[split]
            counts[(split, gold)] += 1
            if row["scenario"] != scenario or gold not in splits[source_split]:
                add(issues, "error", "eval_membership", query_id,
                    "scenario or gold ID does not match manifest", row=row)
            prefix = {"seen_test": "ST", "unseen_validation": "UV", "unseen_test": "UT"}[split]
            if not re.fullmatch(prefix + "_" + re.escape(gold) + r"_\d{2}", query_id):
                add(issues, "error", "query_id_format", query_id,
                    "expected %s_<gold_faq_id>_<two digit index>" % prefix)
        if row["review_status"] not in ("needs_review", "approved"):
            add(issues, "error", "query_review_status", query_id,
                "only needs_review or approved are accepted")
        if row["review_status"] != "approved":
            unreviewed.append(query_id)
        if text in exact_refs:
            add(issues, "error", "eval_exact_copy", query_id,
                "query exactly copies original/training/diagnostic text",
                references=exact_refs[text], text=text)
        elif normalized(text) in normalized_refs:
            add(issues, "error", "eval_normalized_copy", query_id,
                "query differs only in punctuation/case/width",
                references=normalized_refs[normalized(text)], text=text)
        matches = []
        for reference_text, refs in exact_refs.items():
            if text == reference_text:
                continue
            ratio = SequenceMatcher(None, normalized(text), normalized(reference_text),
                                    autojunk=False).ratio()
            if ratio >= near_threshold:
                matches.append({"references": refs, "text": reference_text, "ratio": ratio})
        if matches:
            add(issues, "review", "eval_near_text", query_id,
                "lexical near-copy candidates; human intent/template review required",
                query=text, matches=sorted(matches, key=lambda x: -x["ratio"])[:5])
    for query_id, count in Counter(query_ids).items():
        if count > 1:
            add(issues, "error", "duplicate_query_id", query_id, "query ID repeated")
    for text, owners in texts.items():
        if len(owners) > 1:
            add(issues, "error", "duplicate_eval_query", "evaluation",
                "query text repeated within/across eval splits", query_ids=owners, text=text)
    for split, (_, source_split, count) in GROUPS.items():
        for faq_id in sorted(splits[source_split]):
            actual = counts[(split, faq_id)]
            if actual != count:
                add(issues, "error", "eval_count_per_intent", split + "." + faq_id,
                    "wrong query count", expected=count, actual=actual)
    if len(rows) != 138:
        add(issues, "error", "eval_total", "evaluation", "expected 138 queries", actual=len(rows))
    # Compare evaluation queries to each other independently of source text.
    for i, first in enumerate(valid_queries):
        for second in valid_queries[i + 1:]:
            if first["query"] == second["query"]:
                continue
            a, b = normalized(first["query"]), normalized(second["query"])
            ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
            if a == b:
                add(issues, "error", "eval_normalized_duplicate", first["query_id"],
                    "evaluation queries normalize to identical text", other=second["query_id"])
            elif ratio >= near_threshold:
                add(issues, "review", "eval_pair_near", first["query_id"],
                    "check near-duplicate wording or cross-split template leakage",
                    other=second["query_id"], ratio=ratio,
                    texts=[first["query"], second["query"]])
    if unreviewed:
        add(issues, "review", "unreviewed_evaluation", "evaluation",
            "query candidates are not frozen; approval requires human review",
            query_ids=unreviewed)
    return {"evaluation_rows": len(rows),
            "evaluation_group_counts": dict(Counter(row.get("split") for row in valid_queries))}


def collect_validation(root, paths, include_eval=True, near_threshold=0.82):
    issues, stats = [], {}
    report = {"structural_ok": False, "semantic_approval": False,
              "note": "No model-based or semantic leakage guarantee. Review findings require human judgement.",
              "statistics": stats, "input_sha256": {}, "issues": issues}
    try:
        hashed_paths = [paths["manifest"], paths["training"], paths["diagnostics"]]
        if include_eval:
            hashed_paths.extend([paths["candidates"], paths["queries"]])
        hashed_paths.extend(root / relative for _, _, relative in CSV_SOURCES)
        # Snapshot BEFORE reading. A post-read hash alone could describe new,
        # unvalidated bytes rather than the data that produced this report.
        report["input_sha256"] = {str(path): sha256_file(path) for path in hashed_paths}
        faqs = load_source_faqs(root)
        faq_by_id = {row["faq_id"]: row for row in faqs}
        splits = manifest_sets(read_json(paths["manifest"]), issues)
        stats["source_faqs"] = len(faqs)
        diagnostic_rows = read_jsonl(paths["diagnostics"], issues)
        diagnostic_refs = defaultdict(list)
        if len(diagnostic_rows) != 15:
            add(issues, "error", "diagnostic_count", "diagnostics", "expected 15 historical queries")
        for row in diagnostic_rows:
            if not isinstance(row.get("question"), str) or not isinstance(row.get("question_id"), str):
                add(issues, "error", "diagnostic_schema", "diagnostics", "expected question/question_id strings")
            else:
                diagnostic_refs[row["question"]].append(row["question_id"])
        training = read_jsonl(paths["training"], issues)
        refs, train_stats = validate_training(training, faq_by_id, splits, diagnostic_refs,
                                             issues, near_threshold)
        stats.update(train_stats)
        queries = []
        if include_eval:
            corpus = read_jsonl(paths["candidates"], issues)
            validate_corpus(corpus, faq_by_id, issues)
            queries = read_jsonl(paths["queries"], issues)
            stats.update(validate_queries(queries, faq_by_id, splits, refs, diagnostic_refs,
                                          issues, near_threshold))
        split_of = {faq_id: name for name, ids in splits.items() for faq_id in ids}
        for ids, reason in BOUNDARIES:
            add(issues, "review", "known_semantic_boundary", "/".join(ids), reason,
                faqs=[{"faq_id": faq_id, "split": split_of.get(faq_id),
                       "question": faq_by_id.get(faq_id, {}).get("question")}
                      for faq_id in ids],
                affected_query_ids=[q.get("query_id") for q in queries if q.get("gold_faq_id") in ids])
        for path in hashed_paths:
            if sha256_file(path) != report["input_sha256"][str(path)]:
                raise DataError("input changed during validation: %s" % path)
    except (DataError, OSError, UnicodeError) as exc:
        add(issues, "error", "input_error", "inputs", str(exc))
    report["error_count"] = sum(issue["severity"] == "error" for issue in issues)
    report["review_count"] = sum(issue["severity"] == "review" for issue in issues)
    report["structural_ok"] = report["error_count"] == 0
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--scope", choices=("training", "all"), default="all")
    parser.add_argument("--near-threshold", type=float, default=0.82,
                        help="lexical warning threshold only, not semantic proof")
    parser.add_argument("--report", help="optional NEW JSON report path; relative to project root")
    args = parser.parse_args()
    if not 0 < args.near_threshold <= 1:
        parser.error("--near-threshold must be in (0,1]")
    root = args.project_root.expanduser().resolve()
    report = collect_validation(root, data_paths(args), args.scope == "all", args.near_threshold)
    try:
        if args.report:
            output = resolve_path(root, args.report)
            write_json_exclusive(output, report)
            print("Report: %s" % output)
            print("deterministic errors=%d; human review findings=%d; semantic approval=False"
                  % (report["error_count"], report["review_count"]))
            for issue in report["issues"]:
                if issue["severity"] == "error":
                    print(json.dumps(issue, ensure_ascii=False))
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
    except OSError as exc:
        parser.exit(2, "ERROR writing report: %s\n" % exc)
    return 0 if report["structural_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
