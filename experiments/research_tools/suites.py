"""Source-linked candidate suites; reuse existing adapters and model-free readers."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from pathlib import Path

from .common import ROOT, fingerprint, locate, read_json, read_rows, sha

NAME = "experiment_suites_v1"
FIELDS = ["研究項目", "階段", "題組或對話編號", "題號", "使用者會說的話", "審核備註"]
STAGES = {"historical": "Historical", "calibration": "Calibration", "evaluation": "Evaluation"}


def load(root, revision="spoken_candidate_r3"):
    if revision not in {"spoken_candidate_r3", "spoken_candidate_r4"}:
        raise ValueError("unsupported suite revision")
    from .common import load_dataset
    base = load_dataset(root)  # original immutable dataset, all artifact/source hashes checked
    directory = root / "experiments/datasets" / NAME
    parent_directory = directory
    if revision == "spoken_candidate_r4":
        directory = directory / "revisions" / revision
    plan = read_json(directory / "suites.json")
    if (plan["schema_version"] != NAME + "_candidate" or plan["status"] != "needs_review"
            or plan["frozen"] is not False):
        raise ValueError("suite is a candidate version; approval requires an explicit reviewed view")
    source_manifest = root / "experiments/datasets/unified_research_v1/dataset_manifest.json"
    if sha(source_manifest) != plan["source_manifest_sha256"]:
        raise ValueError("original dataset manifest changed; review suite source version")
    hashes = {"original/" + k: v for k, v in base["hashes"].items()}
    hashes["suites.json"] = sha(directory / "suites.json")
    expected_files = {"questions.jsonl", "review.json"} if revision == "spoken_candidate_r4" else {"questions.jsonl"}
    if set(plan["files_sha256"]) != expected_files:
        raise ValueError("unexpected suite payload inventory")
    for name, expected in plan["files_sha256"].items():
        if sha(directory / name) != expected:
            raise ValueError(f"suite payload changed: {name}")
        hashes[name] = expected
    audit = None
    if revision == "spoken_candidate_r4":
        if set(plan.get("parent_files_sha256", {})) != {"suites.json", "questions.jsonl"}:
            raise ValueError("r4 must pin the complete r3 parent payload")
        for name, expected in plan["parent_files_sha256"].items():
            if sha(parent_directory / name) != expected:
                raise ValueError("r3 parent changed; do not silently rebase r4")
            hashes["parent/" + name] = expected
        audit = read_json(directory / "review.json")
        if audit.get("human_gold_certified") is not False or audit.get("revision") != revision:
            raise ValueError("machine-assisted review cannot certify human Gold")
        for item in audit["inputs"]:
            path = locate(root, item["path"])
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("r4 review input missing/outside project: " + item["path"])
            actual = sha(path)
            if actual != item["sha256"]:
                import hashlib
                raw = path.read_bytes()
                # Diagnostic only: normalized bytes NEVER replace the strict comparison above.
                lf_hash = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
                crlf = raw.count(b"\r\n")
                bare_lf = raw.count(b"\n") - crlf
                raise ValueError(
                    f"r4 review input changed: {item['path']}; expected={item['sha256']}; "
                    f"actual={actual}; bytes={len(raw)}; utf8_bom={raw.startswith(bytes.fromhex('efbbbf'))}; "
                    f"CRLF={crlf}; bare_LF={bare_lf}; "
                    f"LF_only_normalization_matches_expected={lf_hash == item['sha256']}. "
                    "Restore the pinned byte-exact source after preserving the GPU copy; "
                    "do not update review/manifest hashes to accept an unexplained difference.")
            hashes["review_input/" + item["path"]] = item["sha256"]
    original = {q["question_id"]: q for q in base["questions"]}
    expected_qa = [p + f"{n:02d}" for p in ("P", "S") for n in range(1, 31)]
    if (plan["qa"]["question_ids"] != expected_qa or plan["qa"]["split"] != "historical"
            or plan.get("revision") != revision):
        raise ValueError("QA must retain 60 original source IDs in the explicit named spoken suite")
    from experiments.evaluation.common import dataset
    csv_rows, _ = dataset(root, "experiments/datasets/faq_finetuning/faq_split_manifest.json")
    for row in csv_rows:
        q = original[row["faq_id"]]
        if (q["question"], q["reference_answer"]) != (row["question"], row["reference_answer"]):
            raise ValueError("suite QA differs from the original eight-architecture CSV snapshot")
    hashes["qa_split_manifest"] = sha(root / "experiments/datasets/faq_finetuning/faq_split_manifest.json")
    additions = read_rows(directory / "questions.jsonl")
    if len(additions) != 150 or len({q["question_id"] for q in additions}) != 150:
        raise ValueError("expected 60 QA overlays and 90 unique SC/SD candidates")
    qa_overlays = {q["question_id"] for q in additions if q.get("uses") == ["qa"]}
    if qa_overlays != set(expected_qa):
        raise ValueError("spoken QA overlay must contain exactly P01–P30 and S01–S30")
    questions = deepcopy(base["questions"])
    for item in additions:
        item = dict(item)
        if "existing_query_id" in item:
            candidate = original[item["existing_query_id"]]
            if "question" in item or candidate["intent_id"] != item["source_question_ids"][0]:
                raise ValueError("reused query must retain its text and original information intent")
            item["question"] = candidate["question"]
        is_qa = item["question_id"] in qa_overlays
        if ((item["question_id"] in original and not is_qa) or item["review_status"] != "needs_review"
                or item["split"] not in ({"historical"} if is_qa else {"calibration", "evaluation"})
                or (is_qa and item["source_question_ids"] != [item["question_id"]])
                or len(item["source_question_ids"]) != 1
                or item["source_question_ids"][0] not in expected_qa
                or not isinstance(item["question"], str) or not item["question"].strip()):
            raise ValueError(f"invalid derived question: {item.get('question_id')}")
        source = original[item["source_question_ids"][0]]
        q = deepcopy(source)
        q.update(item)
        q.update(primary_type="B" if item["question_id"].endswith("-P") else "A",
                 related_question_ids=item["source_question_ids"],
                 source_parent_family_id=source["family_id"],
                 provenance={"kind": "suite_spoken_candidate", "source_question_ids": item["source_question_ids"],
                             "original_question": source["question"], "new_question": item["question"],
                             "existing_query_id": item.get("existing_query_id"),
                             "modification_reason": item["modification_reason"], "uses": item["uses"],
                             "prior_query_exposure": "source_intent_and_prior_benchmarks_exposed_needs_review",
                             "suite_revision": plan["revision"],
                             "previous_candidate": item.get("previous_candidate")})
        if is_qa:
            # Replace only this named dataset's in-memory view. Original CSV/old CLI remain unchanged.
            questions = [q if row["question_id"] == q["question_id"] else row for row in questions]
        else:
            questions.append(q)
    if audit is not None:
        for q in questions:
            if q["question_id"] in audit["questions"]:
                q["source_audit"] = audit["questions"][q["question_id"]]
    index = {q["question_id"]: q for q in questions}
    groups = plan["sc"]
    if len(groups) != 30 or len({g["group_id"] for g in groups}) != 30 or len({g["family_id"] for g in groups}) != 30:
        raise ValueError("expected 30 distinct declared SC information-intent groups")
    intent_splits, used_ids = {}, set()
    for split, count in (("calibration", 10), ("evaluation", 20)):
        selected = [g for g in groups if g["split"] == split]
        if len(selected) != count:
            raise ValueError("SC 10/20 group counts differ")
        for g in selected:
            ids = g["question_ids"]
            if len(ids) != 4 or ids[0] != ids[1] or len(set(ids)) != 3:
                raise ValueError("SC group must contain original/exact-repeat/paraphrase/contrast")
            if g["relations"] != ["original", "exact_repeat", "paraphrase_reuse_candidate", "different_intent_no_reuse_candidate"]:
                raise ValueError("invalid SC relationship candidates")
            for key in set(ids):
                q = index[key]
                if q["split"] != split or q["family_id"] != g["family_id"] or key in used_ids:
                    raise ValueError("SC group membership/split overlap")
                used_ids.add(key)
                intent = q["intent_id"]
                if intent in intent_splits and intent_splits[intent] != split:
                    raise ValueError("same original information intent crosses SC/SD splits")
                intent_splits[intent] = split
            if (index[ids[0]]["intent_id"] == index[ids[3]]["intent_id"]
                    or index[ids[0]]["intent_id"] != index[ids[2]]["intent_id"]):
                raise ValueError("paraphrase/contrast source intent mismatch")
            if set(g["source_intent_ids"]) != {index[key]["intent_id"] for key in ids}:
                raise ValueError("declared group sources differ from question provenance")
        expected_sd = [key for g in selected for key in (g["question_ids"][0], g["question_ids"][3])]
        if plan["sd"][split] != expected_sd or len(expected_sd) != count * 2:
            raise ValueError("SD must use the fixed 20/40 source-linked queries")
        expected_int = [g["group_id"] for g in selected[:count // 2]]
        if plan["int"][split] != expected_int:
            raise ValueError("INT must retain its frozen 5/10 complete SC group subset")
    if used_ids != {q["question_id"] for q in additions} - qa_overlays:
        raise ValueError("unused or missing authored suite questions")
    if set(intent_splits) != set(expected_qa):
        raise ValueError("SC/SD information-intent partition must cover all 60 distinct source FAQ IDs")
    # Exact source identity is checked above; semantic independence is NOT certified.
    texts = {}
    for item in additions:
        if item["question_id"] in qa_overlays:
            continue  # Cross-research QA/SC overlap is exposed provenance, not extra SC groups.
        q = index[item["question_id"]]
        normalized = "".join(q["question"].split())
        if normalized in texts:
            raise ValueError("duplicate derived queries; do not fabricate independent groups")
        texts[normalized] = q["split"]
    turns = []
    conversations = plan["ti"]
    if len(conversations) != 20 or len({c["source_conversation_id"] for c in conversations}) != 20:
        raise ValueError("TI must preserve 20 unique complete source conversations")
    families = {"calibration": set(), "evaluation": set()}
    for split, count in (("calibration", 8), ("evaluation", 12)):
        if sum(c["split"] == split for c in conversations) != count:
            raise ValueError("TI requires 8/12 complete conversations")
    for c in conversations:
        source = [t for t in base["turns"] if t["conversation_id"] == c["source_conversation_id"]]
        if len(source) != 4 or [t["turn_index"] for t in source] != [1, 2, 3, 4] or len(c["queries"]) != 4:
            raise ValueError("TI requires four ordered turns per source conversation")
        families[c["split"]].add(source[0]["family_id"])
        for field in ("topic_ids", "shift_labels", "required_history_turn_indices", "source_question_ids"):
            if len(c.get(field, [])) != 4:
                raise ValueError(f"{c['conversation_id']}: missing four-turn annotation {field}")
        for i, old in enumerate(source):
            t = deepcopy(old)
            text = c["queries"][i]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty spoken TI query")
            topic = c["topic_ids"][i]
            expected_label = "initial" if i == 0 else (
                "same_topic" if topic == c["topic_ids"][i - 1] else "shift")
            required = c["required_history_turn_indices"][i]
            source_ids = c["source_question_ids"][i]
            if (not isinstance(topic, str) or not topic.strip()
                    or c["shift_labels"][i] != expected_label
                    or not isinstance(required, list) or len(set(required)) != len(required)
                    or any(type(j) is not int or not 1 <= j <= i for j in required)
                    or not source_ids or not set(source_ids) <= set(original)):
                raise ValueError(f"{c['conversation_id']} turn {i + 1}: inconsistent topic/history/source")
            t.update(question=text, split=c["split"], review_status="needs_review",
                     suite_conversation_id=c["conversation_id"], original_question=old["question"],
                     original_split=old["split"], source_turn_id=old["turn_id"],
                     modification_reason=c["modification_reason"],
                     topic_evidence=c["topic_annotation_basis"],
                     original_topic_id=old["topic_id"],
                     original_topic_label=old["same_topic_or_shift"],
                     original_source_question_ids=old["source_question_ids"],
                     topic_id=topic, same_topic_or_shift=expected_label,
                     required_history_turn_ids=[source[j - 1]["turn_id"] for j in required],
                     removable_history_turn_ids=[],  # Do not certify deletions from candidate labels.
                     source_question_ids=source_ids, prior_exposure=c["prior_exposure"],
                     suite_revision=plan["revision"])
            if (c["prior_exposure"]["source_split"] != old["split"]
                    or c["prior_exposure"]["moved_from_calibration"] !=
                    (old["split"] == "calibration" and c["split"] == "evaluation")):
                raise ValueError("TI prior exposure metadata contradicts original source")
            for history in t["history"]:
                if history["role"] == "user":
                    j = next(j for j, r in enumerate(source) if r["turn_id"] == history["turn_id"])
                    history["content"] = c["queries"][j]
            if audit is not None:
                t["source_audit"] = audit["questions"][c["conversation_id"] + f"-T{i + 1}"]
            turns.append(t)
    if families["calibration"] & families["evaluation"]:
        raise ValueError("TI conversation family crosses Calibration/Evaluation")
    queries_by_split = {s: {t["question"].strip() for t in turns if t["split"] == s} for s in families}
    if queries_by_split["calibration"] & queries_by_split["evaluation"]:
        raise ValueError("same TI utterance crosses splits")
    if audit is not None:
        expected_audits = {q["question_id"] for q in additions} | {
            c["conversation_id"] + f"-T{i}" for c in conversations for i in range(1, 5)}
        if (set(audit["questions"]) != expected_audits
                or len(audit["decisions"]) != 380
                or len(audit["answer_sources"]) != 225
                or any(r.get("review_status") != "needs_review" for r in audit["questions"].values())
                or any(r.get("human_gold_certified") is not False for r in audit["answer_sources"])):
            raise ValueError("r4 integrated review coverage/status differs from candidate contract")
    counts = {"qa": 60, "sc_calibration_groups": 10, "sc_evaluation_groups": 20,
              "sd_calibration": 20, "sd_evaluation": 40, "ti_calibration_conversations": 8,
              "ti_evaluation_conversations": 12, "int_calibration_groups": 5, "int_evaluation_groups": 10}
    if plan["counts"] != counts:
        raise ValueError("suite declared counts changed")
    result = {**base, "identity": fingerprint(hashes), "hashes": hashes, "questions": questions,
              "turns": turns, "suite": plan, "integrated_review": audit,
              "dataset_name": NAME if revision == "spoken_candidate_r3" else NAME + "_r4"}
    return execution(result, "cold")


def execution(data, cache_state="cold", groups_limit=None):
    """Materialize ID references only; never load a model or share answer caches."""
    if cache_state not in {"cold", "prefilled"}:
        raise ValueError("invalid cache state")
    plan = data["suite"]
    index = {q["question_id"]: q for q in data["questions"]}
    workloads, fills = [], []
    for target in ("sc", "int"):
        for split in ("calibration", "evaluation"):
            selected = [g for g in plan["sc"] if g["split"] == split and
                        (target == "sc" or g["group_id"] in plan["int"][split])]
            if groups_limit is not None:
                if groups_limit < 1:
                    raise ValueError("functional group prefix must be positive")
                selected = selected[:groups_limit]
            workload_id = target.upper() + "-" + split
            for g in selected:
                if cache_state == "prefilled":
                    fills.append({"workload_id": workload_id, "question_id": g["question_ids"][0],
                                  "source_split": split, "group_id": g["group_id"],
                                  "cache_state_policy": "empty_then_explicit_initial_fill"})
                for position, key in enumerate(g["question_ids"], 1):
                    workloads.append({"workload_id": workload_id, "question_id": key,
                                      "source_split": split, "group_id": g["group_id"], "group_position": position,
                                      "reference_question_id": g["question_ids"][0] if position > 1 else None,
                                      "relationship_candidate": g["relations"][position - 1],
                                      "expected_reuse_review_status": "needs_review", "review_status": "needs_review",
                                      "cache_state_policy": "empty_then_explicit_initial_fill" if cache_state == "prefilled" else "empty_at_workload_start"})
    return {**data, "workloads": workloads, "fills": fills}


def prepare(args, data):
    """Select the frozen research-specific contract before the existing adapter."""
    if args.ids or args.types or args.workload_id or args.initial_fill_ids:
        raise ValueError("suite membership is frozen; use functional --limit or --groups, not arbitrary selectors")
    if args.groups is not None and (args.purpose != "functional" or args.limit is not None):
        raise ValueError("--groups is a functional-only complete-group prefix; do not combine --limit")
    if args.suite_cache_state != "cold" and args.target != "int":
        raise ValueError("prefilled suite execution currently uses only phase-aware INT runner")
    if args.target == "qa":
        if args.split != "historical" or args.groups is not None:
            raise ValueError("QA spoken suite retains Historical exposure; no Calibration split")
        args.ids = ",".join(data["suite"]["qa"]["question_ids"])
    elif args.split not in {"calibration", "evaluation"}:
        raise ValueError("this suite target requires Calibration/Evaluation")
    if args.target in {"sd", "prompt_lookup"}:
        if args.groups is not None:
            raise ValueError("SD uses --limit for functional prefix, not groups")
        args.ids = ",".join(data["suite"]["sd"][args.split])
    if args.target in {"sc", "sc_threshold", "int"}:
        if args.limit is not None:
            raise ValueError("cache suites require --groups so four-query groups remain complete")
        args.workload_id = ("INT" if args.target == "int" else "SC") + "-" + args.split
        count = len(data["suite"]["int"][args.split]) if args.target == "int" else sum(g["split"] == args.split for g in data["suite"]["sc"])
        if args.groups is not None and args.groups > count:
            raise ValueError("functional group prefix exceeds target suite")
        data = execution(data, args.suite_cache_state, args.groups)
    if args.target == "ti" and args.groups is not None:
        if not 1 <= args.groups <= sum(c["split"] == args.split for c in data["suite"]["ti"]):
            raise ValueError("functional conversation prefix exceeds TI suite")
        args.limit = args.groups  # existing TI adapter treats limit as whole conversations
    return data


def review_rows(data):
    index = {q["question_id"]: q for q in data["questions"]}
    plan = data["suite"]
    rows = []
    def add(target, split, group, key, query, note):
        rows.append(dict(zip(FIELDS, [target, STAGES[split], group, key, query, note + "；待人工審核"])))
    for key in plan["qa"]["question_ids"]:
        add("QA", "historical", "QA60", key, index[key]["question"], "口語候選；Suite " + plan["revision"] + "；來源" + key + "原題保留；研究用途為QA評估，Historical表示先前曝光，非新盲測")
    for g in plan["sc"]:
        for i, key in enumerate(g["question_ids"], 1):
            note = ["原始資訊需求的口語候選", "完全重複，答案品質及重用仍需確認",
                    "自然改寫，需確認可安全重用", "近主題但資訊需求不同，不直接重用的候選"][i - 1]
            add("SC", g["split"], g["group_id"], g["group_id"] + f"-Q{i}", index[key]["question"], note + "；" + g["semantic_boundary_review"])
    for split in ("calibration", "evaluation"):
        for i, key in enumerate(plan["sd"][split], 1):
            add("SD", split, "SD-" + split, f"SD-{split}-{i:02d}", index[key]["question"], "與SC共用來源；三種解碼模式使用同一完整Prompt")
    for c in plan["ti"]:
        for i, query in enumerate(c["queries"], 1):
            source = next(t for t in data["turns"] if t["source_turn_id"] == c["source_conversation_id"] + f"_T{i}")
            add("TI", c["split"], c["conversation_id"], c["conversation_id"] + f"-T{i}", query,
                "主題候選：" + source["topic_id"] + "；切換候選：" + source["same_topic_or_shift"]
                + "；必要歷史候選：" + ",".join(source["required_history_turn_ids"])
                + ("；原Calibration已曝光，不是新盲測" if source["prior_exposure"]["moved_from_calibration"] else ""))
    groups = {g["group_id"]: g for g in plan["sc"]}
    for split in ("calibration", "evaluation"):
        for group_id in plan["int"][split]:
            for i, key in enumerate(groups[group_id]["question_ids"], 1):
                add("INT", split, group_id, "INT-" + group_id + f"-Q{i}", index[key]["question"], "固定引用SC同組同序；填充成本另計")
    if len(rows) != 380:
        raise ValueError("review must contain exactly 380 research-use rows, not 380 independent questions")
    targets = {name: i for i, name in enumerate(("QA", "SC", "SD", "TI", "INT"))}
    stages = {name: i for i, name in enumerate(("Historical", "Calibration", "Evaluation"))}
    return sorted(rows, key=lambda r: (targets[r[FIELDS[0]]], stages[r[FIELDS[1]]], r[FIELDS[2]], r[FIELDS[3]]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "export-review", "export-integrated"))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--revision", choices=("spoken_candidate_r3", "spoken_candidate_r4"), default="spoken_candidate_r3")
    parser.add_argument("--output", default="experiments/datasets/研究題目審核表.csv")
    args = parser.parse_args()
    root = args.project_root.resolve()
    data = load(root, args.revision)
    rows = review_rows(data)
    if args.action == "export-integrated":
        if args.revision != "spoken_candidate_r4":
            raise ValueError("integrated workbook is available only for explicit r4")
        path = root / "experiments/datasets/研究題庫整合審核_r4.xlsx"
        from .suite_review import export
        digest = export(data, path)
        if load(root, args.revision)["identity"] != data["identity"]:
            raise ValueError("r4 changed during integrated export; do not certify the workbook")
        print({"path": str(path), "sha256": digest, "human_gold_certified": False})
        return
    if args.action == "export-review":
        path = locate(root, args.output)
        if path.parent != root / "experiments/datasets" or path.suffix.lower() != ".csv" or path.exists():
            raise ValueError("review export must be a NEW CSV directly in experiments/datasets")
        from .review import write_csv
        if load(root, args.revision)["identity"] != data["identity"]:
            raise ValueError("suite changed before export")
        write_csv(path, rows)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            if list(csv.DictReader(stream)) != rows:
                raise ValueError("review CSV round-trip mismatch")
        if load(root, args.revision)["identity"] != data["identity"]:
            raise ValueError("suite changed during export")
        print(f"{path}: 380 rows, 6 columns, UTF-8 BOM; candidates only")
    else:
        print({"status": "structurally_checked_semantics_unverified", "counts": data["suite"]["counts"],
               "review_rows": len(rows), "human_gold_certified": False})


if __name__ == "__main__":
    main()
