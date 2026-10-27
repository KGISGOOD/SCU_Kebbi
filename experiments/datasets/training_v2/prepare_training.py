"""Freeze checked V2 bytes and derive one family-held-out split; no ML imports.

Run from project root with Python 3.8+. Existing frozen/derived files are never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil

from experiments.a2_faq_finetuning.v2_data import convert, digest, load_bundle, read, read_rows


def write(path, value, lines=False):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        if lines:
            for row in value:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        else:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--report", type=Path, help="optional NEW JSON validation report; never overwrite")
    args = p.parse_args()
    root = args.project_root.resolve()
    base = root / "experiments/datasets/training_v2"
    frozen, splits, output = (base / s for s in ("frozen", "splits", "converted"))
    if args.check_only:
        data, m, hashes = load_bundle(root, output)
        report = {"structural_ok": True, "counts": m["counts"],
                  "checked_input_files": len(hashes), "human_approved": False,
                  "native_tokenization_checked": False,
                  "cross_split_hard_excluded": m["cross_split_hard_excluded"],
                  "split_counts": {s: {"rows": len(data[s + ".jsonl"]),
                      "intents": len({r["intent_group_id"] for r in data[s + ".jsonl"]}),
                      "families": len({r["leakage_family_id"] for r in data[s + ".jsonl"]})}
                      for s in ("train", "validation")}}
        if args.report:
            path = (root / args.report).resolve()
            if root not in path.parents:
                raise ValueError("report must be inside project")
            path.parent.mkdir(exist_ok=True, parents=True)
            write(path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if any(path.exists() for path in (frozen, splits, output)):
        raise ValueError("frozen/splits/converted already exists; use --check-only, never overwrite a frozen version")
    validation = read(base / "validation_report.json")
    if validation["errors_count"] or not validation["excel_jsonl_identical"]:
        raise ValueError("source/master/workbook validation must pass first")
    # Reject a stale successful report even if its boolean flags still say passed.
    artifact_names = {"qa_master_sha256": "qa_master.jsonl", "excel_sha256": "訓練問答總表.xlsx",
                      "source_manifest_sha256": "source_manifest.json", "excluded_sha256": "excluded_candidates.jsonl",
                      "validator_sha256": "validate.ps1"}
    for key, name in artifact_names.items():
        value = validation["artifacts"][key]
        candidate = base / name
        if digest(candidate) != value:
            raise ValueError("validation artifact changed: " + name)
    source = read(base / "source_manifest.json")
    for entry in source["sources"]:
        if digest(root / entry["path"]) != entry["sha256"]:
            raise ValueError("source changed: " + entry["path"])
    rows = read_rows(base / "qa_master.jsonl")
    families = {r["intent_group_id"]: r["leakage_family_id"] for r in rows}
    parent = {f: f for f in set(families.values())}

    def find(f):
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f

    def join(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    for r in rows:
        for kid in r["knowledge_point_ids"] + r["negative_exclusion_intent_ids"]:
            join(r["leakage_family_id"], families[kid])
    components = {}
    for r in rows:
        components.setdefault(find(r["leakage_family_id"]), []).append(r)
    keys = sorted(components)
    random.Random(args.seed).shuffle(keys)
    # Deterministic bounded subset-sum: prioritize 20% rows, then 20% intents.
    # For equal counts retain the candidate with broader current topic coverage.
    # Whole connected families only; this is not question-level stratification.
    best, chosen = None, None
    total_intents = len(families)
    states = {(0, 0): ((), frozenset(), 0)}
    for k in keys:
        group = components[k]
        size, intents = len(group), len({r["intent_group_id"] for r in group})
        topics = {r["topic"] for r in group}
        has_pairs = sum(bool(r["positive_pairs"]) for r in group)
        for (count, count_intents), (subset, covered, pair_rows) in list(states.items()):
            state = (count + size, count_intents + intents)
            candidate = (subset + (k,), covered | topics, pair_rows + has_pairs)
            old = states.get(state)
            if old is None or (bool(candidate[2]), len(candidate[1])) > (bool(old[2]), len(old[1])):
                states[state] = candidate
    for (count, count_intents), (subset, topics, pair_rows) in states.items():
        if not 0 < count < len(rows) or pair_rows < 2:
            continue
        score = (abs(count - len(rows) * .2), abs(count_intents - total_intents * .2), -len(topics))
        if best is None or score < best:
            best, chosen = score, set(subset)
    if chosen is None:
        raise ValueError("cannot create nonempty family-held-out validation")
    selected = {r["question_id"] for k in chosen for r in components[k]}
    split = {"schema_version": "training_v2_split_v1", "seed": args.seed,
             "strategy": "whole_leakage_families_plus_shared_knowledge_and_overlap_components",
             "target_validation_fraction": .2,
             "question_ids": {s: [r["question_id"] for r in rows if (r["question_id"] in selected) == (s == "validation")]
                              for s in ("train", "validation")},
             "components": {k: sorted({r["leakage_family_id"] for r in v}) for k, v in sorted(components.items())},
             "test_set_used": False,
             "limitation": "held-out knowledge families; topics may be absent from Train or Validation; not IID stratification"}
    converted, dropped = convert(rows, split)
    frozen.mkdir()
    for name in ("qa_master.jsonl", "訓練問答總表.xlsx", "validation_report.json", "source_manifest.json"):
        shutil.copyfile(base / name, frozen / name)
    fm = {"schema_version": "training_v2_frozen_v1", "version": "V2.0",
          "frozen_at": datetime.now(timezone.utc).isoformat(),
          "rows": len(rows), "knowledge_points": sum(r["qa_type"] == "basic_qa" for r in rows),
          "intent_groups": len(families), "leakage_families": len(parent),
          "positive_pairs": len({p["pair_id"] for r in rows for p in r["positive_pairs"]}),
          "directed_hard_candidates": len({p["pair_id"] for r in rows for p in r["hard_negative_pairs"]}),
          "files_sha256": {p.name: digest(p) for p in sorted(frozen.iterdir())},
          "source_sha256": {s["path"]: s["sha256"] for s in source["sources"]},
          "source_manifest_sha256": digest(base / "source_manifest.json"),
          "human_gold_certified": False, "human_training_approved": False,
          "limitations": validation["content_review"]["unresolved"],
          "meaning": "frozen source-reviewed candidate, not independent factual or human approval"}
    write(frozen / "dataset_manifest.json", fm)
    splits.mkdir()
    write(splits / "split_manifest.json", split)
    output.mkdir()
    for name, records in converted.items():
        write(output / name, records, lines=True)
    counts = {k: len(v) for k, v in converted.items()}
    inputs = [frozen / "dataset_manifest.json", splits / "split_manifest.json"]
    m = {"schema_version": "training_v2_converted_v1", "version": "V2.0",
         "frozen_dir": frozen.relative_to(root).as_posix(),
         "split_manifest": (splits / "split_manifest.json").relative_to(root).as_posix(),
         "input_sha256": {p.relative_to(root).as_posix(): digest(p) for p in inputs},
         "source_sha256": fm["source_sha256"],
         "files_sha256": {name: digest(output / name) for name in converted},
         "counts": counts, "cross_split_hard_excluded": dropped,
         "human_approved": False, "native_tokenization_checked": False,
         "generator_sha256": digest(Path(__file__)),
         "conversion_contract_sha256": digest(root / "experiments/a2_faq_finetuning/v2_data.py")}
    write(output / "dataset_manifest.json", m)
    load_bundle(root, output)
    report = {"schema_checks": "passed", "leakage_checks": "passed",
              "positive_identity_a2_1_a2_2": "same files and same deterministic schedule",
              "native_chat_template": "pending local tokenizer --check-only; not inferred from messages JSON",
              "label_masking": "existing encode_sample/collate unchanged; assistant-only and positional pad mask",
              "counts": counts, "hard_excluded": dropped,
              "split_counts": {s: {"rows": len(split["question_ids"][s]),
                                   "intents": len({r["intent_group_id"] for r in rows if r["question_id"] in set(split["question_ids"][s])}),
                                   "families": len({r["leakage_family_id"] for r in rows if r["question_id"] in set(split["question_ids"][s])})}
                               for s in ("train", "validation")},
              "model_loaded": False, "human_approved": False}
    (base / "reports").mkdir(exist_ok=True)
    write(base / "reports/preparation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
