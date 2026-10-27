"""Reproducible CPU/stdlib acceptance of conversion, schedule and mask contracts."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from experiments.a2_faq_finetuning.training_data import batch_documents, prepare_training, schedule_hash
from experiments.a2_faq_finetuning.v2_data import a2_inputs, load_bundle, make_schedule, read, read_rows, validate_partition
from experiments.a3_faq_sft.common import validate_data


def main():
    root = Path.cwd().resolve()
    base = root / "experiments/datasets/training_v2"
    directory = base / "converted"
    files, manifest, hashes = load_bundle(root, directory)
    a3, _ = validate_data(root, directory)
    assert a3["train"] == files["train.jsonl"] and a3["validation"] == files["validation.jsonl"]
    config = read(root / "experiments/a2_faq_finetuning/training_v2_config.json")
    rows, diagnostics, effective, _ = a2_inputs(root, directory, config)
    by_id, hard, diagnostic, excluded, _ = prepare_training(rows, diagnostics, effective)
    schedule = make_schedule(by_id, 8, 3, 42, diagnostic, excluded)
    comparison = make_schedule(by_id, 8, 3, 42, diagnostic, excluded)
    assert schedule_hash(schedule) == schedule_hash(comparison)
    expected = {p["pair_id"] for p in files["a2_train_pairs.jsonl"]}
    for epoch in (1, 2, 3):
        seen = [s["pair_id"] for b in schedule if b["epoch"] == epoch for s in b["samples"]]
        assert len(seen) == len(expected) and set(seen) == expected
    exposure = {"a2_1": 0, "a2_2": 0}
    for batch in schedule:
        for mode in exposure:
            docs = batch_documents(batch["samples"], by_id, hard, mode, diagnostic, excluded)
            for i, sample in enumerate(batch["samples"]):
                assert docs["allowed_mask"][i][docs["targets"][i]]
                assert sum(docs["allowed_mask"][i]) >= 2
                for j, document in enumerate(docs["document_ids"]):
                    if frozenset((sample["faq_id"], document)) in excluded:
                        assert not docs["allowed_mask"][i][j]
            if mode == "a2_1":
                assert docs["extra_document_count"] == 0 and docs["hard_visible_exposures"] == 0
            exposure[mode] += docs["hard_visible_exposures"]
    val_rows = files["a2_validation.jsonl"]
    val_ids = {r["faq_id"] for r in val_rows}
    val_exclusions = {tuple(sorted((r["intent_group_id"], other)))
                      for r in files["a2_validation_questions.jsonl"]
                      for other in r["negative_exclusion_intent_ids"] if other in val_ids}
    val_config = dict(config, false_negative_pairs=[list(p) for p in sorted(val_exclusions)])
    val_by_id, val_hard, val_diagnostic, val_excluded, _ = prepare_training(val_rows, [], val_config)
    val_schedule = make_schedule(val_by_id, 8, 1, 42, val_diagnostic, val_excluded)
    assert {s["pair_id"] for b in val_schedule for s in b["samples"]} == {p["pair_id"] for p in files["a2_validation_pairs.jsonl"]}
    for batch in val_schedule:
        for mode in exposure:
            batch_documents(batch["samples"], val_by_id, val_hard, mode, val_diagnostic, val_excluded)
    # Independent source-span overlap audit across the chosen split.
    master = read_rows(base / "frozen/qa_master.jsonl")
    split = read(base / "splits/split_manifest.json")
    owner = validate_partition(master, split)
    overlaps = []
    for i, first in enumerate(master):
        for second in master[i + 1:]:
            if owner[first["question_id"]] == owner[second["question_id"]]:
                continue
            for a in first["sources"]:
                for b in second["sources"]:
                    if (a["path"] == b["path"] and max(a["start_utf16"], b["start_utf16"]) <
                            min(a["start_utf16"] + a["length_utf16"], b["start_utf16"] + b["length_utf16"])):
                        overlaps.append((first["question_id"], second["question_id"]))
    assert not overlaps, "overlapping evidence crosses split"
    parsed = []
    for parent in ("experiments/a2_faq_finetuning", "experiments/a3_faq_sft", "experiments/datasets/training_v2"):
        for path in sorted((root / parent).glob("*.py")):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            parsed.append(path.relative_to(root).as_posix())
    print(json.dumps({"stdlib_data_checks": "passed", "a2_a3_loaders": "passed",
                      "syntax_files": len(parsed), "epochs_checked": 3,
                      "optimizer_steps_planned_each_a2": len(schedule),
                      "positive_exposures_each_a2": sum(len(b["samples"]) for b in schedule),
                      "schedule_sha256_each_a2": schedule_hash(schedule),
                      "hard_visible_exposures": exposure,
                      "validation_batches_checked_each_a2": len(val_schedule),
                      "cross_split_overlapping_source_spans": len(overlaps),
                      "chat_template_tokenization": "pending actual GPU-computer local tokenizer",
                      "large_models_loaded": False, "human_approval": False}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
