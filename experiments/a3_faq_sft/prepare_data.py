"""Prepare or validate FAQ SFT data without importing torch/tokenizers/models."""
import argparse
import csv
import json
from pathlib import Path

from .common import (DATA, MANIFEST, ROOT, SYSTEM, resolve, sha, snapshot,
                     validate_data, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--data-dir", default=DATA)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--answer-revisions", help="42 Train-only candidate revisions JSONL; never changes original snapshots")
    parser.add_argument("--training-review", help="optionally verify a genuine human receipt, without loading models")
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    directory = resolve(root, args.data_dir)
    try:
        if not args.check_only:
            manifest_path = resolve(root, args.manifest)
            rows, hashes = snapshot(root, manifest_path)
            revised, revision_hashes = {}, {}
            revision_path = None
            if args.answer_revisions:
                from .revised_data import revisions
                revision_path = resolve(root, args.answer_revisions)
                revised, revision_hashes = revisions(root, revision_path, rows)
                rows = [revised.get(row["faq_id"], row) for row in rows]
            directory.mkdir(parents=True, exist_ok=False)
            counts = {}
            for split in ("train", "validation", "test"):
                selected = [row for row in rows if row["split"] == split]
                counts[split] = len(selected)
                with (directory / (split + ".jsonl")).open("x", encoding="utf-8") as handle:
                    for row in selected:
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            write_json(directory / "dataset_manifest.json", {
                "schema_version": "a3_sft_data_v2" if revised else "a3_sft_data_v1", "review_status": "needs_review",
                "answer_revisions": revision_path.relative_to(root).as_posix() if revision_path else None,
                "revision_inputs_sha256": revision_hashes,
                "intent_manifest": manifest_path.relative_to(root).as_posix(),
                "system_prompt": SYSTEM, "counts": counts, "source_sha256": hashes,
                "files_sha256": {split + ".jsonl": sha(directory / (split + ".jsonl"))
                                 for split in counts},
                "construction": "source-linked Train answer revisions; held-out rows unchanged" if revised else "original Q/A only; no A2 positives or evaluation queries",
            })
            if revised:
                from experiments.research_tools.training_approval import template
                write_json(directory / "training_review.template.json", template(
                    "a3_training_review_v1", directory / "train.jsonl", directory / "dataset_manifest.json", sorted(revised)))
        data, _ = validate_data(root, directory)
        if args.training_review:
            from experiments.research_tools.training_approval import verify
            verify(resolve(root, args.training_review), "a3_training_review_v1", directory / "train.jsonl",
                   directory / "dataset_manifest.json", [r["faq_id"] for r in data["train"]])
        print(json.dumps({"structural_ok": True, "counts": {k: len(v) for k, v in data.items()},
                          "human_receipt_verified": bool(args.training_review),
                          "semantic_review_required": not bool(args.training_review), "tokenization_checked": False}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        parser.exit(2, "ERROR: %s\n" % exc)


if __name__ == "__main__":
    raise SystemExit(main())
