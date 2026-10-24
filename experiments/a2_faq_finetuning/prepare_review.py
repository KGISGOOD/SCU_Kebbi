"""Copy the existing 42 Train candidates for human approval; never auto-approve."""
import argparse
import json

from .common import DATA_DIR, default_root, read_jsonl, resolve_path, sha256_file, write_json_exclusive
from .validate_a2_data import collect_validation
from experiments.research_tools.training_approval import template


def validate_manifest(root, path, training):
    from .common import DataError, read_json
    value = read_json(path)
    if (value.get("schema_version") != "a2_reviewed_data_v1"
            or value.get("train_sha256") != sha256_file(training)):
        raise DataError("A2 derived data manifest does not match actual Train bytes")
    if not value.get("source_sha256"):
        raise DataError("A2 original source hashes missing")
    for name, digest in value["source_sha256"].items():
        if sha256_file(resolve_path(root, name)) != digest:
            raise DataError("A2 derived source changed: " + name)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(default_root()))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--training-review", help="optionally verify explicit human approval, not model weights")
    args = parser.parse_args()
    from pathlib import Path
    root = Path(args.project_root).resolve()
    output = resolve_path(root, args.output_dir)
    paths = {"training": root / DATA_DIR / "faq_training_candidates.jsonl",
             "manifest": root / DATA_DIR / "faq_split_manifest.json",
             "diagnostics": root / "experiments/datasets/scu_faq_matching_test_v1.jsonl"}
    if not args.check_only:
        report = collect_validation(root, paths, include_eval=False)
        if not report["structural_ok"]:
            raise ValueError(json.dumps(report["issues"], ensure_ascii=False))
        issues = []
        rows = read_jsonl(paths["training"], issues)
        if issues or len(rows) != 42:
            raise ValueError("expected exactly 42 valid existing candidates")
        output.mkdir(parents=True, exist_ok=False)
        (output / "train.jsonl").write_bytes(paths["training"].read_bytes())
        sources = {Path(p).relative_to(root).as_posix(): digest for p, digest in report["input_sha256"].items()}
        write_json_exclusive(output / "dataset_manifest.json", {
            "schema_version": "a2_reviewed_data_v1", "review_status": "needs_review",
            "source_sha256": sources, "train_sha256": sha256_file(output / "train.jsonl"),
            "faq_ids": [r["faq_id"] for r in rows],
            "review_notes": ["Keep all 126 positives; do not fill the 12 empty Hard Negative arrays.",
                             "P13/P14 are excluded by training_config; retain the false-negative mask.",
                             "Review P01/P08, P02/P15, P08/P09/P10/P12, S09/S14/S17, S19/S20/S21.",
                             "S15/S16 difficulty is unverified. HN coverage is uneven; no model ranking certification.",
                             "FAQ60 contains 42 seen intents; no claim of wholly unseen evaluation."]})
        write_json_exclusive(output / "training_review.template.json", template(
            "a2_training_review_v1", output / "train.jsonl", output / "dataset_manifest.json", [r["faq_id"] for r in rows]))
    manifest = validate_manifest(root, output / "dataset_manifest.json", output / "train.jsonl")
    derived_paths = {**paths, "training": output / "train.jsonl"}
    report = collect_validation(root, derived_paths, include_eval=False)
    if not report["structural_ok"]:
        raise ValueError(json.dumps(report["issues"], ensure_ascii=False))
    if args.training_review:
        from experiments.research_tools.training_approval import verify
        verify(resolve_path(root, args.training_review), "a2_training_review_v1", output / "train.jsonl",
               output / "dataset_manifest.json", manifest["faq_ids"])
    print(json.dumps({"derived_hashes_valid": True, "human_receipt_verified": bool(args.training_review),
                      "path": str(output)}))


if __name__ == "__main__":
    main()
