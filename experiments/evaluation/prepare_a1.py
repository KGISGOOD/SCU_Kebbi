"""Explicit, non-overwriting A1 60-FAQ preparation. GPU execution only by user."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import (ROOT, DataError, atomic_json, dataset, differences,
                     read_json, read_rows, read_run, resolve, sha)
from .model_resources import cached_model

CANONICAL = "experiments/datasets/faq/faq_canonical_60_v2.jsonl"
INDEX = "experiments/vector_stores/faq_bge_m3_60_v2"
MANIFEST = "experiments/datasets/faq_finetuning/faq_split_manifest.json"


def index_completion_report(canonical, index_dir, version=None):
    """One byte-level receipt contract for builder, preflight and read-only audit.

    Never reserialize JSON, normalize line endings, unpickle, load FAISS or repair
    a receipt. Missing/malformed files raise with their concrete path.
    """
    version_path = index_dir / "version.json"
    if version is None:
        version = read_json(version_path)
    expected = {"schema_version": "a1_60_preparation_v1", "status": "completed", "faq_count": 60,
                "canonical_sha256": version.get("canonical_sha256"),
                "index_sha256": version.get("index_sha256"),
                "docstore_sha256": version.get("docstore_sha256")}
    actual = {"schema_version": version.get("schema_version"), "status": version.get("status"),
              "faq_count": version.get("faq_count"), "canonical_sha256": sha(canonical),
              "index_sha256": sha(index_dir / "index.faiss"),
              "docstore_sha256": sha(index_dir / "index.pkl")}
    changes = differences(expected, actual, "a1_index_completion")
    paths = {"version": str(version_path), "canonical": str(canonical),
             "index": str(index_dir / "index.faiss"), "docstore": str(index_dir / "index.pkl")}
    return {"status": "failed" if changes else "passed", "paths": paths,
            "expected": expected, "actual": actual, "differences": changes,
            "hash_algorithm": "SHA-256 of raw file bytes; lowercase hex; no text normalization"}


def validate_canonical(root, canonical, manifest):
    source, hashes = dataset(root, manifest)
    rows = read_rows(canonical)
    expected = {row["faq_id"]: row for row in source}
    ids = [row.get("faq_id") for row in rows]
    if (len(rows) != 60 or any(not isinstance(i, str) for i in ids)
            or len(set(ids)) != 60 or set(ids) != set(expected)):
        raise DataError("A1 canonical must contain exactly 60 unique original P/S IDs")
    for row in rows:
        original = expected[row["faq_id"]]
        if row.get("question") != original["question"] or row.get("answer") != original["reference_answer"]:
            raise DataError("A1 %s Q/A differs from CSV" % row["faq_id"])
    return rows, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--manifest", default=MANIFEST)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("canonical", help="CSV -> NEW 60-row A1 JSONL; no model load")
    p.add_argument("--output", default=CANONICAL)
    p = sub.add_parser("index", help="explicitly build a NEW A1 index using existing FAQMatcher")
    p.add_argument("--canonical", default=CANONICAL)
    p.add_argument("--output-dir", default=INDEX)
    p.add_argument("--bge-snapshot", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p = sub.add_parser("check", help="check A1 and A2 IDs/Q/A against CSV; no model load")
    p.add_argument("--canonical", default=CANONICAL)
    p.add_argument("--a2-candidates", default="experiments/datasets/faq_finetuning/faq_retrieval_candidates.jsonl")
    p = sub.add_parser("audit-index", help="read-only completion/hash/path/CSV audit; no FAISS/model load or writes")
    context = p.add_mutually_exclusive_group()
    context.add_argument("--config", help="inspect paths from this evaluation config")
    context.add_argument("--run-dir", type=Path, help="inspect exact frozen paths from a failed Smoke run")
    p.add_argument("--canonical", help="explicit path override, shown in the report")
    p.add_argument("--index-dir", help="explicit path override, shown in the report")
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    try:
        if args.command == "audit-index":
            if args.run_dir:
                frozen = read_run(resolve(root, args.run_dir))
                if Path(frozen["project_root"]).resolve() != root:
                    raise DataError("frozen run project_root differs from --project-root; audit on its original GPU project")
                config = frozen["config"]
                context_path = str(resolve(root, args.run_dir) / "run.json")
            else:
                config_path = resolve(root, args.config) if args.config else Path(__file__).with_name("config.json")
                config = read_json(config_path)
                context_path = str(config_path)
            canonical = resolve(root, args.canonical or config["a1_canonical"])
            index_dir = resolve(root, args.index_dir or config["a1_index"])
            report = index_completion_report(canonical, index_dir)
            report["configuration_source"] = context_path
            report["path_overrides"] = {"canonical": args.canonical, "index_dir": args.index_dir}
            if args.run_dir:
                recorded = frozen["resources"]["a1_index"]
                report["frozen_a1_resources"] = {name: frozen["resources"][name]
                                                 for name in ("a1_index", "a1_canonical")}
                if recorded.get("value"):
                    current_files = {name: sha(index_dir / name)
                                     for name in recorded["value"]["files_sha256"]}
                    report["changes_since_run_creation"] = differences(
                        recorded["value"]["files_sha256"], current_files, "frozen_a1_index.files_sha256")
                    if report["changes_since_run_creation"]:
                        report["status"] = "failed"
            try:
                rows, _ = validate_canonical(root, canonical, config["manifest"])
                report["csv_validation"] = {"status": "passed", "faq_count": len(rows)}
            except Exception as exc:
                report["csv_validation"] = {"status": "failed", "error": "%s: %s" % (type(exc).__name__, exc)}
                report["status"] = "failed"
            print(json.dumps(report, ensure_ascii=True, indent=2))
            return 0 if report["status"] == "passed" else 2
        if args.command == "canonical":
            output = resolve(root, args.output)
            metadata = output.with_name(output.name + ".metadata.json")
            if output.suffix != ".jsonl" or output.exists() or metadata.exists():
                raise DataError("choose a NEW .jsonl path; never overwrite: %s" % output)
            source, hashes = dataset(root, args.manifest)
            rows = [{"faq_id": row["faq_id"], "question": row["question"],
                     "answer": row["reference_answer"], "source_file": Path(row["source"]).name,
                     "source_type": "parents_faq" if row["faq_id"].startswith("P") else "highschool_faq",
                     "review_status": "needs_review"} for row in source]
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            validate_canonical(root, output, args.manifest)
            atomic_json(metadata, {"schema_version": "a1_canonical_60_v2", "faq_count": 60,
                                   "source_sha256": hashes, "canonical_sha256": sha(output),
                                   "encoding": "utf-8", "construction": "original CSV Q/A verbatim; existing IDs"})
            print("A1 canonical: 60 unique original FAQs; output=%s" % output)
            return 0
        canonical = resolve(root, args.canonical)
        canonical_sha = sha(canonical)
        rows, sources = validate_canonical(root, canonical, args.manifest)
        if args.command == "check":
            from experiments.a2_faq_finetuning.faq_answering import load_answering_data
            corpus, answers = load_answering_data(root, resolve(root, args.a2_candidates))
            expected = {row["faq_id"]: row for row in rows}
            if set(answers) != set(expected) or len(corpus) != 60:
                raise DataError("A1/A2 ID sets differ")
            for faq_id, answer in answers.items():
                if (answer["question"] != expected[faq_id]["question"]
                        or answer["answer"] != expected[faq_id]["answer"]):
                    raise DataError("A1/A2 Q/A mismatch: %s" % faq_id)
            print("A1=60, A2 candidates=60, A2 answers=60; unique IDs and verbatim Q/A match CSV")
            return 0
        output = resolve(root, args.output_dir)
        allowed = (root / "experiments/vector_stores").resolve()
        if allowed not in output.parents or output.exists():
            raise DataError("index must be a NEW directory under experiments/vector_stores: %s" % output)
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        model = cached_model(root, "BAAI/bge-m3", snapshot=args.bge_snapshot)
        # Reserve once. Never resume/rebuild an existing directory, even if a prior
        # build failed. Existing matcher.save() writes only into this NEW directory.
        output.mkdir(parents=True, exist_ok=False)
        from experiments.baselines.faq_semantic_matching.config import settings
        from experiments.baselines.faq_semantic_matching.matcher import FAQMatcher
        settings.embedding_model = model["path"]
        settings.embedding_normalize = True
        settings.use_hnsw = False
        matcher = FAQMatcher(faq_dataset_path=str(canonical), faq_vector_store_dir=str(output))
        matcher.embedder.model.to(args.device).eval()
        matcher.embedder.device = args.device
        if matcher.faqs != rows:
            raise DataError("FAQMatcher did not load the validated 60-row canonical exactly")
        matcher.build_index()  # Original A1 encode -> normalized IndexFlatIP -> save.
        if not matcher.load_index():
            raise DataError("new A1 index cannot be reloaded")
        store = matcher.vector_store
        if (store.index.ntotal != 60 or len(store.docstore) != 60
                or store.index.d != matcher.embedder.embedding_dim
                or store.index_to_docstore_id != {i: i for i in range(60)}):
            raise DataError("new A1 index count/dimension/mapping mismatch")
        for doc, row in zip(store.docstore, rows):
            if (doc.page_content != row["question"] or doc.metadata.get("faq_id") != row["faq_id"]
                    or doc.metadata.get("answer") != row["answer"]):
                raise DataError("new A1 index Q/A metadata mismatch: %s" % row["faq_id"])
        _, after_sources = validate_canonical(root, canonical, args.manifest)
        if canonical_sha != sha(canonical) or sources != after_sources or model != cached_model(
                root, "BAAI/bge-m3", snapshot=args.bge_snapshot):
            raise DataError("canonical/CSV/model changed during index preparation")
        atomic_json(output / "version.json", {
            "schema_version": "a1_60_preparation_v1", "status": "completed", "faq_count": 60,
            "canonical_sha256": canonical_sha, "source_sha256": sources,
            "embedding_model": model, "embedding_max_length": 512, "normalization": "L2",
            "index_type": "IndexFlatIP", "index_sha256": sha(output / "index.faiss"),
            "docstore_sha256": sha(output / "index.pkl")})
        completion = index_completion_report(canonical, output)
        if completion["status"] != "passed":
            raise DataError("A1 index receipt failed self-check: " + json.dumps(completion, ensure_ascii=True))
        print("A1 NEW index built/reloaded: 60 FAQs; output=%s" % output)
        return 0
    except Exception as exc:
        parser.exit(2, "ERROR [%s]: %s\n" % (type(exc).__name__, exc))


if __name__ == "__main__":
    raise SystemExit(main())
