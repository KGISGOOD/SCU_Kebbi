"""Explicit, non-overwriting A1 60-FAQ preparation. GPU execution only by user."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .common import (ROOT, DataError, atomic_json, dataset, differences,
                     read_json, read_rows, read_run, resolve, sha)
from .model_resources import cached_model

CANONICAL = "experiments/datasets/faq/faq_canonical_60_v2.jsonl"
INDEX = "experiments/vector_stores/faq_bge_m3_60_v2"
MANIFEST = "experiments/datasets/faq_finetuning/faq_split_manifest.json"


def canonical_bytes_report(path, expected_hash):
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return {"bytes": len(raw), "utf8_bom": raw.startswith(b"\xef\xbb\xbf"),
            "crlf_count": raw.count(b"\r\n"), "bare_lf_count": raw.count(b"\n") - raw.count(b"\r\n"),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "lf_bytes": len(lf), "lf_sha256": hashlib.sha256(lf).hexdigest(),
            "lf_matches_receipt": hashlib.sha256(lf).hexdigest() == expected_hash}


def inspect_docstore(canonical, index_dir, manifest, root):
    """Explicitly opted-in, hash-checked local pickle inspection; no FAISS/model.

    The receipt must be trusted locally. A hash in an untrusted receipt alone is
    not a security guarantee. Refuse globals other than the original Document.
    """
    import pickle
    completion = index_completion_report(canonical, index_dir)
    if any(row["field"] != "a1_index_completion.canonical_sha256" for row in completion["differences"]):
        raise DataError("docstore inspection blocked by non-canonical receipt mismatch: "
                        + json.dumps(completion, ensure_ascii=True))
    rows, sources = validate_canonical(root, canonical, manifest)
    expected = {row["faq_id"]: row for row in rows}
    pickle_path = index_dir / "index.pkl"

    class LocalDocumentUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("langchain_core.documents.base", "Document"):
                from langchain_core.documents import Document
                return Document
            raise DataError("unsupported pickle global; do not weaken inspection whitelist: %s.%s" % (module, name))

    with pickle_path.open("rb") as handle:
        saved = LocalDocumentUnpickler(handle).load()
        if handle.read(1):
            raise DataError("unexpected bytes after docstore pickle")
    if not isinstance(saved, dict):
        raise DataError("docstore payload must be a dictionary")
    docs, mapping = saved.get("docstore"), saved.get("index_to_docstore_id")
    if not isinstance(docs, list) or len(docs) != 60 or mapping != {i: i for i in range(60)}:
        raise DataError("docstore must have 60 documents and complete identity mapping")
    changes, seen = [], set()
    for position, doc in enumerate(docs):
        metadata = getattr(doc, "metadata", None)
        faq_id = metadata.get("faq_id") if isinstance(metadata, dict) else None
        if not isinstance(faq_id, str) or faq_id not in expected or faq_id in seen:
            raise DataError("docstore[%d] duplicate/unknown FAQ ID: %r" % (position, faq_id))
        seen.add(faq_id)
        row = expected[faq_id]
        changes += differences({"question": row["question"], "answer": row["answer"]},
                               {"question": getattr(doc, "page_content", None), "answer": metadata.get("answer")},
                               "docstore." + faq_id)
        if metadata.get("question") != row["question"]:
            changes += differences(row["question"], metadata.get("question"), "docstore." + faq_id + ".metadata.question")
    after = index_completion_report(canonical, index_dir)
    if after != completion:
        raise DataError("index/canonical/receipt changed during docstore inspection")
    for relative, expected_hash in sources.items():
        if sha(root / relative) != expected_hash:
            raise DataError("CSV/manifest changed during docstore inspection: %s" % relative)
    return {"status": "failed" if changes else "passed", "faq_count": len(docs),
            "id_set_matches": seen == set(expected), "differences": changes,
            "scope": "exact ID/question/answer and mapping; not FAISS tensor validation"}


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
    p.add_argument("--inspect-docstore", action="store_true", help="compare exact stored Q/A; requires trusted local pickle opt-in")
    p.add_argument("--trust-local-docstore", action="store_true")
    p = sub.add_parser("restore-canonical", help="write NEW LF-only byte restoration and NEW config; never edit receipt/index")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--config")
    source.add_argument("--from-run", type=Path, help="inherit original frozen evaluation settings")
    p.add_argument("--canonical", help="override source canonical")
    p.add_argument("--index-dir", help="override existing trusted index directory")
    p.add_argument("--output", required=True, help="NEW .jsonl path")
    p.add_argument("--output-config", required=True, help="NEW evaluation config path")
    p.add_argument("--trust-local-docstore", action="store_true", required=True)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    try:
        if args.command == "restore-canonical":
            source_run = read_run(resolve(root, args.from_run)) if args.from_run else None
            if source_run and Path(source_run["project_root"]).resolve() != root:
                raise DataError("restore on the original run's project root")
            config = (dict(source_run["config"]) if source_run else read_json(
                resolve(root, args.config or str(Path(__file__).with_name("config.json")))))
            canonical = resolve(root, args.canonical or config["a1_canonical"])
            index_dir = resolve(root, args.index_dir or config["a1_index"])
            output, config_output = resolve(root, args.output), resolve(root, args.output_config)
            receipt_output = output.with_name(output.name + ".restoration.json")
            if (output.suffix != ".jsonl" or config_output.suffix != ".json"
                    or any(p.exists() for p in (output, config_output, receipt_output))
                    or len({output, config_output, receipt_output}) != 3):
                raise DataError("restoration requires distinct NEW JSONL/config/receipt outputs; no overwrites")
            # Keep the new corpus in the project, and allow only a separate config
            # file outside the existing index. Never publish over any input.
            output.relative_to(root / "experiments/datasets/faq")
            config_output.relative_to(root / "experiments/evaluation")
            watched = [canonical, index_dir / "index.faiss", index_dir / "index.pkl", index_dir / "version.json"]
            before = {str(p): sha(p) for p in watched}
            version = read_json(index_dir / "version.json")
            content = inspect_docstore(canonical, index_dir, config["manifest"], root)
            if content["status"] != "passed":
                raise DataError("canonical differs from stored Q/A; restore blocked: " + json.dumps(content, ensure_ascii=True))
            restored = canonical.read_bytes().replace(b"\r\n", b"\n")
            target_hash = hashlib.sha256(restored).hexdigest()
            if target_hash != version.get("canonical_sha256"):
                raise DataError("LF-only restoration does not reproduce original receipt hash; use original backup or a NEW index")
            if before != {str(p): sha(p) for p in watched}:
                raise DataError("input changed during restoration checks; no outputs written")
            config["a1_canonical"] = output.relative_to(root).as_posix()
            config["a1_index"] = index_dir.relative_to(root).as_posix()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as handle:
                handle.write(restored)
                handle.flush()
                os.fsync(handle.fileno())
            completion = index_completion_report(output, index_dir)
            if completion["status"] != "passed":
                raise DataError("new restoration failed receipt check; retained for investigation; no config published")
            if sha(output) != target_hash or before != {str(p): sha(p) for p in watched}:
                raise DataError("original input or restored output changed; no restoration receipt/config published")
            atomic_json(receipt_output, {"schema_version": "a1_lf_restoration_v1", "status": "completed",
                                        "transformation": "CRLF bytes to LF only; exact original byte hash restored",
                                        "original_inputs_sha256": before, "output_sha256": sha(output),
                                        "configuration_source_run_id": source_run["run_id"] if source_run else None,
                                        "docstore_validation": content})
            atomic_json(config_output, config)
            print(json.dumps({"status": "restored_to_new_files", "canonical": str(output),
                              "config": str(config_output), "sha256": target_hash,
                              "original_index_and_receipt_unchanged": True}, ensure_ascii=True, indent=2))
            return 0
        if args.command == "audit-index":
            if args.run_dir:
                run = resolve(root, args.run_dir)
                if not (run / "run.json").is_file():
                    partial = sorted(str(p) for p in (run / "preflight").glob("*_dataset_failed.json"))
                    raise DataError("cannot audit frozen run: " + json.dumps({
                        "requested_run_root": str(run), "directory_exists": run.is_dir(),
                        "required_manifest": str(run / "run.json"), "dataset_failure_reports": partial,
                        "reason": "run.json missing; wrong/subdirectory path, incomplete copy, or creation failed before manifest",
                        "action": "locate actual run root containing run.json; otherwise audit explicit canonical/index paths",
                        "frozen_config_verified": False, "fallback_performed": False}, ensure_ascii=True))
                frozen = read_run(run)
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
            report["canonical_representation"] = canonical_bytes_report(canonical, report["expected"]["canonical_sha256"])
            if args.inspect_docstore:
                if not args.trust_local_docstore:
                    raise DataError("--inspect-docstore requires --trust-local-docstore for your own hash-checked local pickle")
                report["docstore_validation"] = inspect_docstore(canonical, index_dir, config["manifest"], root)
                if report["docstore_validation"]["status"] != "passed":
                    report["status"] = "failed"
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
