"""Evaluate one original/fine-tuned dense BGE-M3 model on the same A2 corpus.

No training. Defaults to unseen_validation to avoid using final tests for tuning.
Use --split all only for the locked final comparison.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import uuid

from .common import (DataError, GROUPS, add_data_arguments, data_paths, read_json, read_jsonl,
                     resolve_path, sha256_file, write_json_exclusive, write_jsonl_exclusive,
                     assert_input_hashes, read_checkpoint_provenance)
from .validate_a2_data import collect_validation


def local_model_identity(path: Path):
    if not path.exists():
        return None
    if path.is_file():
        return {"path": str(path), "files": [{"name": path.name, "sha256": sha256_file(path)}]}
    files = [{"name": str(item.relative_to(path)), "sha256": sha256_file(item)}
             for item in sorted(path.rglob("*")) if item.is_file()]
    if not files:
        raise DataError("local model directory is empty: %s" % path)
    import hashlib
    fingerprint = hashlib.sha256(
        json.dumps(files, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"path": str(path), "file_manifest_sha256": fingerprint, "files": files}


def package_versions():
    result = {}
    for name in ("torch", "numpy", "sentence-transformers", "transformers", "huggingface-hub"):
        try:
            result[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            result[name] = None
    return result


def mean_metrics(rows):
    count = len(rows)
    if not count:
        return None
    return {
        "top1_accuracy": sum(row["hit_at_1"] for row in rows) / count,
        "recall_at_5": sum(row["hit_at_5"] for row in rows) / count,
        "mrr": sum(row["reciprocal_rank"] for row in rows) / count,
        "mean_top1_cosine": sum(row["top5_scores"][0] for row in rows) / count,
        "mean_gold_cosine": sum(row["gold_score"] for row in rows) / count,
    }


def main(diagnostic=False) -> int:
    parser = argparse.ArgumentParser(description=(
        "Evaluate only the separate provisional Train-intent hard diagnostic, never formal tests."
        if diagnostic else __doc__))
    add_data_arguments(parser)
    parser.add_argument("--model", default="BAAI/bge-m3",
                        help="Hugging Face model ID or local path relative to --project-root")
    parser.add_argument("--model-tag", choices=("A1", "A2-1", "A2-2"), default="A1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    if diagnostic:
        parser.add_argument("--hard-diagnostic",
                            default="experiments/datasets/faq_finetuning/faq_hard_diagnostic.jsonl")
        parser.add_argument("--check-only", action="store_true",
                            help="print diagnostic validation only, without importing/loading a model")
        parser.set_defaults(split="hard_diagnostic")
    else:
        parser.add_argument("--split", choices=("all",) + tuple(GROUPS), default="unseen_validation")
    parser.add_argument("--offline", action="store_true",
                        help="use only cached/local model files; do not access Hugging Face")
    parser.add_argument("--allow-unreviewed", action="store_true",
                        help="explicit diagnostic run only; outputs marked provisional")
    parser.add_argument("--output-dir", help="NEW result directory; never overwrite existing results")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 1 or not 0 <= args.seed < 2**32:
        parser.error("positive batch size/max length and seed in [0, 2**32) required")
    root = args.project_root.expanduser().resolve()
    paths = data_paths(args)
    try:
        groups = {"hard_diagnostic": None} if diagnostic else GROUPS
        if diagnostic:
            from .diagnostic_data import collect_diagnostic_validation
            diagnostic_path = resolve_path(root, args.hard_diagnostic)
            validation = collect_diagnostic_validation(root, paths, diagnostic_path)
        else:
            validation = collect_validation(root, paths, include_eval=True)
        if diagnostic and args.check_only:
            print(json.dumps(validation, ensure_ascii=False, indent=2))
            return 0 if validation["structural_ok"] else 1
        if not validation["structural_ok"]:
            errors = [issue for issue in validation["issues"] if issue["severity"] == "error"]
            raise DataError("data validation failed before model loading:\n" +
                            json.dumps(errors, ensure_ascii=False, indent=2))
        input_hashes = validation["input_sha256"]
        read_issues = []
        corpus = read_jsonl(paths["candidates"], read_issues)
        all_queries = read_jsonl(diagnostic_path if diagnostic else paths["queries"], read_issues)
        if diagnostic:
            all_queries = [dict(row, scenario="diagnostic", split="hard_diagnostic")
                           for row in all_queries]
        if read_issues:
            raise DataError(json.dumps(read_issues, ensure_ascii=False))
        assert_input_hashes(input_hashes)
        corpus = sorted(corpus, key=lambda row: row["faq_id"])
        selected_groups = list(groups) if args.split == "all" else [args.split]
        queries = [row for row in all_queries if row["split"] in selected_groups]
        if not queries:
            raise DataError("no queries selected")
        pending = [row["query_id"] for row in queries if row["review_status"] != "approved"]
        if pending and not args.allow_unreviewed:
            raise DataError("selected queries require human review; use --allow-unreviewed only "
                            "for a provisional diagnostic run: %s" % ", ".join(pending))
        if args.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # Set before importing torch/creating a CUDA context.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import numpy as np
        import torch
        from experiments.shared.embeddings import BGEM3Embedder

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        if args.device == "cuda" and not torch.cuda.is_available():
            raise DataError("--device cuda requested but CUDA is unavailable; use --device cpu/auto")
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        local_path = resolve_path(root, args.model)
        identity = local_model_identity(local_path)
        # Model tags are a data contract, not a cosmetic output directory name.
        model_provenance = read_checkpoint_provenance(
            local_path, require_a2=args.model_tag != "A1")
        if model_provenance is not None:
            expected_tag = {"a2_1": "A2-1", "a2_2": "A2-2"}[model_provenance["mode"]]
            if args.model_tag != expected_tag:
                raise DataError("model tag %s contradicts checkpoint mode %s; use --model-tag %s"
                                % (args.model_tag, model_provenance["mode"], expected_tag))
            if model_provenance["provisional"] and not args.allow_unreviewed:
                raise DataError("model was trained provisionally; --allow-unreviewed is required "
                                "for diagnostic evaluation and results remain provisional")
        model_name = str(local_path) if identity is not None else args.model
        embedder = BGEM3Embedder(model_name=model_name, device=device,
                                batch_size=args.batch_size, normalize_embeddings=True)
        embedder.model.max_seq_length = args.max_length
        embedder.model.float()
        embedder.model.eval()

        def encode_checked(texts, label):
            with torch.no_grad():
                values = np.asarray(embedder.encode(texts), dtype=np.float32)
            if (values.ndim != 2 or values.shape[0] != len(texts) or
                    not np.isfinite(values).all()):
                raise DataError("%s embeddings have invalid shape or non-finite values" % label)
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            if np.any(norms <= 0):
                raise DataError("%s contains zero embeddings" % label)
            return values / norms

        candidate_vectors = encode_checked([row["question"] for row in corpus], "corpus")
        query_vectors = encode_checked([row["query"] for row in queries], "queries")
        if candidate_vectors.shape[1] != query_vectors.shape[1]:
            raise DataError("query/corpus embedding dimensions differ")
        cosine = query_vectors @ candidate_vectors.T
        if not np.isfinite(cosine).all():
            raise DataError("cosine matrix has non-finite values")
        ids = [row["faq_id"] for row in corpus]
        results = []
        for query, scores in zip(queries, cosine):
            # Full 60-way exact normalized inner product, stable FAQ ID tie-break.
            order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))
            ranking = [ids[index] for index in order]
            gold_rank = ranking.index(query["gold_faq_id"]) + 1
            top = order[:5]
            results.append({
                "query_id": query["query_id"], "query": query["query"],
                "gold_faq_id": query["gold_faq_id"], "scenario": query["scenario"],
                "split": query["split"], "query_review_status": query["review_status"],
                "top5_faq_ids": [ids[index] for index in top],
                "top5_scores": [float(scores[index]) for index in top],
                "hit_at_1": gold_rank == 1, "hit_at_5": gold_rank <= 5,
                "gold_rank": gold_rank, "gold_score": float(scores[ids.index(query["gold_faq_id"]) ]),
                "reciprocal_rank": 1.0 / gold_rank,
                "full_ranking_faq_ids": ranking,
                "full_ranking_scores": [float(scores[index]) for index in order],
            })
            if diagnostic:
                results[-1].update({key: query[key] for key in (
                    "difficulty_type", "confusable_faq_ids", "rationale")})
                results[-1].update({"diagnostic": True, "provisional": True})
        summary = {}
        for group in groups:
            rows = [row for row in results if row["split"] == group]
            by_faq = defaultdict(list)
            for row in rows:
                by_faq[row["gold_faq_id"]].append(row)
            per_faq = {faq_id: mean_metrics(values) for faq_id, values in sorted(by_faq.items())}
            macro = None
            if per_faq:
                macro = {key: sum(value[key] for value in per_faq.values()) / len(per_faq)
                         for key in ("top1_accuracy", "recall_at_5", "mrr")}
            summary[group] = {"evaluated": group in selected_groups, "query_count": len(rows),
                              "faq_count": len(by_faq), "metrics": mean_metrics(rows),
                              "macro_faq_metrics": macro, "per_faq": per_faq}
            if diagnostic:
                from .diagnostic_data import DIFFICULTY_TYPES
                summary[group].update({"provisional": True,
                                       "purpose": "development_diagnostic_not_final_test"})
                summary[group]["by_difficulty_type"] = {
                    kind: {"query_count": sum(row["difficulty_type"] == kind for row in rows),
                           "metrics": mean_metrics([row for row in rows
                                                    if row["difficulty_type"] == kind])}
                    for kind in DIFFICULTY_TYPES}
        # Detect input edits during a run; do not save misleading results.
        for name, digest in input_hashes.items():
            if sha256_file(Path(name)) != digest:
                raise DataError("input changed during evaluation: %s" % name)
        if identity is not None and local_model_identity(local_path) != identity:
            raise DataError("local model files changed during evaluation")
        revisions = sorted({
            str(getattr(getattr(getattr(module, "auto_model", None), "config", None), "_commit_hash"))
            for module in embedder.model._modules.values()
            if getattr(getattr(getattr(module, "auto_model", None), "config", None), "_commit_hash", None)
        })
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        result_base = ("experiments/results/a2_hard_diagnostic" if diagnostic
                       else "experiments/results/a2_faq_retrieval")
        default_output = Path(result_base) / args.model_tag / (
            timestamp + "_" + uuid.uuid4().hex[:8])
        output = resolve_path(root, args.output_dir or default_output)
        if output.exists():
            raise DataError("result directory already exists; choose a NEW --output-dir: %s" % output)
        if diagnostic and resolve_path(root, "experiments/results/a2_faq_retrieval") in (output, *output.parents):
            raise DataError("diagnostic output must be separate from the formal retrieval result tree")
        metadata = {
            "schema_version": "a2_retrieval_run_v1", "model_tag": args.model_tag,
            "model_argument": args.model, "effective_model": model_name,
            "local_model_identity": identity, "resolved_hf_revisions": revisions,
            "remote_revision_unresolved": identity is None and not revisions,
            "device_requested": args.device, "device_actual": device,
            "batch_size": args.batch_size, "max_length": args.max_length,
            "seed": args.seed, "precision": "float32", "normalize_embeddings": True,
            "similarity": "exact normalized inner product (cosine)",
            "tie_break": "FAQ ID ascending", "candidate_count": len(corpus),
            "ranking_depth": len(corpus), "top_k_saved": 5, "mrr": "full ranking, not MRR@5",
            "query_count": len(queries), "selected_splits": selected_groups,
            "provisional": (diagnostic or bool(pending) or args.allow_unreviewed or
                            bool(model_provenance and model_provenance["provisional"])),
            "model_training_provenance": model_provenance,
            "unreviewed_query_ids": pending,
            "semantic_risks_require_human_review": True,
            "validation_review_findings": validation["review_count"],
            "input_sha256": input_hashes, "versions": package_versions(),
            "python": platform.python_version(), "platform": platform.platform(),
            "utc_timestamp": timestamp, "offline": args.offline,
            "torch_deterministic_algorithms": True,
            **({"evaluation_purpose": "provisional_development_diagnostic_not_final_test",
                "diagnostic": True, "diagnostic_path": str(diagnostic_path),
                "diagnostic_gold_scope": "train_only",
                "designed_after_a1_validation_aggregate_observed": True,
                "diagnostic_validator_sha256": sha256_file(Path(__file__).with_name("diagnostic_data.py")),
                "diagnostic_entry_sha256": sha256_file(Path(__file__).with_name("evaluate_hard_diagnostic.py"))}
               if diagnostic else {}),
            "code_sha256": {str(path): sha256_file(path) for path in (
                Path(__file__).resolve(), Path(__file__).with_name("common.py").resolve(),
                Path(__file__).with_name("validate_a2_data.py").resolve(),
                Path(sys.modules[BGEM3Embedder.__module__].__file__).resolve())},
        }
        # Exclusive directory/files: never overwrite a prior run, even under a fixed path.
        output.mkdir(parents=True, exist_ok=False)
        write_jsonl_exclusive(output / "per_query.jsonl", results)
        write_json_exclusive(output / "metrics.json", summary)
        write_json_exclusive(output / "run_metadata.json", metadata)
        print(json.dumps({"output": str(output), "provisional": metadata["provisional"],
                          "groups": summary}, ensure_ascii=False, indent=2))
        return 0
    except (DataError, OSError, ImportError, RuntimeError, ValueError) as exc:
        parser.exit(2, "ERROR: %s\n" % exc)


if __name__ == "__main__":
    raise SystemExit(main())
