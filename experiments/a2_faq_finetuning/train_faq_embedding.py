"""Train A2-1/A2-2 with masked cosine MNRL; no evaluation queries are read.

Targets the repository's SentenceTransformers 2.2.2 <= version < 2.5 API.
Only the user may execute this program on the training computer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import uuid

from .common import (DATA_DIR, DataError, default_root, read_json, read_jsonl,
                     resolve_path, sha256_file, write_json_exclusive, write_jsonl_exclusive,
                     assert_input_hashes, checkpoint_files, read_checkpoint_provenance)
from .evaluate_faq_retrieval import local_model_identity, package_versions
from .training_data import (batch_documents, make_schedule, prepare_training, schedule_hash)
from .validate_a2_data import collect_validation


def load_arguments():
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--project-root", type=Path, default=default_root())
    preliminary.add_argument("--config", default=str(Path(__file__).with_name("training_config.json")))
    first, _ = preliminary.parse_known_args()
    config_path = resolve_path(first.project_root.expanduser().resolve(), first.config)
    config = read_json(config_path)
    config_digest = sha256_file(config_path)
    if read_json(config_path) != config or sha256_file(config_path) != config_digest:
        raise DataError("training config changed while being read")
    required = {"schema_version", "model", "batch_size", "epochs", "learning_rate", "seed",
                "max_length", "scale", "weight_decay", "max_grad_norm", "save_every_epochs",
                "false_negative_pairs"}
    if set(config) != required or config.get("schema_version") != "a2_training_v1":
        raise DataError("training config has missing/unknown fields or wrong schema_version")
    if not isinstance(config["model"], str) or not isinstance(config["false_negative_pairs"], list):
        raise DataError("config.model must be string; false_negative_pairs must be a list")
    parser = argparse.ArgumentParser(description=__doc__, parents=[preliminary])
    parser.add_argument("--mode", required=True, choices=("a2_1", "a2_2"))
    parser.add_argument("--model", default=config["model"])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    for key in ("batch_size", "epochs", "seed", "max_length", "save_every_epochs"):
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=config[key])
    for key in ("learning_rate", "scale", "weight_decay", "max_grad_norm"):
        parser.add_argument("--" + key.replace("_", "-"), type=float, default=config[key])
    parser.add_argument("--training", default=str(DATA_DIR / "faq_training_candidates.jsonl"))
    parser.add_argument("--manifest", default=str(DATA_DIR / "faq_split_manifest.json"))
    parser.add_argument("--diagnostics", default="experiments/datasets/scu_faq_matching_test_v1.jsonl")
    parser.add_argument("--output-dir", help="must not already exist; relative to project root")
    parser.add_argument("--allow-unreviewed", action="store_true",
                        help="explicit provisional research/debug run, never a formal experiment")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="enable underlying Transformer checkpointing or fail explicitly")
    args = parser.parse_args()
    import math
    for key in ("batch_size", "epochs", "seed", "max_length", "save_every_epochs"):
        if type(getattr(args, key)) is not int:
            parser.error("%s must be an integer" % key)
    for key in ("learning_rate", "scale", "weight_decay", "max_grad_norm"):
        value = getattr(args, key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            parser.error("%s must be a finite number" % key)
    if not (2 <= args.batch_size <= 42 and args.epochs >= 1 and args.max_length >= 1
            and args.save_every_epochs >= 1 and 0 <= args.seed < 2**32
            and args.learning_rate > 0 and args.scale > 0 and args.weight_decay >= 0
            and args.max_grad_norm > 0):
        parser.error("invalid hyperparameters: batch 2..42, positive epochs/length/LR/scale/clip/save interval required")
    args.config = str(config_path)
    return parser, args, config, config_digest


def main() -> int:
    output = None
    steps = 0
    started = time.monotonic()
    try:
        parser, args, config, config_digest = load_arguments()
        root = args.project_root.expanduser().resolve()
        # This path dictionary deliberately has no eval-query or candidate-corpus path.
        paths = {key: resolve_path(root, getattr(args, key))
                 for key in ("training", "manifest", "diagnostics")}
        validation = collect_validation(root, paths, include_eval=False)
        if not validation["structural_ok"]:
            raise DataError("training data validation failed before model loading:\n" +
                            json.dumps([i for i in validation["issues"] if i["severity"] == "error"],
                                       ensure_ascii=False, indent=2))
        read_issues = []
        rows = read_jsonl(paths["training"], read_issues)
        diagnostics = read_jsonl(paths["diagnostics"], read_issues)
        if read_issues:
            raise DataError(json.dumps(read_issues, ensure_ascii=False))
        assert_input_hashes(validation["input_sha256"])
        pending = [row["faq_id"] for row in rows if row["review_status"] != "approved"]
        if pending and not args.allow_unreviewed:
            raise DataError("formal training is blocked: review_status is not approved for %s. "
                            "Do not auto-approve. --allow-unreviewed is provisional research only."
                            % ", ".join(pending))
        by_id, hard, diagnostic_ids, excluded, excluded_hard = prepare_training(rows, diagnostics, config)
        schedule = make_schedule(by_id, args.batch_size, args.epochs, args.seed,
                                 diagnostic_ids, excluded)
        expected_pairs = 126 * args.epochs
        if sum(len(batch["samples"]) for batch in schedule) != expected_pairs:
            raise DataError("schedule coverage mismatch; no training will start")
        # Check API ranges without importing a model or selecting a different installed API.
        from packaging.version import Version
        versions = package_versions()
        for name, lower, upper in (("sentence-transformers", "2.2.2", "2.5"),
                                    ("torch", "2.0", "2.4"),
                                    ("transformers", "4.43", "4.48")):
            value = versions.get(name)
            if value is None or not Version(lower) <= Version(value) < Version(upper):
                raise DataError("%s=%r outside repository supported range [%s,%s); "
                                "this program does not install or upgrade packages"
                                % (name, value, lower, upper))
        input_hashes = dict(validation["input_sha256"])
        input_hashes[args.config] = config_digest
        for path, digest in input_hashes.items():
            if sha256_file(Path(path)) != digest:
                raise DataError("input changed during validation: %s" % path)
        if args.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import numpy as np
        import torch
        from experiments.shared.embeddings import BGEM3Embedder
        from .training_loss import MaskedMultipleNegativesRankingLoss

        if args.device == "cuda" and not torch.cuda.is_available():
            raise DataError("CUDA requested but unavailable; choose cpu or auto")
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        model_path = resolve_path(root, args.model)
        identity = local_model_identity(model_path)
        if identity is not None and read_checkpoint_provenance(model_path) is not None:
            raise DataError("initial model is an A2-trained output, not the common original checkpoint")
        effective_model = str(model_path) if identity is not None else args.model
        wrapper = BGEM3Embedder(model_name=effective_model, device=device,
                                batch_size=args.batch_size, normalize_embeddings=True)
        model = wrapper.model
        if identity is not None and local_model_identity(model_path) != identity:
            raise DataError("initial checkpoint files changed during model loading")
        model.max_seq_length = args.max_length
        model.to(device).float()
        if args.gradient_checkpointing:
            enabled = False
            for module in model._modules.values():
                transformer = getattr(module, "auto_model", None)
                if transformer is not None:
                    method = getattr(transformer, "gradient_checkpointing_enable", None)
                    if not callable(method):
                        raise DataError("underlying Transformer does not support gradient checkpointing")
                    method()
                    if hasattr(transformer.config, "use_cache"):
                        transformer.config.use_cache = False
                    enabled = True
            if not enabled:
                raise DataError("no underlying Transformer found for gradient checkpointing")
        revisions = sorted({
            str(getattr(getattr(getattr(module, "auto_model", None), "config", None), "_commit_hash"))
            for module in model._modules.values()
            if getattr(getattr(getattr(module, "auto_model", None), "config", None), "_commit_hash", None)
        })
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        default_output = Path("experiments/models/a2_faq_finetuning") / args.mode / (
            timestamp + "_" + uuid.uuid4().hex[:8])
        requested_output = resolve_path(root, args.output_dir or default_output)
        if requested_output.exists():
            raise DataError("output directory already exists; choose a NEW output: %s" % requested_output)
        if identity is not None and model_path.is_dir() and model_path in requested_output.parents:
            raise DataError("output must not be placed inside the frozen initial model directory")
        requested_output.mkdir(parents=True, exist_ok=False)
        output = requested_output
        provisional = bool(pending) or args.allow_unreviewed
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        parameter_count = sum(parameter.numel() for parameter in parameters)
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate,
                                     betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=args.weight_decay)
        objective = MaskedMultipleNegativesRankingLoss(scale=args.scale)
        provenance = {
            "schema_version": "a2_training_provenance_v1", "mode": args.mode,
            "provisional": provisional, "unreviewed_faq_ids": pending,
            "input_sha256": input_hashes, "positive_pair_schedule_sha256": schedule_hash(schedule),
            "initial_model": effective_model, "initial_local_identity": identity,
            "initial_hf_revisions": revisions,
            "remote_revision_unresolved": identity is None and not revisions,
            "hard_negatives_model_rank_verified": False,
            "loss": "MaskedMultipleNegativesRankingLoss (cosine InfoNCE)",
            "explicit_stored_negative_arrays_used": False,
            "excluded_hard_candidates": excluded_hard,
        }
        code_paths = [Path(__file__).resolve(), Path(__file__).with_name("training_data.py"),
                      Path(__file__).with_name("training_loss.py"), Path(__file__).with_name("common.py"),
                      Path(__file__).with_name("validate_a2_data.py"),
                      Path(__file__).with_name("evaluate_faq_retrieval.py"),
                      Path(sys.modules[BGEM3Embedder.__module__].__file__).resolve()]
        run_config = {
            **provenance, "arguments": {key: str(value) if isinstance(value, Path) else value
                                       for key, value in vars(args).items()},
            "config_contents": config, "versions": versions, "python": platform.python_version(),
            "platform": platform.platform(), "device_actual": device, "precision": "float32",
            "optimizer": "torch.optim.AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
            "scheduler": "constant learning rate", "weight_decay_scope": "all trainable parameters",
            "planned_optimizer_steps": len(schedule), "planned_positive_pair_exposures": expected_pairs,
            "trainable_parameters": parameter_count,
            "fp32_adam_parameter_gradient_moment_bytes_lower_bound": parameter_count * 16,
            "diagnostic_document_ids_masked_as_negatives": sorted(diagnostic_ids),
            "hard_candidate_count_before_exclusions": sum(len(row["hard_negative"]) for row in rows),
            "hard_candidate_count_after_exclusions": sum(len(value) for value in hard.values()),
            "empty_effective_hard_ids": sorted(faq_id for faq_id, value in hard.items() if not value),
            "data_validation_statistics": validation["statistics"],
            "code_sha256": {str(path): sha256_file(path) for path in code_paths},
            "utc_started": timestamp, "evaluation_during_training": False,
        }
        write_json_exclusive(output / "run_config.json", run_config)
        write_json_exclusive(output / "data_validation.json", validation)
        write_jsonl_exclusive(output / "positive_pair_schedule.jsonl", schedule)

        def forward_texts(texts):
            # encode() is inference-only and detaches gradients; train through forward().
            features = model.tokenize(texts)
            features = {key: value.to(device) if torch.is_tensor(value) else value
                        for key, value in features.items()}
            return model(features)["sentence_embedding"]

        def save_model(path, epoch):
            if identity is not None and local_model_identity(model_path) != identity:
                raise DataError("initial checkpoint files changed before checkpoint export")
            for name, digest in input_hashes.items():
                if sha256_file(Path(name)) != digest:
                    raise DataError("input changed during training: %s" % name)
            path.mkdir(parents=True, exist_ok=False)
            # Written first: interrupted exports cannot fall back to a raw model.
            write_json_exclusive(path / "a2_checkpoint_pending.json",
                                 {"schema_version": "a2_checkpoint_pending_v1"})
            model.save(str(path), create_model_card=False)
            write_json_exclusive(path / "a2_training_provenance.json",
                                 {**provenance, "completed_epoch": epoch, "optimizer_steps": steps})
            if not (path / "modules.json").is_file():
                raise DataError("SentenceTransformer export is missing modules.json")
            write_json_exclusive(path / "a2_checkpoint_complete.json", {
                "schema_version": "a2_checkpoint_complete_v1",
                "files_sha256": checkpoint_files(path),
            })
            read_checkpoint_provenance(path, require_a2=True)

        totals = {"positive_pair_exposures": 0, "base_negative_exposures": 0,
                  "hard_visible_exposures": 0, "hard_new_exposures": 0,
                  "effective_negative_exposures": 0, "extra_document_encodings": 0}
        weighted_loss = 0.0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with (output / "training_log.jsonl").open("x", encoding="utf-8") as log:
            for index, batch in enumerate(schedule):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                # Keep query/base-document dropout streams mode-independent at each update.
                torch.manual_seed((args.seed + index) % 2**32)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all((args.seed + index) % 2**32)
                samples = batch["samples"]
                documents = batch_documents(samples, by_id, hard, args.mode, diagnostic_ids, excluded)
                query_embeddings = forward_texts([sample["query"] for sample in samples])
                base_embeddings = forward_texts([sample["positive"] for sample in samples])
                extras = documents["document_texts"][len(samples):]
                document_embeddings = (torch.cat([base_embeddings, forward_texts(extras)], dim=0)
                                       if extras else base_embeddings)
                loss = objective(query_embeddings, document_embeddings,
                                 documents["allowed_mask"], documents["targets"])
                if not torch.isfinite(loss):
                    raise DataError("non-finite training loss at step %d" % (steps + 1))
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, args.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                loss_value = float(loss.detach().cpu())
                weighted_loss += loss_value * len(samples)
                totals["positive_pair_exposures"] += len(samples)
                for key in ("base_negative_exposures", "hard_visible_exposures",
                            "hard_new_exposures", "effective_negative_exposures"):
                    totals[key] += documents[key]
                totals["extra_document_encodings"] += documents["extra_document_count"]
                record = {
                    "optimizer_step": steps, "epoch": batch["epoch"], "round": batch["round"],
                    "pair_ids": [sample["pair_id"] for sample in samples],
                    "batch_size_actual": len(samples), "loss": loss_value,
                    "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    **{key: documents[key] for key in (
                        "base_negative_exposures", "hard_visible_exposures",
                        "hard_new_exposures", "effective_negative_exposures", "extra_document_count")},
                    "document_ids": documents["document_ids"], "allowed_mask": documents["allowed_mask"],
                }
                log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                epoch_finished = index + 1 == len(schedule) or schedule[index + 1]["epoch"] != batch["epoch"]
                if (epoch_finished and batch["epoch"] < args.epochs
                        and batch["epoch"] % args.save_every_epochs == 0):
                    save_model(output / "checkpoints" / ("epoch_%03d" % batch["epoch"]), batch["epoch"])
                    model.train()
        if steps != len(schedule) or totals["positive_pair_exposures"] != expected_pairs:
            raise DataError("completed training coverage differs from planned schedule")
        if identity is not None and local_model_identity(model_path) != identity:
            raise DataError("initial checkpoint files changed during training")
        save_model(output / "final_model", args.epochs)
        write_json_exclusive(output / "training_summary.json", {
            "status": "completed", "provisional": provisional, "optimizer_steps": steps,
            **totals, "mean_query_weighted_training_loss": weighted_loss / expected_pairs,
            "elapsed_seconds": time.monotonic() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "final_model": str(output / "final_model"), "validation_selection_performed": False,
        })
        print(json.dumps({"output": str(output), "final_model": str(output / "final_model"),
                          "optimizer_steps": steps, "provisional": provisional}, ensure_ascii=False))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if output is not None:
            try:
                write_json_exclusive(output / "failure.json",
                                     {"status": "failed", "exception": type(exc).__name__,
                                      "message": str(exc), "completed_optimizer_steps": steps})
            except OSError:
                pass
        print("ERROR: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
