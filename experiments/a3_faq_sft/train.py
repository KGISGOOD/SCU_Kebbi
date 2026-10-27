"""Single-device PEFT LoRA SFT with explicit assistant-only causal labels.

No Trainer/TRL version mixing, no retrieval, no validation/test optimization.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import uuid

from .common import (DATA, ROOT, DataError, file_hashes, resolve, read_json, sha,
                     validate_data, write_json)
from .model_loader import (configure_offline, load_base, load_tokenizer, model_identity,
                           versions, verify_adapter)
from .training_data import collate, encode_sample


def arguments():
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument("--project-root", type=Path, default=ROOT)
    first.add_argument("--config", type=Path, default=Path(__file__).with_name("training_config.json"))
    early, _ = first.parse_known_args()
    config = read_json(resolve(early.project_root.expanduser().resolve(), early.config))
    parser = argparse.ArgumentParser(description=__doc__, parents=[first])
    parser.add_argument("--data-dir", default=DATA)
    parser.add_argument("--training-data", help="must be the validated data-dir/train.jsonl")
    parser.add_argument("--base-model", default=config["base_model"])
    parser.add_argument("--revision", help="Hub revision; resolved commit recorded and reused on reload")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default=config["precision"])
    for name in ("lora_r", "lora_alpha", "epochs", "batch_size", "gradient_accumulation",
                 "max_length", "seed"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=config[name])
    for name in ("lora_dropout", "learning_rate", "weight_decay", "max_grad_norm"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=config[name])
    parser.add_argument("--target-modules", nargs="+", default=config["target_modules"])
    parser.add_argument("--max-steps", type=int, help="explicit smoke limit; always provisional, not full training")
    parser.add_argument("--output-dir", help="must not exist; default: unique timestamped A3 run")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--training-review", help="external human approval JSON bound to the exact Train and dataset manifest hashes")
    parser.add_argument("--check-only", action="store_true", help="check data and tokenizer labels; no model weights")
    parser.add_argument("--final-only", action="store_true", help="save only final_adapter, not intermediate epoch checkpoints")
    args = parser.parse_args()
    for name in ("lora_r", "lora_alpha", "epochs", "batch_size", "gradient_accumulation", "max_length"):
        if getattr(args, name) < 1:
            parser.error("%s must be positive" % name)
    if not 0 <= args.seed < 2**32 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("seed out of range or nonpositive max-steps")
    if (not all(math.isfinite(getattr(args, n)) for n in
                ("lora_dropout", "learning_rate", "weight_decay", "max_grad_norm"))
            or not 0 <= args.lora_dropout < 1 or args.learning_rate <= 0
            or args.weight_decay < 0 or args.max_grad_norm <= 0):
        parser.error("invalid dropout/learning-rate/weight-decay/max-grad-norm")
    if not args.target_modules or len(set(args.target_modules)) != len(args.target_modules):
        parser.error("target-modules must be nonempty and unique")
    return parser, args, config


def main():
    output = None
    steps = 0
    try:
        parser, args, config = arguments()
        root = args.project_root.expanduser().resolve()
        directory = resolve(root, args.data_dir)
        data, dataset_manifest = validate_data(root, directory)
        training_path = directory / "train.jsonl"
        if args.training_data and resolve(root, args.training_data) != training_path:
            raise DataError("--training-data must be the validated data-dir/train.jsonl; no split substitution")
        dataset_hashes = file_hashes(directory)
        pending = [row["faq_id"] for row in data["train"] if row["review_status"] != "approved"]
        review_path, review_hash, training_review = None, None, None
        if args.training_review:
            review_path = resolve(root, args.training_review)
            review_hash = sha(review_path)
            training_review = read_json(review_path)
            if dataset_manifest["schema_version"] in {"a3_sft_data_v2", "training_v2_converted_v1"}:
                from experiments.research_tools.training_approval import verify
                training_review, review_hash = verify(review_path, "a3_training_review_v1", training_path,
                                                     directory / "dataset_manifest.json",
                                                     [r["faq_id"] for r in data["train"]])
            ids = [row["faq_id"] for row in data["train"]]
            approved = training_review.get("approved_faq_ids")
            if (training_review.get("schema_version") != "a3_training_review_v1"
                    or training_review.get("train_sha256") != sha(training_path)
                    or training_review.get("dataset_manifest_sha256") != sha(directory / "dataset_manifest.json")
                    or not isinstance(approved, list) or any(not isinstance(i, str) for i in approved)
                    or len(approved) != len(ids) or set(approved) != set(ids)
                    or any(not isinstance(training_review.get(k), str) or not training_review[k].strip()
                           for k in ("reviewer", "reviewed_at", "evidence"))):
                raise DataError("invalid human Train approval: require exact Train IDs, hashes, reviewer, timestamp and evidence")
            reviewed_at = datetime.fromisoformat(training_review["reviewed_at"].replace("Z", "+00:00"))
            if reviewed_at.tzinfo is None or sha(review_path) != review_hash:
                raise DataError("Train approval needs a timezone and must remain unchanged")
            pending = []  # Explicit external human approval; immutable dataset stays needs_review.
        if (dataset_manifest["schema_version"] in {"a3_sft_data_v2", "training_v2_converted_v1"}
                and not args.training_review and not args.check_only and not args.allow_unreviewed):
            raise DataError("revised supervised answers need human --training-review for formal training; --allow-unreviewed explicitly permits provisional training")
        if (not args.check_only and not args.allow_unreviewed
                and dataset_manifest["schema_version"] not in {"a3_sft_data_v2", "training_v2_converted_v1"}):
            raise DataError("formal A3 training requires a revised v2 dataset; original v1 remains available for provisional Smoke")
        if pending and not args.allow_unreviewed and not args.check_only:
            raise DataError("Train rows need human review: %s; --allow-unreviewed permits provisional smoke only"
                            % ",".join(pending))
        configure_offline(args.offline)
        name, identity = model_identity(root, args.base_model)
        tokenizer = load_tokenizer(name, args.offline, args.revision)
        tokenized = [encode_sample(tokenizer, row, args.max_length) for row in data["train"]]
        validation_tokens = ([encode_sample(tokenizer, row, args.max_length) for row in data["validation"]]
                             if dataset_manifest["schema_version"] == "training_v2_converted_v1" else [])
        token_summary = {
            "train_examples": len(tokenized), "train_faq_ids": [r["faq_id"] for r in tokenized],
            "max_sequence_tokens": max(len(r["input_ids"]) for r in tokenized),
            "supervised_tokens": sum(r["supervised_tokens"] for r in tokenized),
            "masking": "system/user/assistant header and padding=-100; assistant content+turn ending supervised",
            "truncation": "forbidden; overflow is an error", "pending_review": pending,
            "validation_examples_tokenized": len(validation_tokens),
        }
        if args.check_only:
            print(json.dumps({"structural_ok": True, "tokenization_ok": True,
                              "weights_loaded": False, "package_versions": versions(),
                              **token_summary}, ensure_ascii=False, indent=2))
            return 0
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = resolve(root, args.output_dir or
                              ("experiments/models/a3_faq_sft/" + timestamp + "_" + uuid.uuid4().hex[:8]))
        if output_path.exists():
            raise DataError("output already exists; choose a NEW directory: %s" % output_path)
        # Never write a run inside a base/data directory.
        for protected in (directory, resolve(root, args.base_model)):
            if protected.exists() and (output_path == protected or protected in output_path.parents):
                raise DataError("output must not be inside model/data inputs")
        runtime_versions = versions()
        if any(runtime_versions[k] is None for k in ("torch", "transformers", "peft")):
            raise DataError("missing torch/transformers/peft; no environment changes are made: %r" % runtime_versions)
        import torch
        from peft import LoraConfig, get_peft_model
        if not hasattr(torch, "amp") or not hasattr(torch.amp, "GradScaler"):
            raise DataError("this loop requires torch.amp.GradScaler; installed=%r" % runtime_versions)

        output_path.mkdir(parents=True, exist_ok=False)
        output = output_path  # only a directory created by THIS run may receive failure.json
        write_json(output / "started.json", {"status": "started", "utc": timestamp})
        write_json(output / "training_config.json", {
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "defaults": config, "versions": runtime_versions, "python": platform.python_version(),
            "code_sha256": {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "approval_code_sha256": sha(Path(__file__).resolve().parents[1] / "research_tools/training_approval.py"),
            "config_sha256": sha(resolve(root, args.config)),
            "optimizer": "AdamW; constant LR; token-weighted gradient accumulation",
        })
        write_json(output / "dataset_manifest.json", dataset_manifest)
        if training_review is not None:
            if sha(review_path) != review_hash:
                raise DataError("human Train approval changed before training")
            write_json(output / "training_review.snapshot.json", training_review)
        random.seed(args.seed)
        if file_hashes(directory) != dataset_hashes:
            raise DataError("prepared inputs changed before model loading")
        validate_data(root, directory)
        if review_path is not None and sha(review_path) != review_hash:
            raise DataError("Train approval changed before model loading")
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        model = load_base(name, args.device, args.precision, args.offline, args.revision)
        revision = getattr(model.config, "_commit_hash", None)
        if identity["kind"] == "hub" and not revision:
            raise DataError("cannot resolve base commit; use a pinned complete local checkpoint")
        # If Hub 'main' moved between tokenizer and model load, reload the tokenizer
        # against the actual base commit and revalidate all label boundaries.
        if identity["kind"] == "hub":
            tokenizer = load_tokenizer(name, args.offline, revision)
            tokenized = [encode_sample(tokenizer, row, args.max_length) for row in data["train"]]
            if validation_tokens:
                validation_tokens = [encode_sample(tokenizer, row, args.max_length) for row in data["validation"]]
        token_summary.update(
            max_sequence_tokens=max(len(r["input_ids"]) for r in tokenized),
            supervised_tokens=sum(r["supervised_tokens"] for r in tokenized))
        write_json(output / "tokenization_summary.json", token_summary)
        if args.max_length > model.config.max_position_embeddings:
            raise DataError("max-length exceeds base context window")
        matched = {target: [n for n, _ in model.named_modules() if n.endswith("." + target)]
                   for target in args.target_modules}
        if any(not names for names in matched.values()):
            raise DataError("missing LoRA target modules: %r" % matched)
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none", target_modules=args.target_modules))
        # Frozen base is BF16/FP16; adapters stay FP32, avoiding unscale on FP16 grads.
        parameters = []
        for parameter_name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if "lora_" not in parameter_name:
                    raise DataError("unexpected trainable base parameter: %s" % parameter_name)
                parameter.data = parameter.data.float()
                parameters.append(parameter)
        if not parameters:
            raise DataError("no trainable adapter parameters")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16" and args.device == "cuda")
        amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
        provisional = bool(pending or args.allow_unreviewed or args.max_steps is not None)
        provenance = {
            "schema_version": "a3_training_v1", "base_identity": identity, "base_revision": revision,
            "base_argument": args.base_model, "system_prompt": data["train"][0]["messages"][0]["content"],
            "provisional": provisional, "vocab_size": model.config.vocab_size,
            "tokenizer_size": len(tokenizer), "seed": args.seed,
            "source_sha256": dataset_manifest["source_sha256"],
            "train_sha256": sha(training_path), "training_faq_ids": token_summary["train_faq_ids"],
            "versions": runtime_versions, "matched_target_modules": matched,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "training_precision": args.precision, "max_length": args.max_length,
            "training_review_sha256": review_hash,
            "data_contract": dataset_manifest["schema_version"],
            "dataset_manifest_sha256": sha(directory / "dataset_manifest.json"),
        }

        def save_adapter(destination, epoch):
            if review_path is not None and sha(review_path) != review_hash:
                raise DataError("human Train approval changed during training")
            if file_hashes(directory) != dataset_hashes:
                raise DataError("prepared inputs changed during training")
            # Recheck source files and split against the preparation snapshot.
            validate_data(root, directory)
            if identity["kind"] == "local" and model_identity(root, args.base_model)[1] != identity:
                raise DataError("base files changed during training")
            destination.mkdir(parents=True, exist_ok=False)
            model.save_pretrained(str(destination), safe_serialization=True)
            tokenizer.save_pretrained(str(destination))
            write_json(destination / "a3_provenance.json", {
                **provenance, "optimizer_steps": steps, "epoch": epoch,
                "smoke_step_limit": args.max_steps,
            })
            if not (destination / "adapter_model.safetensors").is_file():
                raise DataError("PEFT did not export adapter_model.safetensors")
            write_json(destination / "a3_complete.json", {
                "schema_version": "a3_adapter_v1", "files_sha256": file_hashes(destination),
            })
            verify_adapter(destination)

        rng = random.Random(args.seed)
        model.train()
        stop = False
        completed_epochs = 0
        consumed_examples = 0
        with (output / "training_log.jsonl").open("x", encoding="utf-8") as log:
            for epoch in range(1, args.epochs + 1):
                order = list(range(len(tokenized)))
                rng.shuffle(order)
                batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
                epoch_consumed = 0
                for start in range(0, len(batches), args.gradient_accumulation):
                    window = batches[start:start + args.gradient_accumulation]
                    token_count = sum(tokenized[i]["supervised_tokens"] for batch in window for i in batch)
                    optimizer.zero_grad(set_to_none=True)
                    weighted_loss = 0.0
                    window_ids = []
                    for indices in window:
                        samples = [tokenized[i] for i in indices]
                        inputs = collate(samples, tokenizer.pad_token_id, args.device)
                        # HF causal LM shifts labels internally; prefix labels are -100.
                        # A window uses one mean over supervised tokens, including tails.
                        count = sum(row["supervised_tokens"] for row in samples)
                        context = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                                   if args.device == "cuda" and args.precision != "fp32" else nullcontext())
                        with context:
                            loss = model(**inputs).loss
                        if loss is None or not torch.isfinite(loss):
                            raise DataError("missing/non-finite assistant loss")
                        weighted_loss += float(loss.detach().float().cpu()) * count
                        scaler.scale(loss * (count / token_count)).backward()
                        window_ids.extend(row["faq_id"] for row in samples)
                    scaler.unscale_(optimizer)
                    gradients = [p.grad for p in parameters if p.grad is not None]
                    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                        raise DataError("missing/non-finite adapter gradients; step not counted")
                    norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
                    if float(norm.detach().cpu()) <= 0:
                        raise DataError("zero adapter gradient; refusing false-success step")
                    scaler.step(optimizer)
                    scaler.update()
                    if not all(torch.isfinite(p).all() for p in parameters):
                        raise DataError("non-finite adapter parameters after optimizer step")
                    steps += 1
                    epoch_consumed += len(window_ids)
                    consumed_examples += len(window_ids)
                    record = {"optimizer_step": steps, "epoch": epoch, "faq_ids": window_ids,
                              "supervised_tokens": token_count, "loss": weighted_loss / token_count,
                              "gradient_norm_before_clip": float(norm.detach().cpu())}
                    log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    log.flush()
                    print("step=%d epoch=%d assistant_loss=%.6f" % (steps, epoch, record["loss"]), flush=True)
                    if args.max_steps is not None and steps >= args.max_steps:
                        stop = True
                        break
                if epoch_consumed == len(tokenized):
                    completed_epochs += 1
                if validation_tokens and not stop:
                    model.eval()
                    val_loss, val_count = 0.0, 0
                    with torch.inference_mode():
                        for start_index in range(0, len(validation_tokens), args.batch_size):
                            samples = validation_tokens[start_index:start_index + args.batch_size]
                            inputs = collate(samples, tokenizer.pad_token_id, args.device)
                            context = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                                       if args.device == "cuda" and args.precision != "fp32" else nullcontext())
                            with context:
                                loss = model(**inputs).loss
                            if loss is None or not torch.isfinite(loss):
                                raise DataError("nonfinite validation assistant loss")
                            count = sum(r["supervised_tokens"] for r in samples)
                            val_loss += float(loss.float().cpu()) * count
                            val_count += count
                    log.write(json.dumps({"phase": "validation_no_optimizer", "epoch": epoch,
                                          "validation_loss": val_loss / val_count,
                                          "supervised_tokens": val_count,
                                          "examples": len(validation_tokens)}, allow_nan=False) + "\n")
                    log.flush()
                    model.train()
                if stop:
                    break
                if epoch < args.epochs and not args.final_only:
                    save_adapter(output / "checkpoints" / ("epoch_%03d" % epoch), epoch)
        save_adapter(output / "final_adapter", epoch)
        write_json(output / "completed.json", {
            "status": "completed_smoke" if args.max_steps is not None else "completed_training",
            "provisional": provisional, "optimizer_steps": steps, "completed_epochs": completed_epochs,
            "examples_consumed": consumed_examples, "validation_used": bool(validation_tokens),
            "validation_selection_performed": False,
            "adapter": str(output / "final_adapter"), "reload_inference_verified": False,
        })
        print("Saved A3 adapter: %s; provisional=%s" % (output / "final_adapter", provisional))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if output is not None:
            try:
                write_json(output / "failure.json", {"status": "failed", "optimizer_steps": steps,
                                                     "type": type(exc).__name__, "message": str(exc)})
            except OSError:
                pass
        import sys
        print("ERROR [%s]: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2


if __name__ == "__main__":
    raise SystemExit(main())
