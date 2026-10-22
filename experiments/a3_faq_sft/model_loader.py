"""Shared base/adapter loading for A3 and a future A5; no FAQ retrieval."""
from __future__ import annotations

import os
from importlib import metadata

from .common import BASE, DataError, file_hashes, read_json, resolve


def versions():
    result = {}
    for name in ("torch", "transformers", "peft", "accelerate", "trl", "datasets", "bitsandbytes"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def configure_offline(offline):
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def model_identity(root, name):
    path = resolve(root, name)
    if path.exists():
        if not path.is_dir() or not (path / "config.json").is_file():
            raise DataError("not a local base model directory: %s" % path)
        if (path / "adapter_config.json").exists():
            raise DataError("--base-model must be a complete base, not an adapter")
        # Includes weights: relocation is supported only with identical file content.
        return str(path), {"kind": "local", "files_sha256": file_hashes(path)}
    if name != BASE:
        raise DataError("base path missing (no silent Hub fallback): %s" % path)
    return name, {"kind": "hub", "model_id": name}


def load_tokenizer(name, offline, revision=None):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=offline, revision=revision,
                                              trust_remote_code=False)
    if not tokenizer.chat_template or tokenizer.eos_token_id is None:
        raise DataError("base tokenizer must supply native chat template and EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base(name, device, precision, offline, revision=None):
    import torch
    from transformers import AutoModelForCausalLM
    if device not in ("cuda", "cpu"):
        raise DataError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise DataError("CUDA requested but unavailable")
    if device == "cpu" and precision != "fp32":
        raise DataError("CPU requires --precision fp32")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise DataError("BF16 unavailable; explicitly select fp16 or fp32")
    from packaging.version import Version
    # Transformers 5 uses dtype; legacy project 4.x uses torch_dtype.
    key = "dtype" if Version(metadata.version("transformers")) >= Version("5") else "torch_dtype"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    model = AutoModelForCausalLM.from_pretrained(
        name, **{key: dtype}, local_files_only=offline, revision=revision,
        trust_remote_code=False, attn_implementation="sdpa")
    model.to(device)
    if getattr(model.config, "model_type", None) != "llama":
        raise DataError("this A3 protocol requires a Llama causal LM")
    return model


def verify_adapter(path):
    if not path.is_dir():
        raise DataError("adapter directory missing: %s" % path)
    complete = read_json(path / "a3_complete.json")
    if (complete.get("schema_version") != "a3_adapter_v1" or
            complete.get("files_sha256") != file_hashes(path, exclude=("a3_complete.json",))):
        raise DataError("adapter save incomplete or file hashes changed: %s" % path)
    if not (path / "adapter_model.safetensors").is_file():
        raise DataError("adapter weights missing")
    provenance = read_json(path / "a3_provenance.json")
    if (provenance.get("schema_version") != "a3_training_v1"
            or type(provenance.get("provisional")) is not bool
            or type(provenance.get("optimizer_steps")) is not int
            or provenance["optimizer_steps"] < 1
            or not isinstance(provenance.get("system_prompt"), str)
            or not provenance["system_prompt"].strip()):
        raise DataError("invalid adapter training provenance")
    return provenance


def load_adapter_for_chat(root, base_argument, adapter, device, precision, offline,
                          allow_unreviewed=False):
    """Does not read CSV, SFT examples, FAQ indexes or any evaluation queries."""
    configure_offline(offline)
    provenance = verify_adapter(adapter)
    if provenance.get("provisional") and not allow_unreviewed:
        raise DataError("provisional adapter; pass --allow-unreviewed for manual smoke use")
    name, identity = model_identity(root, base_argument)
    if identity != provenance["base_identity"]:
        raise DataError("base model differs from adapter training source")
    revision = provenance.get("base_revision")
    if identity["kind"] == "hub" and not revision:
        raise DataError("missing pinned base revision")
    from peft import PeftModel
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(adapter), local_files_only=True,
                                              trust_remote_code=False)
    if not tokenizer.chat_template or tokenizer.pad_token_id is None:
        raise DataError("adapter tokenizer/template/pad configuration missing")
    base = load_base(name, device, precision, offline, revision)
    if base.config.vocab_size != provenance["vocab_size"] or len(tokenizer) != provenance["tokenizer_size"]:
        raise DataError("base/tokenizer vocabulary differs from training")
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False,
                                     local_files_only=True)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer, provenance
