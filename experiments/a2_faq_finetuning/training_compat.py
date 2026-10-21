"""Explicit legacy/ST 6.1 API adapters; no Trainer, inference API or pooling replacement.

ST 6.1.0 source: sentence_transformers/base/model.py at tag v6.1.0.
Profiles describe statically reviewed APIs, NOT an executed compatibility test.
"""
from __future__ import annotations

from collections.abc import Mapping
import sys

from .common import DataError


def select_training_profile(versions):
    from packaging.version import Version

    required = ("sentence-transformers", "torch", "transformers", "numpy")
    if any(versions.get(name) is None for name in required):
        raise DataError("missing training packages; installed metadata: %r" % versions)
    st, torch_version, transformers, numpy = (Version(versions[name]) for name in required)
    if st == Version("6.1.0"):
        # ST v6.1.0 dependency minima/Transformers major range; A2 additionally caps torch <3.
        if not (sys.version_info >= (3, 10) and Version("2.2") <= torch_version < Version("3")
                and Version("5") <= transformers < Version("6") and numpy >= Version("1.24")):
            raise DataError("ST 6.1.0 profile needs Python>=3.10, torch>=2.2,<3, "
                            "transformers>=5,<6 and numpy>=1.24; actual=%r" % versions)
        return "st_6_1_0_preprocess"
    if (Version("2.2.2") <= st < Version("2.5")
            and Version("2") <= torch_version < Version("2.4")
            and Version("4.43") <= transformers < Version("4.48")):
        return "legacy_st_2_tokenize"
    raise DataError("no statically reviewed A2 training API profile for %r; "
                    "use the existing legacy stack or the ST 6.1.0 profile; "
                    "this program never changes the environment" % versions)


def load_training_model(name, device, offline, profile, max_length):
    from sentence_transformers import SentenceTransformer

    kwargs = {"device": device}
    if profile == "st_6_1_0_preprocess":
        kwargs.update(backend="torch", local_files_only=offline)
    model = SentenceTransformer(name, **kwargs)
    if max_length is not None:
        model.max_seq_length = max_length
    model.to(device).float()
    method = "preprocess" if profile == "st_6_1_0_preprocess" else "tokenize"
    for api in (method, "forward", "save_pretrained" if profile == "st_6_1_0_preprocess" else "save"):
        if not callable(getattr(model, api, None)):
            raise DataError("model lacks required %s API for %s" % (api, profile))
    if callable(getattr(model, "get_backend", None)) and model.get_backend() != "torch":
        raise DataError("A2 gradients require the torch backend")
    if getattr(model, "default_prompt_name", None):
        raise DataError("A2 assumes unprompted BGE-M3 text in training and evaluation; "
                        "a model with a default prompt requires a separate protocol")
    # Preserve loaded modules (including pretrained CLS/mean pooling), never rebuild them.
    if not any(type(module).__name__ == "Pooling" for module in model.children()):
        raise DataError("expected a dense BGE-M3 ST checkpoint with its saved Pooling module")
    return model


def underlying_transformers(model, profile):
    if profile == "st_6_1_0_preprocess":
        transformer = model.transformers_model
        return [transformer] if transformer is not None else []
    return [module.auto_model for module in model.children() if hasattr(module, "auto_model")]


def model_api_description(model, profile):
    modules = []
    for name, module in model.named_children():
        entry = {"name": name, "class": type(module).__module__ + "." + type(module).__name__}
        if type(module).__name__ == "Pooling":
            # v6 and legacy expose different configuration attribute names.
            entry["pooling"] = {key: getattr(module, key) for key in (
                "pooling_mode", "include_prompt", "pooling_mode_cls_token",
                "pooling_mode_mean_tokens", "pooling_mode_max_tokens",
                "pooling_mode_mean_sqrt_len_tokens", "pooling_mode_weightedmean_tokens",
                "pooling_mode_lasttoken") if hasattr(module, key)}
        modules.append(entry)
    return {"profile": profile, "runtime_verified_by_codex": False,
            "training_input_api": "preprocess" if profile == "st_6_1_0_preprocess" else "tokenize",
            "forward_api": "model(features)['sentence_embedding']",
            "pooling_policy": "preserve checkpoint modules", "modules": modules}


def _move_tensors(value, device):
    import torch
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    return value


def forward_training_texts(model, texts, device, profile):
    import torch
    if not model.training or torch.is_inference_mode_enabled():
        raise DataError("training forward requires model.train() outside inference_mode")
    with torch.enable_grad():
        features = (model.preprocess(inputs=texts, prompt=None)
                    if profile == "st_6_1_0_preprocess" else model.tokenize(texts))
        if not isinstance(features, Mapping) or not features:
            raise DataError("preprocessing did not return a nonempty feature mapping")
        output = model(_move_tensors(features, device))
        embeddings = output.get("sentence_embedding") if isinstance(output, Mapping) else None
        if (not torch.is_tensor(embeddings) or embeddings.ndim != 2 or
                embeddings.shape[0] != len(texts) or not embeddings.requires_grad or
                embeddings.grad_fn is None):
            raise DataError("sentence_embedding has wrong shape or is detached from the training graph")
        return embeddings


def embedding_gradient_report(branches):
    """Called AFTER backward on retained query/base/extra outputs; no loss graph conversion."""
    import torch
    report = {}
    for name, embeddings in branches.items():
        gradient = embeddings.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise DataError("missing/non-finite gradient at %s embedding branch" % name)
        report[name] = {"received_gradient": True, "norm": float(gradient.detach().norm().cpu())}
    return report


def save_training_model(model, path, profile):
    if profile == "st_6_1_0_preprocess":
        model.save_pretrained(str(path), create_model_card=False, safe_serialization=True)
    else:
        model.save(str(path), create_model_card=False)


def verify_saved_model(model, path, texts, profile, max_length):
    """Optional future GPU-computer check, not an evaluation query or training step.

    The exported copy loads on CPU to avoid a second full CUDA model allocation.
    Inference is restricted to this post-training round trip, never training forward.
    """
    import torch
    was_training = model.training
    try:
        copy = load_training_model(str(path), "cpu", True, profile, None)
        if copy.max_seq_length != max_length:
            raise DataError("export did not preserve max_seq_length: expected %s, got %s"
                            % (max_length, copy.max_seq_length))
        copy.eval()
        model.eval()
        kwargs = dict(convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        if profile == "st_6_1_0_preprocess":
            kwargs["prompt"] = ""
        with torch.inference_mode():
            expected = model.encode(texts, **kwargs).float().cpu()
            actual = copy.encode(texts, **kwargs).float().cpu()
            matched = expected.shape == actual.shape and torch.allclose(expected, actual, rtol=1e-4, atol=1e-4)
            difference = (float((expected - actual).abs().max()) if expected.shape == actual.shape else None)
        if not matched:
            raise DataError("export reload embedding mismatch (CPU vs original device), max_abs=%r" % difference)
        return {"performed": True, "passed": True, "reload_device": "cpu",
                "rtol": 1e-4, "atol": 1e-4, "max_abs_difference": difference,
                "sample_count": len(texts), "saved_max_seq_length": copy.max_seq_length}
    finally:
        model.train(was_training)
