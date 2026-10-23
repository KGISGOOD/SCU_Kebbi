"""Strict, fail-closed compatibility policy, reviewed against HF 5.16.1."""
from __future__ import annotations

import inspect
import json
import hashlib

from experiments.algorithms.semantic_cache.cache import fingerprint


SUPPORTED_TRANSFORMERS = "5.16.1"


def check_api():
    import transformers
    from transformers.generation.configuration_utils import GenerationConfig
    from transformers.generation.utils import GenerationMixin
    from transformers.generation.candidate_generator import AssistedCandidateGenerator
    if transformers.__version__ != SUPPORTED_TRANSFORMERS:
        raise ValueError(f"SD source review covers transformers={SUPPORTED_TRANSFORMERS}, got "
                         f"{transformers.__version__}; review that version before running, no fallback")
    if "assistant_model" not in inspect.signature(GenerationMixin.generate).parameters:
        raise ValueError("installed generate API has no assistant_model")
    for key in ("num_assistant_tokens", "num_assistant_tokens_schedule", "assistant_confidence_threshold"):
        if not hasattr(GenerationConfig(), key):
            raise ValueError(f"installed GenerationConfig lacks {key}")
    return {"transformers": transformers.__version__, "reviewed_not_runtime_certified": True,
            "generate_source_sha256": hashlib.sha256(inspect.getsource(GenerationMixin.generate).encode("utf-8")).hexdigest(),
            "assisted_source_sha256": hashlib.sha256(inspect.getsource(GenerationMixin._assisted_decoding).encode("utf-8")).hexdigest(),
            "candidate_source_sha256": hashlib.sha256(inspect.getsource(AssistedCandidateGenerator).encode("utf-8")).hexdigest()}


def tokenizer_identity(tokenizer):
    if not tokenizer.is_fast:
        raise ValueError("SD V1 requires fast tokenizers for complete backend comparison")
    return {"class": type(tokenizer).__name__, "length": len(tokenizer),
            "vocabulary_sha256": fingerprint(tokenizer.get_vocab()),
            "backend_sha256": fingerprint(json.loads(tokenizer.backend_tokenizer.to_str())),
            "special_tokens": {k: str(v) for k, v in tokenizer.special_tokens_map.items()},
            "all_special_ids": list(tokenizer.all_special_ids),
            "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id, "chat_template": tokenizer.chat_template}


def check_pair(target, target_tokenizer, draft, draft_tokenizer):
    left, right = tokenizer_identity(target_tokenizer), tokenizer_identity(draft_tokenizer)
    # chat_template does not need to match: the TARGET's A0 prompt is encoded
    # once and its token IDs are sent directly to both models. Record both.
    for key in left.keys() - {"chat_template"}:
        if left[key] != right[key]:
            raise ValueError(f"Target/Draft tokenizer mismatch: {key}; no automatic translation")
    for name, model in (("target", target), ("draft", draft)):
        if model.config.model_type != "llama" or model.config.is_encoder_decoder:
            raise ValueError(f"{name}: a local decoder-only Llama checkpoint is required")
        if (model.get_input_embeddings().weight.shape[0] != len(target_tokenizer)
                or model.get_output_embeddings().weight.shape[0] != len(target_tokenizer)):
            raise ValueError(f"{name}: model vocabulary dimensions differ from shared token space")
        for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            # Missing model pad is allowed: both tokenizers explicitly use EOS
            # as pad and the common GenerationConfig provides this value.
            a, b = getattr(target.config, key, None), getattr(draft.config, key, None)
            if a != b:
                raise ValueError(f"model special-token config differs: {key}: {a!r} vs {b!r}")
    return {"status": "passed_static_runtime_checks", "target_tokenizer": left,
            "draft_tokenizer": right, "token_space": "identical; no UAD conversion",
            "output_equivalence": "must still be checked for every measured run"}


def adaptive_policy(*args, **kwargs):
    """Reserved SD2-3 extension; never silently routed to a fixed policy."""
    raise NotImplementedError("SD2-3 Adaptive Speculation is not implemented in this phase")
