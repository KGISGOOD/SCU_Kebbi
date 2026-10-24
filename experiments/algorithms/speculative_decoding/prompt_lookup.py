"""Request-local HF prompt lookup audit. No package changes or substitute decoding."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect


def check_lookup_api():
    from transformers import GenerationConfig
    from transformers.generation.candidate_generator import PromptLookupCandidateGenerator
    from transformers.generation.utils import GenerationMixin
    config = GenerationConfig()
    for key in ("prompt_lookup_num_tokens", "max_matching_ngram_size"):
        if not hasattr(config, key):
            raise ValueError(f"installed Prompt Lookup API lacks {key}; no fallback")
    return {"candidate_class": PromptLookupCandidateGenerator.__name__,
            "candidate_source_sha256": hashlib.sha256(inspect.getsource(PromptLookupCandidateGenerator).encode()).hexdigest(),
            "selector_source_sha256": hashlib.sha256(inspect.getsource(GenerationMixin._get_candidate_generator).encode()).hexdigest()}


@contextmanager
def lookup_request(target, length, ngram):
    """Sequential engine only: restore instance defaults/hooks even on failure.

    Count real candidate calls/proposals, NOT accepted tokens. An empty proposal
    is valid. Target's official assisted loop still verifies every continuation.
    """
    from transformers.generation.candidate_generator import PromptLookupCandidateGenerator
    original_config = target.generation_config
    local = deepcopy(original_config)
    local.use_cache, local.cache_implementation, local.cache_config = True, None, None
    local.prompt_lookup_num_tokens, local.max_matching_ngram_size = length, ngram
    existed = "_get_candidate_generator" in target.__dict__
    original_attribute = target.__dict__.get("_get_candidate_generator")
    factory = target._get_candidate_generator
    stats = {"generator_created": False, "candidate_calls": 0, "proposed_token_count": 0,
             "empty_proposal_calls": 0, "accepted_token_count": None}

    def audited_factory(*args, **kwargs):
        generator = factory(*args, **kwargs)
        if not isinstance(generator, PromptLookupCandidateGenerator):
            raise RuntimeError(f"requested Prompt Lookup, got {type(generator).__name__}; no fallback")
        if generator.num_output_tokens != length or generator.max_matching_ngram_size != ngram:
            raise RuntimeError("HF candidate length/ngram differs from requested settings")
        stats["generator_created"] = True
        get_candidates = generator.get_candidates

        def audited_candidates(input_ids, **candidate_kwargs):
            result = get_candidates(input_ids, **candidate_kwargs)
            proposed = result[0].shape[-1] - input_ids.shape[-1]
            stats["candidate_calls"] += 1
            stats["proposed_token_count"] += proposed
            stats["empty_proposal_calls"] += int(proposed == 0)
            return result

        generator.get_candidates = audited_candidates
        return generator

    try:
        target.generation_config = local
        target._get_candidate_generator = audited_factory
        yield stats
    finally:
        target.generation_config = original_config
        if existed:
            target._get_candidate_generator = original_attribute
        else:
            delattr(target, "_get_candidate_generator")
