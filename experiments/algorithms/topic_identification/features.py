"""Real eager prefill attention, explicit content offsets, no guessed spans."""
from __future__ import annotations

import statistics
import time

from experiments.algorithms.shared.a0_adapter import measured, sync
from experiments.algorithms.semantic_cache.cache import fingerprint


METHOD_VERSION = "content_mass_ratio_layer_head_query_mean_v1"


def layer_groups(count):
    if count < 3:
        raise ValueError("layer comparison requires at least three decoder layers")
    a, b = count // 3, 2 * count // 3
    return {"TI3-1": list(range(count)), "TI3-2-early": list(range(a)),
            "TI3-2-middle": list(range(a, b)), "TI3-2-late": list(range(b, count))}


def configure_attention(adapter):
    import transformers
    from transformers.models.llama import modeling_llama
    import inspect
    if transformers.__version__ != "5.16.1":
        raise ValueError("Attention API reviewed for transformers=5.16.1 only; review another version before running")
    model = adapter.llm.model
    if model.config.model_type != "llama" or not callable(getattr(model, "set_attn_implementation", None)):
        raise ValueError("requires local Llama with switchable eager attention")
    model.set_attn_implementation("eager")
    if model.config._attn_implementation != "eager" or model.model.config._attn_implementation != "eager":
        raise ValueError("failed to select eager backend; cannot substitute a fake attention feature")
    return {"method": METHOD_VERSION, "attention_backend": "eager", "capture": "LlamaModel prefill; no LM head",
            "layers": layer_groups(model.config.num_hidden_layers), "heads": "all query heads",
            "source_identity": fingerprint(inspect.getsource(modeling_llama.eager_attention_forward))}


def content_inputs(tokenizer, history, query, device, limit):
    if not tokenizer.is_fast:
        raise ValueError("fast tokenizer with character offsets required")
    messages = [{"role": "system", "content": "請依據對話理解使用者目前的問題。"}]
    messages.extend({"role": m["role"], "content": m["content"].strip()} for m in history)
    messages.append({"role": "user", "content": query.strip()})
    markers = [f"__TI_CONTENT_{i:04d}__" for i in range(len(messages))]
    if any(marker in m["content"] for marker in markers for m in messages):
        raise ValueError("content collides with span marker")
    skeleton = [{"role": m["role"], "content": marker} for m, marker in zip(messages, markers)]
    rendered = tokenizer.apply_chat_template(skeleton, tokenize=False, add_generation_prompt=False)
    spans = []
    previous = -1
    for marker, message in zip(markers, messages):
        if rendered.count(marker) != 1:
            raise ValueError("chat template does not preserve message markers exactly once")
        start = rendered.index(marker)
        if start <= previous:
            raise ValueError("chat template reordered content spans")
        content = message["content"]
        rendered = rendered.replace(marker, content, 1)
        spans.append((start, start + len(content)))
        previous = start + len(content)
    actual = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if rendered != actual:
        raise ValueError("template content transform unsupported; refusing to guess token spans")
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False,
                        truncation=False, return_offsets_mapping=True, padding=False)
    if encoded["input_ids"].shape[1] > limit:
        raise ValueError(f"attention input exceeds {limit} tokens; no silent history truncation")
    offsets = encoded.pop("offset_mapping")[0].tolist()
    ids = encoded["input_ids"][0].tolist()
    mask = encoded["attention_mask"][0].tolist()
    specials = set(tokenizer.all_special_ids)
    indices = []
    for start, end in spans:
        selected = [i for i, ((a, b), token, valid) in enumerate(zip(offsets, ids, mask))
                    if valid and token not in specials and b > a and start <= a and b <= end]
        if not selected:
            raise ValueError("message has no unambiguous content tokens")
        indices.append(selected)
    h = [i for part in indices[1:-1] for i in part]
    q = indices[-1]
    if not h or max(h) >= min(q) or set(h) & set(q):
        raise ValueError("invalid history/current content token ordering")
    return {k: v.to(device) for k, v in encoded.items()}, {
        "input_token_ids": ids, "history_token_indices": h, "query_token_indices": q,
        "content_character_spans": spans, "content_token_indices": indices,
        "rendered_prompt": rendered, "prompt_identity": fingerprint(ids)}


class TopicFeatures:
    def __init__(self, adapter, config):
        self.adapter, self.config = adapter, config
        self.definition = configure_attention(adapter)

    def embedding(self, case):
        import numpy as np
        sync()
        start = time.perf_counter()
        anchor = next(m["content"] for m in reversed(case["history"]) if m["role"] == "user")
        for text in (anchor, case["query"]):
            self.adapter.validate_query(text)
        vectors = self.adapter.embedder.encode([anchor, case["query"]])
        if vectors.shape[0] != 2 or not np.isfinite(vectors).all():
            raise ValueError("invalid BGE embeddings")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
            raise ValueError("invalid embedding norms")
        score = float(np.dot(vectors[0] / norms[0], vectors[1] / norms[1]))
        sync()
        return {"scores": {"TI0": score}, "detection_seconds": time.perf_counter() - start,
                "anchor": anchor, "history_anchor": "most recent historical user turn"}

    def attention(self, case):
        import torch
        sync()
        start = time.perf_counter()
        inputs, trace = content_inputs(self.adapter.llm.tokenizer, case["history"], case["query"],
                                       self.config["device"], self.config["max_attention_tokens"])
        def forward():
            with torch.inference_mode():
                return self.adapter.llm.model.model(**inputs, output_attentions=True,
                                                     use_cache=False, return_dict=True)
        cuda = self.config["device"] == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        outputs, forward_seconds = measured(forward)
        weights = getattr(outputs, "attentions", None)
        groups = self.definition["layers"]
        count = self.adapter.llm.model.config.num_hidden_layers
        heads = self.adapter.llm.model.config.num_attention_heads
        size = inputs["input_ids"].shape[1]
        if weights is None or len(weights) != count:
            raise ValueError("eager LlamaModel did not return one attention tensor per layer")
        q = torch.tensor(trace["query_token_indices"], device=self.config["device"])
        h = torch.tensor(trace["history_token_indices"], device=self.config["device"])
        per_layer = []
        with torch.inference_mode():
            for layer in weights:
                if layer is None or tuple(layer.shape) != (1, heads, size, size):
                    raise ValueError("attention shape must be (batch=1, query_heads, S, S)")
                selected = layer[0].index_select(1, q).float()  # (heads,Q,S), never average batch with heads
                history_mass = selected.index_select(2, h).sum(-1)
                current_mass = selected.index_select(2, q).sum(-1)  # future positions have causal zero mass
                denom = history_mass + current_mass
                if (not torch.isfinite(denom).all() or torch.any(denom <= 0)
                        or torch.any(history_mass < 0) or torch.any(current_mass < 0)):
                    raise ValueError("invalid real attention mass")
                per_layer.append(float((history_mass / denom).mean().item()))
        scores = {name: statistics.mean(per_layer[i] for i in indices) for name, indices in groups.items()}
        peak = torch.cuda.max_memory_allocated() if cuda else None
        del weights, outputs
        sync()
        return {"scores": scores, "per_layer_scores": per_layer,
                "detection_seconds": time.perf_counter() - start, "forward_seconds": forward_seconds,
                "cuda_peak_allocated_bytes": peak,
                "trace": trace, "feature_definition": METHOD_VERSION}
