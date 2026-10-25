"""Stable FP32/SDPA resource policy; independent of diagnostic Runs."""
def memory_state():
    import torch
    free, total = torch.cuda.mem_get_info()
    allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    return {"free_bytes": free, "total_bytes": total, "allocated_bytes": allocated,
            "reserved_bytes": reserved, "usable_estimate_bytes": free + max(0, reserved - allocated),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()}


def configure(config):
    import torch
    if config["device"] != "cuda" or config["precision"] != "fp32":
        raise ValueError("FP32 control requires explicit CUDA/fp32; no CPU or lower-precision fallback")
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; stop FP32 control")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return memory_state()


def bind(engine):
    import torch
    models = {"target": engine.target}
    if engine.draft is not None:
        models["draft"] = engine.draft
    policy = {"attention_backend": "sdpa", "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
              "tf32_cudnn": torch.backends.cudnn.allow_tf32,
              "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "autocast": torch.is_autocast_enabled(), "models": {}}
    if policy["tf32_matmul"] or policy["tf32_cudnn"] or policy["autocast"]:
        raise ValueError("FP32 policy not effective")
    for name, model in models.items():
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError(f"{name}: not all floating parameters are FP32; no fallback")
        if any(p.device.type != "cuda" for p in model.parameters()):
            raise ValueError(f"{name}: parameters offloaded from CUDA; not this comparison")
        if getattr(model.config, "_attn_implementation", None) != "sdpa":
            setter = getattr(model, "set_attn_implementation", None)
            if setter is None:
                raise ValueError(f"{name}: cannot explicitly select SDPA")
            setter("sdpa")
        if getattr(model.config, "_attn_implementation", None) != "sdpa":
            raise ValueError(f"{name}: attention backend did not become SDPA")
        policy["models"][name] = {"dtype": str(next(model.parameters()).dtype),
                                 "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                                 "attention_backend": model.config._attn_implementation}
    engine.identity["attention_implementation"] = {name: "sdpa" for name in models}
    engine.identity["fp32_control"] = policy
    return policy


def ensure_memory(engine, prompt_length):
    """Conservative KV+workspace headroom check, not a guarantee against OOM."""
    budget = 0
    for model in (engine.target, engine.draft):
        if model is None:
            continue
        c = model.config
        head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        heads = getattr(c, "num_key_value_heads", None) or c.num_attention_heads
        length = prompt_length + engine.config["max_new_tokens"] + max(8, engine.config["standard_draft_tokens"]) + 1
        budget += 2 * c.num_hidden_layers * heads * head_dim * length * 4
    state = memory_state()
    state.update(estimated_kv_bytes=budget, workspace_reserve_bytes=1024 ** 3,
                 estimate_limitation="headroom heuristic; allocator/workspace OOM still fails without fallback")
    if state["usable_estimate_bytes"] < budget + state["workspace_reserve_bytes"]:
        raise RuntimeError(f"Insufficient FP32 KV/workspace headroom: {state}")
    return state


