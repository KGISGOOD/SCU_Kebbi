"""Independent greedy speculative reference: local Draft proposals, Target block verification.

Not HF assisted generation, not a replacement for V1 evidence. Block forward
floating-point results may still differ from serial greedy; equality is tested,
never assumed. Only plain greedy configurations are supported explicitly.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import time
import traceback

from .diagnose_logits import read
from .reporting import write_json, append_jsonl
from .engine import SDEngine
from experiments.algorithms.shared.a0_adapter import sync
from experiments.algorithms.semantic_cache.cache import fingerprint


def check_plain(config):
    required = {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 0, "min_length": 0, "min_new_tokens": None,
                "forced_bos_token_id": None, "forced_eos_token_id": None,
                "bad_words_ids": None, "force_words_ids": None, "suppress_tokens": None,
                "begin_suppress_tokens": None, "sequence_bias": None,
                "constraints": None, "stop_strings": None, "watermarking_config": None,
                "exponential_decay_length_penalty": None, "penalty_alpha": None,
                "max_time": None, "guidance_scale": None, "token_healing": False}
    # Reviewed against HF v5.16.1 generation/configuration_utils.py global
    # defaults and generation/utils.py _get_logits_processor / token_healing.
    # None -> 0 for these two constraints (active only if >0); None healing
    # is false in its `if` branch. This is validation-only: never mutate the
    # GenerationConfig or broadly treat arbitrary None/0/False as equivalent.
    # SDEngine.check_api enforces the reviewed 5.16.1 version before decode.
    none_disabled = {"no_repeat_ngram_size": 0, "min_length": 0, "token_healing": False}
    errors = {}
    for key, expected in required.items():
        actual = getattr(config, key, expected)
        checked = none_disabled[key] if actual is None and key in none_disabled else actual
        if checked != expected:
            errors[key] = {"expected": expected, "actual": actual}
    if errors:
        raise ValueError(f"V2 supports plain greedy only; no silent processor removal: {errors}")


def decode(engine, prompt, k):
    import torch
    check_plain(engine.generation)
    eos = engine.generation.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    limit = engine.config["max_new_tokens"]
    prefix = list(prompt)
    initial = len(prefix)
    stats = {"target_forward_calls": 0, "draft_calls": 0, "proposed_tokens": 0,
             "accepted_draft_tokens": 0, "cache_crops": [], "eos_candidate_seen": False}
    cache = None

    def forward(values, start):
        nonlocal cache
        x = torch.tensor([values], device=engine.config["device"])
        result = engine.target(input_ids=x,
            attention_mask=torch.ones((1, start + len(values)), dtype=torch.long, device=x.device),
            cache_position=torch.arange(start, start + len(values), device=x.device),
            past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1 if start == 0 else 0)
        cache = result.past_key_values
        if cache is None or not hasattr(cache, "crop") or not hasattr(cache, "get_seq_length"):
            raise RuntimeError("V2 requires an observable, crop-capable DynamicCache; no fallback")
        if int(cache.get_seq_length()) != start + len(values):
            raise RuntimeError("Target KV cache length does not match committed/candidate prefix")
        stats["target_forward_calls"] += 1
        return result.logits[0]

    def crop(length):
        before = int(cache.get_seq_length())
        cache.crop(length)
        after = int(cache.get_seq_length())
        if after != length:
            raise RuntimeError("Target cache crop failed")
        stats["cache_crops"].append({"before": before, "after": after})

    sync()
    started = time.perf_counter()
    with torch.inference_mode():
        next_logits = forward(prefix, 0)[-1]
        while len(prefix) - initial < limit:
            remaining = limit - (len(prefix) - initial)
            if k is None:
                chosen = int(next_logits.argmax())
                prefix.append(chosen)
                if chosen in eos or len(prefix) - initial == limit:
                    break
                next_logits = forward([chosen], len(prefix) - 1)[-1]
                continue
            draft_config = deepcopy(engine.generation)
            draft_config.max_new_tokens = min(k, remaining)
            draft_config.cache_implementation = "dynamic"
            draft_input = torch.tensor([prefix], device=engine.config["device"])
            draft_defaults = engine.draft.generation_config
            try:
                # HF can fill None request fields from model defaults. Bind both
                # layers to the same explicit policy for this fresh Draft call.
                engine.draft.generation_config = deepcopy(draft_config)
                proposals = engine.draft.generate(input_ids=draft_input, attention_mask=torch.ones_like(draft_input),
                                                   generation_config=draft_config)
            finally:
                engine.draft.generation_config = draft_defaults
            if proposals[0, :len(prefix)].tolist() != prefix:
                raise RuntimeError("Draft did not preserve prefix")
            candidates = proposals[0, len(prefix):].tolist()
            if not candidates or len(candidates) > min(k, remaining):
                raise RuntimeError("invalid Draft proposal count")
            # Candidate generation may stop at EOS; EOS is still verified by Target.
            stats["draft_calls"] += 1
            stats["proposed_tokens"] += len(candidates)
            stats["eos_candidate_seen"] |= any(t in eos for t in candidates)
            start = len(prefix)
            block = forward(candidates, start)
            predictions = [int(next_logits.argmax())] + block[:-1].argmax(dim=-1).tolist()
            mismatch = next((j for j, (a, b) in enumerate(zip(candidates, predictions)) if a != b), len(candidates))
            emitted = candidates[:mismatch]
            stats["accepted_draft_tokens"] += mismatch
            if mismatch < len(candidates):
                emitted.append(predictions[mismatch])
            elif len(emitted) < remaining:
                emitted.append(int(block[-1].argmax()))
            # Stop at the first TARGET-approved EOS, including an accepted Draft EOS.
            stop_at = next((j + 1 for j, token in enumerate(emitted) if token in eos), len(emitted))
            emitted = emitted[:stop_at]
            prefix.extend(emitted)
            stopped = prefix[-1] in eos or len(prefix) - initial >= limit
            if stopped:
                # Last correction/bonus may not yet be cached. The cache is discarded
                # at request end; never claim that an unprocessed token is cached.
                if int(cache.get_seq_length()) > len(prefix):
                    crop(len(prefix))
                break
            if mismatch < len(candidates):
                crop(start + mismatch)
                next_logits = forward([emitted[-1]], len(prefix) - 1)[-1]
            else:
                # All proposals accepted; process the Target bonus for next round.
                next_logits = forward([emitted[-1]], len(prefix) - 1)[-1]
        sync()
    elapsed = time.perf_counter() - started
    ids = prefix[initial:]
    return {"output_token_ids": ids, "output_token_sha256": fingerprint(ids),
            "output_text": engine.tokenizer.decode(ids, skip_special_tokens=True),
            "generation_seconds": elapsed, "generated_token_count": len(ids),
            "stop_reason": "eos" if ids[-1] in eos else "token_limit",
            "final_cache_length": int(cache.get_seq_length()), "committed_sequence_length": len(prefix),
            "terminal_token_not_cached": int(cache.get_seq_length()) == len(prefix) - 1, **stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scenario-id", default="SC-E16-N")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    root = args.project_root.resolve()
    def locate(p):
        return (p if p.is_absolute() else root / p).resolve()
    source, output = locate(args.source_run), locate(args.run_dir)
    base = root / "experiments/results/algorithms"
    if not output.is_relative_to(base) or output == base or output.is_relative_to(source) or args.repeats < 2:
        raise ValueError("new independent output and at least two repeatability measurements required")
    config = read(source / "run.json")["config"]
    source_rows = [r for r in (read_line(line) for line in (source / "results.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip())
                   if r["scenario_id"] == args.scenario_id and r["experiment_version"] == "SD0"]
    if not source_rows:
        raise ValueError("scenario absent from recorded evidence")
    snapshot_path = (source / source_rows[0]["prompt_snapshot"]).resolve()
    if source not in snapshot_path.parents:
        raise ValueError("snapshot escapes source run")
    snapshot = read(snapshot_path)
    if fingerprint(snapshot["input_token_ids"]) != snapshot["prompt_identity"]:
        raise ValueError("saved prompt hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    try:
        import torch
        engine = SDEngine(root, config, with_draft=True)
        if engine.identity["target"] != source_rows[0]["target_identity"] or engine.identity["draft"] != source_rows[0]["draft_identity"]:
            raise ValueError("source model identities differ")
        if engine.tokenizer(snapshot["prompt"], return_tensors="pt", truncation=False)["input_ids"][0].tolist() != snapshot["input_token_ids"]:
            raise ValueError("tokenizer/prompt identity changed")
        write_json(output / "identity.json", {"engine": engine.identity, "source_run": str(source),
            "snapshot": snapshot, "version": "independent_greedy_block_verification_v2",
            "standard_alias": f"SD2-1 has same k as k{config['standard_draft_tokens']}; no duplicate measurement claimed",
            "provisional": True, "formal_qualification": False})
        rows = []
        for repeat in range(args.repeats):
            order = [None, 2, 4, 6, 8]
            order = order[repeat % len(order):] + order[:repeat % len(order)]
            for k in order:
                torch.manual_seed(config["seed"] + repeat)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(config["seed"] + repeat)
                row = {"mode": "SD0-v2" if k is None else f"SD-v2-k{k}", "repeat": repeat,
                       "candidate_limit": k, **decode(engine, snapshot["input_token_ids"], k)}
                rows.append(row)
                append_jsonl(output / "measurements.raw.jsonl", row)
        baseline = {r["repeat"]: r for r in rows if r["mode"] == "SD0-v2"}
        repeatable = len({tuple(r["output_token_ids"]) for r in baseline.values()}) == 1
        for row in rows:
            ref = baseline[row["repeat"]]
            ids, other = row["output_token_ids"], ref["output_token_ids"]
            equal = ids == other
            row.update(exact_token_match_sd0=equal, baseline_repeatable=repeatable,
                status="passed_on_measured_inputs" if equal and repeatable else "failed_lossless",
                first_different_new_token_index=next((i for i in range(min(len(ids), len(other))) if ids[i] != other[i]),
                    min(len(ids), len(other)) if len(ids) != len(other) else None),
                paired_generation_speedup=ref["generation_seconds"] / row["generation_seconds"] if equal and repeatable else None,
                performance_scope="exploratory, no separate warmup; not formal speed evidence")
            append_jsonl(output / "results.jsonl", row)
        # Bounded boundary checks, NOT additional research questions. A shared
        # near-EOS prefix and budgets 1/2/3 test EOS and partial candidate blocks.
        boundary_rows = []
        if len(baseline[0]["output_token_ids"]) < 81:
            raise RuntimeError("near-token-82 boundary prefix unavailable under V2 baseline; do not fabricate coverage")
        boundary_prefix = snapshot["input_token_ids"] + baseline[0]["output_token_ids"][:80]
        old_limit = engine.config["max_new_tokens"]
        try:
            for budget in (1, 2, 3):
                engine.config["max_new_tokens"] = budget
                control = decode(engine, boundary_prefix, None)
                for k in (2, 4, 6, 8):
                    tested = decode(engine, boundary_prefix, k)
                    entry = {"kind": "boundary_only_not_formal_measurement", "budget": budget, "k": k,
                             "prefix_identity": fingerprint(boundary_prefix),
                             "baseline": control, "speculative": tested,
                             "exact_token_match_sd0": control["output_token_ids"] == tested["output_token_ids"]}
                    boundary_rows.append(entry)
                    append_jsonl(output / "boundary_results.jsonl", entry)
        finally:
            engine.config["max_new_tokens"] = old_limit
        engine.verify_sources()
        passed = repeatable and all(r["exact_token_match_sd0"] for r in rows) and all(r["exact_token_match_sd0"] for r in boundary_rows)
        summary = {"status": "passed_on_measured_inputs" if passed else "failed_lossless",
            "provisional": True, "baseline_repeatable": repeatable, "measurements": len(rows),
            "failed": sum(r["status"] == "failed_lossless" for r in rows),
            "boundary_failed": sum(not r["exact_token_match_sd0"] for r in boundary_rows),
            "boundary_tests": len(boundary_rows), "boundary_generation_calls": 15,
            "boundary_coverage": {"k_values": [2, 4, 6, 8], "eos_observed": any(r["stop_reason"] == "eos" for r in rows),
                "candidate_eos_observed": any(r["eos_candidate_seen"] for r in rows),
                "crop_observed": any(r["cache_crops"] for r in rows),
                "token_limit_observed": any(r["stop_reason"] == "token_limit" for r in rows)},
            "limitations": "Unobserved boundaries are unverified. Block-vs-serial floating-point equivalence is not guaranteed; no universal lossless claim or replacement of V1."}
        write_json(output / "summary.json", summary)
        (output / "report.md").write_text("# SD v2 獨立驗收\n\n" + str(summary) + "\n\n未覆寫V1；若仍失敗，停止重做並保留技術限制。\n", encoding="utf-8")
        print(f"SD v2: {summary['status']}: {output}")
        return 0 if passed else 2
    except Exception:
        write_json(output / "failed.json", {"status": "unverifiable", "traceback": traceback.format_exc()})
        raise


def read_line(line):
    import json
    return json.loads(line)


if __name__ == "__main__":
    raise SystemExit(main())
