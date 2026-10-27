"""One isolated GPU process per experiment. Calls existing algorithms, never trains."""
from __future__ import annotations

import argparse
from copy import deepcopy
import random
import time
import traceback
import uuid

from .common import ROOT, Records, path, read, rows, save, sha, verify, fingerprint
from .data import load, review, select
from .adapters import QAEngine, a0_config, sd_config, stable_engine_identity


def error_record(qid, exc, **fields):
    return {"question_id": qid, "status": "failed", "error": str(exc),
            "error_type": type(exc).__name__, "traceback": traceback.format_exc(), **fields}


def cache_trace(pipeline):
    """Observe an existing embedding result by reference, without another forward."""
    adapter = pipeline.adapter
    original = adapter.embed
    def observe(query):
        vector = original(query)
        adapter._benchmark_vector = vector
        return vector
    adapter.embed = observe


def vector_record(pipeline):
    if pipeline.cache.mode != "semantic":
        return None
    return pipeline.adapter._benchmark_vector.tolist()


def restore(pipeline, record):
    """Replay only committed Miss insertions, not retrieval, generation or timing."""
    hit = record.get("cache_hit", record.get("hit"))
    if not hit and record.get("cache_id") is not None:
        cid = pipeline.cache.insert(record.get("question", record.get("query")), record["answer"],
                                    record.get("cache_query_vector"))
        if cid != record["cache_id"]:
            raise ValueError("restored Cache namespace/ID differs; no unsafe resume")


def sd_verdict(result, baseline):
    from experiments.algorithms.speculative_decoding.reporting import token_difference
    if baseline is None:
        result.update(status="failed_missing_sd0", exact_token_match_sd0=None, performance_eligible=False)
        return result
    for field in ("prompt_identity", "retrieval_identity", "target_identity", "seed"):
        if result[field] != baseline[field]:
            raise ValueError("SD paired input/Target differs: " + field)
    equal = result["output_token_ids"] == baseline["output_token_ids"]
    result.update(status="success" if equal else "failed_token_mismatch",
                  exact_token_match_sd0=equal, performance_eligible=equal,
                  error=None if equal else "token IDs differ from paired SD0",
                  token_difference=token_difference(result, baseline),
                  generation_speedup_sd0=baseline["generation_seconds"] / result["generation_seconds"] if equal else None)
    return result


def qa_job(root, config, job, state, canonical, resources, dev, data, smoke_only, allow):
    from experiments.evaluation.wrappers import measured
    engine = QAEngine(job["architecture"], root, config, resources, canonical, allow)
    metadata = engine.metadata
    for i in range(config["warmups"]):
        value = measured(engine.answer, dev[i % len(dev)]["question"])
        if job["architecture"] in {"A3", "A5"} and value.get("output_tokens", 0) < 1:
            raise ValueError("loaded Adapter produced no new tokens; reload inference not verified")
        state.put("smoke_%d" % i, {"phase": "smoke", "warmup": True, "status": "success",
                                   "question_id": dev[i % len(dev)]["id"], "output": value, "model": metadata,
                                   "adapter_reload_inference_verified": job["architecture"] in {"A3", "A5"}})
    if smoke_only:
        return
    label = "FAQ比對" if job["architecture"] in {"A1", "A2-1", "A2-2"} else "LoRA回答" if job["architecture"] == "A3" else "RAG問答"
    cases = select(data, label)
    # Explicit safety allocation is a separate cohort, never mixed into FAQ accuracy.
    cases += [r for r in select(data, "問答安全/不確定性") if r["id"] not in {q["id"] for q in cases}]
    for repeat in range(config["repeats"]):
        for case in cases:
            key = case["id"] + "_r%d" % repeat
            previous = state.get(key)
            if previous and previous["status"] == "success":
                continue
            try:
                value = measured(engine.answer, case["question"])
                result = {"status": "success", "output": value, "end_to_end_seconds": measured.seconds}
            except Exception as e:
                result = error_record(case["id"], e)
            result.update(question_id=case["id"], question=case["question"], type=case["type"],
                          experiment=job["architecture"], repeat=repeat, warmup=False, phase="measured",
                          cohort="allocated_main" if label in case["suites"] else "allocated_safety",
                          model=metadata, quality_status="pending_verified_annotation")
            state.put(key, result)


def sc_job(root, config, job, state, dev, data, smoke_only):
    from experiments.algorithms.shared.a0_adapter import A0Adapter
    from experiments.algorithms.semantic_cache.pipeline import CachedA0
    adapter = A0Adapter(root, a0_config(config))
    smoke = CachedA0(adapter, job["mode"], config["sc_threshold"])
    for i in range(max(2, config["warmups"])):
        state.put("smoke_%d" % i, {"status": "success", "phase": "smoke", "warmup": True,
                                  "output": smoke.answer(dev[0]["question"])})
    if smoke_only:
        adapter.verify_sources()
        return
    for repeat in range(config["repeats"]):
        pipeline = CachedA0(adapter, job["mode"], config["sc_threshold"])
        cache_trace(pipeline)
        for phase in ("cold", "warm"):
            failed = False
            for case in select(data, "語意快取"):
                key = "%s_r%d_%s" % (phase, repeat, case["id"])
                record = state.get(key)
                if record and record["status"] == "success":
                    restore(pipeline, record)
                    continue
                try:
                    result = pipeline.answer(case["question"])
                    result.update(cache_query_vector=vector_record(pipeline), status="success")
                except Exception as e:
                    result = error_record(case["id"], e, cache_failure=getattr(e, "cache_failure", None))
                    failed = True
                result.update(question_id=case["id"], question=case["question"], type=case["type"],
                              experiment="SC-" + job["mode"], phase=phase, repeat=repeat, warmup=False,
                              quality_status="unverified_actual_answer_reuse", initial_cache="empty_before_cold")
                state.put(key, result)
                if failed:
                    break
            if failed:
                break  # do not measure a warm suffix after an invalid cold prefix
    adapter.verify_sources()


def make_sd(root, config, draft, integrated=False):
    from experiments.algorithms.speculative_decoding.engine import SDEngine
    from experiments.algorithms.speculative_decoding.numerics import configure, bind, ensure_memory
    effective = sd_config(config, integrated)
    if effective["precision"] == "fp32":
        configure(effective)
    engine = SDEngine(root, effective, with_draft=draft)
    if effective["precision"] == "fp32":
        bind(engine)
        ensure_memory(engine, effective["max_input_tokens"])
    return engine


def sd_smoke(engine, variants, dev, state, warmups):
    for repeat in range(warmups):
        snapshot, inputs = engine.prepare({"scenario_id": dev[0]["id"], "query": dev[0]["question"]})
        baseline = None
        for version, length in variants:
            result = engine.generate(snapshot, inputs, version, length, repeat, True, engine.config["seed"] + repeat)
            result["measurement_id"] = uuid.uuid4().hex
            if version == "SD0":
                baseline = result
            sd_verdict(result, baseline)
            state.put("smoke_%d_%s" % (repeat, version), dict(result, phase="smoke", experiment=version))
            if result["status"] != "success":
                raise ValueError("development smoke Token Guard failed: " + version)


def sd_job(root, config, job, state, folder, dev, data, smoke_only):
    import torch
    engine = make_sd(root, config, job["draft"])
    variants = [("SD0", None)]
    if job["draft"]:
        variants += [("SD2-1", 4)] + [("SD2-2-k%d" % k, k) for k in (2, 4, 6, 8)]
    else:
        variants += [("SD2-4-k%d" % k, k) for k in (2, 4, 6, 8)]
    save(folder / ("identity_" + uuid.uuid4().hex[:12] + ".json"), engine.identity)
    sd_smoke(engine, variants, dev, state, config["warmups"])
    if smoke_only:
        engine.verify_sources()
        return
    for case_index, case in enumerate(select(data, "推測解碼候選")):
        snapshot_file = folder / "prompts" / (case["id"] + ".json")
        try:
            if snapshot_file.exists():
                snapshot = read(snapshot_file)
                if snapshot["question"] != case["question"]:
                    raise ValueError("saved prompt/source question mismatch")
                receipt = read(snapshot_file.with_suffix(".receipt.json"))
                if receipt["sha256"] != sha(snapshot_file) or fingerprint(snapshot["input_token_ids"]) != snapshot["prompt_identity"]:
                    raise ValueError("saved Prompt snapshot integrity changed")
                inputs = {"input_ids": torch.tensor([snapshot["input_token_ids"]], device=config["device"])}
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            else:
                snapshot, inputs = engine.prepare({"scenario_id": case["id"], "query": case["question"]})
                save(snapshot_file, snapshot)
                save(snapshot_file.with_suffix(".receipt.json"), {"sha256": sha(snapshot_file)})
            if state.get(case["id"] + "_prepare"):
                state.put(case["id"] + "_prepare", {"question_id": case["id"], "status": "success", "phase": "preparation"})
        except Exception as e:
            state.put(case["id"] + "_prepare", error_record(case["id"], e, phase="preparation", experiment=job["id"]))
            continue
        for repeat in range(config["repeats"]):
            baseline = state.get("%s_r%d_SD0" % (case["id"], repeat))
            # SD0 first establishes pairing. Rotate non-baseline arms to avoid k-order bias.
            arms = variants[1:]
            offset = (case_index + repeat) % len(arms)
            sequence = variants[:1] + arms[offset:] + arms[:offset]
            for version, length in sequence:
                key = "%s_r%d_%s" % (case["id"], repeat, version)
                previous = state.get(key)
                if previous and previous.get("output_token_ids"):
                    if version == "SD0":
                        baseline = previous
                    continue  # measured mismatch is evidence, never silently retried/deleted
                try:
                    result = engine.generate(snapshot, inputs, version, length, repeat, False, config["seed"] + repeat)
                    result["measurement_id"] = uuid.uuid4().hex
                    if version == "SD0":
                        baseline = result
                    result = sd_verdict(result, baseline)
                except Exception as e:
                    result = error_record(case["id"], e)
                result.update(question_id=case["id"], question=case["question"], type=case["type"],
                              experiment=version, repeat=repeat, warmup=False, phase="measured",
                              study=job["id"], quality_status="pending_verified_annotation",
                              prompt_snapshot=str(snapshot_file.relative_to(folder)))
                state.put(key, result)
    engine.verify_sources()


def int_case(case, order):
    return {"scenario_id": case["id"], "question": case["question"], "query_order": order,
            "review_status": "needs_review", "reference_scenario_id": None,
            "expected_answer_reuse": "unverified", "history": []}


def int_job(root, config, job, state, folder, dev, data, smoke_only):
    from experiments.algorithms.integrated_efficiency.pipeline import IntegratedPipeline
    engine = make_sd(root, config, job["decoder"] == "draft_1b", integrated=True)
    save(folder / ("identity_" + uuid.uuid4().hex[:12] + ".json"), engine.identity)
    stable_engine_identity(engine)
    for variant in ("INT0", "INT1", "INT2", "INT3"):
        pipeline = IntegratedPipeline(engine, variant, job["decoder"], 0, True, config["sc_threshold"])
        result = pipeline.answer(int_case(dev[0], 1))
        state.put("smoke_" + variant, dict(result, phase="smoke", warmup=True))
    if smoke_only:
        engine.verify_sources()
        return
    for repeat in range(config["repeats"]):
        variants = ["INT0", "INT1", "INT2", "INT3"]
        random.Random(config["seed"] + repeat).shuffle(variants)
        for variant in variants:
            integrated = IntegratedPipeline(engine, variant, job["decoder"], repeat, False, config["sc_threshold"])
            cache_trace(integrated.pipeline)
            failed = False
            for phase in ("cold", "warm"):
                for i, source in enumerate(select(data, "語意快取"), 1):
                    case = int_case(source, i)
                    key = "%s_%s_r%d_%s" % (variant, phase, repeat, source["id"])
                    previous = state.get(key)
                    if previous and previous["status"] == "success":
                        restore(integrated.pipeline, previous)
                        if not previous["cache_hit"] and previous["cache_id"] is not None:
                            integrated.origins[previous["cache_id"]] = {"case": case, "payload": {
                                "snapshot": previous["prompt_snapshot"], "generation": previous["decoding_measurement"]}}
                        continue
                    try:
                        result = integrated.answer(case)
                        result["cache_query_vector"] = vector_record(integrated.pipeline)
                    except Exception as e:
                        result = error_record(source["id"], e)
                        failed = True
                    result.update(question_id=source["id"], question=source["question"], type=source["type"],
                                  experiment=variant, decoder=job["decoder"], phase=phase, repeat=repeat, warmup=False,
                                  quality_status="unverified_actual_answer_reuse", fill_policy="cold pass fills this variant only; warm excludes cold cost")
                    state.put(key, result)
                    if failed:
                        break
                if failed:
                    break
    engine.verify_sources()


def ti_job(root, config, state, dev, data, annotations, smoke_only, allow_provisional=False):
    from experiments.algorithms.shared.a0_adapter import A0Adapter
    from experiments.algorithms.topic_identification.features import TopicFeatures
    adapter = A0Adapter(root, a0_config(config))
    settings = dict(a0_config(config), max_attention_tokens=2048)
    features = TopicFeatures(adapter, settings)
    sample = {"history": [{"role": "user", "content": dev[0]["question"]}], "query": dev[1]["question"]}
    state.put("smoke_features", {"status": "success", "phase": "smoke", "warmup": True,
                                "embedding": features.embedding(sample), "attention": features.attention(sample),
                                "gold_status": "no_score_development_smoke"})
    if smoke_only:
        adapter.verify_sources()
        return
    from experiments.algorithms.semantic_cache.cache import fingerprint as ti_fingerprint
    from experiments.algorithms.topic_identification.context import POLICY_VERSION, DEPENDENCY_PATTERN
    frozen = read(path(root, config["ti_thresholds"]))
    payload = frozen["payload"]
    if payload.get("provisional") and not allow_provisional:
        raise ValueError("provisional TI calibration requires explicit --allow-provisional; not Formal")
    calibration_dir = path(root, config["ti_thresholds"]).parent
    completion = read(calibration_dir / "completed.json")
    calibration = rows(calibration_dir / "calibration.snapshot.json")
    from .data import normalize
    test_questions = {normalize(r["question"]) for r in data}
    if (completion.get("status") != "completed" or completion.get("source_verification") != "passed"
            or any(normalize(c["query"]) in test_questions for c in calibration)):
        raise ValueError("TI calibration incomplete or contains 804 test questions; no test-set tuning")
    identity = {"a0": adapter.identity, "features": features.definition, "max_attention_tokens": 2048,
                "context_policy": POLICY_VERSION, "dependency_pattern": DEPENDENCY_PATTERN,
                "effective_attention_backend_all_arms": "eager"}
    if (ti_fingerprint(payload) != frozen["payload_sha256"] or payload["identity_id"] != ti_fingerprint(identity)):
        raise ValueError("frozen TI model/feature/history protocol differs; no fitting on 804")
    for cid, annotation in annotations["conversations"].items():
        if annotation["history_scope"] == "previous_user_questions_only" and payload.get("history_scope") != "previous_user_questions_only":
            raise ValueError("question-only history requires independent question-only calibration, not legacy full-message thresholds")
        turns = [r for r in data if r["conversation_id"] == cid]
        for index, case in enumerate(turns[1:], 1):
            for repeat in range(config["repeats"]):
                key = "%s_%s_r%d" % (cid, case["id"], repeat)
                if state.get(key) and state.get(key)["status"] == "success":
                    continue
                history = (annotation["histories"][case["id"]] if annotation["history_scope"] == "full_reviewed_messages"
                           else [{"role": "user", "content": t["question"]} for t in turns[:index]])
                sample = {"history": history,
                          "query": case["question"]}
                try:
                    embedding, attention = features.embedding(sample), features.attention(sample)
                    scores = dict(embedding["scores"], **attention["scores"])
                    result = {"status": "success", "scores": scores, "history": sample["history"],
                              "embedding": embedding, "attention": attention,
                              "gold_shift": annotation["shift_labels"][case["id"]],
                              "predictions": {k: s < payload["methods"][k]["threshold"] for k, s in scores.items()},
                              "thresholds": payload["methods"]}
                except Exception as e:
                    result = error_record(case["id"], e)
                result.update(question_id=case["id"], question=case["question"], type=case["type"],
                              experiment="TI", phase="measured", repeat=repeat, warmup=False, conversation_id=cid)
                state.put(key, result)
    adapter.verify_sources()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    root, run = path(ROOT, args.project_root), path(ROOT, args.run_dir)
    identity = read(run / "raw/identity.json")
    verify(identity)
    config = read(run / "raw/config.json")
    data, _ = load(root)
    annotations = review(root, config, data)
    plan = read(run / "raw/plan.json")
    job = next(j for j in plan if j["id"] == args.job)
    folder = run / "raw/jobs" / job["id"]
    state = Records(folder / "records")
    dev = read(run / "raw/development.json")
    try:
        if job["kind"] == "qa":
            qa_job(root, config, job, state, read(run / "raw/faq_corpus.json"),
                   read(run / "raw/resources.json"), dev, data, args.smoke_only, identity["allow_provisional"])
        elif job["kind"] == "sc":
            sc_job(root, config, job, state, dev, data, args.smoke_only)
        elif job["kind"] == "sd":
            sd_job(root, config, job, state, folder, dev, data, args.smoke_only)
        elif job["kind"] == "int":
            int_job(root, config, job, state, folder, dev, data, args.smoke_only)
        elif job["kind"] == "ti":
            ti_job(root, config, state, dev, data, annotations, args.smoke_only, identity["allow_provisional"])
        verify(identity)
        failures = [r for r in state.all() if (not args.smoke_only or r.get("phase") == "smoke")
                    and r.get("status") not in {"success", "generated_pending_comparison"}]
        status = {"status": "failed" if failures else "smoke_passed" if args.smoke_only else "completed_measurements",
                  "failure_count": len(failures), "records": len(state.index),
                  "error": failures[0].get("error") if failures else None,
                  "first_failed_question": failures[0].get("question_id") if failures else None,
                  "formal_quality_certified": False}
        save(folder / "status.json", status, replace=True)
        return 2 if failures else 0
    except Exception as exc:
        record = error_record(None, exc, stage="worker_initialization_or_execution", job=job["id"])
        save(folder / ("failure_" + uuid.uuid4().hex[:12] + ".json"), record)
        save(folder / "status.json", {"status": "failed", **record}, replace=True)
        print(traceback.format_exc(), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
