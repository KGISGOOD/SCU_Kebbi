"""Independent A0 API; no GPU imports or resource loads until server lifespan."""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Optional
import uuid

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictStr


ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("baseline_api")


class InputRejected(ValueError):
    """Only the existing A0 length guards map to a client input error."""


class ChatRequest(BaseModel):
    question: StrictStr = Field(..., min_length=1, max_length=4000)

    class Config:
        extra = "forbid"  # V1 has no conversation/history input.


class ChatResponse(BaseModel):
    answer: str
    success: bool
    error: Optional[str] = None
    request_id: str
    answer_source: Optional[str] = None
    faq_top1_score: Optional[float] = None
    faq_top2_score: Optional[float] = None
    faq_accepted: Optional[bool] = None
    faq_rejection_reason: Optional[str] = None
    faq_question_id: Optional[str] = None
    latency_seconds: Optional[float] = None
    faq_acceptance_mode: Optional[str] = None
    faq_acceptance_reason: Optional[str] = None
    faq_human_verified: Optional[bool] = None
    faq_score_margin: Optional[float] = None


def load_settings(config_path):
    path = Path(config_path).expanduser()
    path = (path if path.is_absolute() else ROOT / path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    expected = {"model", "bge", "knowledge_index", "device", "precision", "seed",
                "max_input_tokens", "max_new_tokens", "request_timeout_seconds"}
    if not isinstance(config, dict) or set(config) - {"prompt_version", "faq"} != expected:
        raise ValueError("use baseline_api/config.example.json schema")
    from experiments.baselines.rag_baseline.prompts import PROMPT_VERSION, validate_prompt_version
    validate_prompt_version(config.get("prompt_version", PROMPT_VERSION))
    if 'faq' in config:
        from baseline_api.faq import validate_faq_config
        validate_faq_config(config['faq'])
        if config.get('prompt_version') != 'baseline_senior_v2':
            raise ValueError('Baseline 2 requires baseline_senior_v2')
    for key in ("model", "bge", "knowledge_index"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError("explicit local resource path required: " + key)
    for key in ("seed", "max_input_tokens", "max_new_tokens"):
        if type(config[key]) is not int or config[key] < (0 if key == "seed" else 1):
            raise ValueError("invalid " + key)
    if config["device"] not in {"cuda", "cpu"} or config["precision"] not in {"fp16", "bf16", "fp32"}:
        raise ValueError("invalid explicit device/precision; no automatic fallback")
    if config["device"] == "cpu" and config["precision"] != "fp32":
        raise ValueError("CPU requires explicit fp32")
    timeout = config["request_timeout_seconds"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 600:
        raise ValueError("request_timeout_seconds must be finite and in (0, 600]")
    return config


def failure(status, error, request_id, retry=False):
    return JSONResponse(status_code=status,
                        content={"answer": "", "success": False, "error": error, "request_id": request_id},
                        headers={"Retry-After": "2"} if retry else None)


def create_app(config_path, run_dir=None):
    config = load_settings(config_path)
    from experiments.baselines.rag_baseline.prompts import PROMPT_VERSION
    selected_prompt = config.get("prompt_version", PROMPT_VERSION)

    @asynccontextmanager
    async def lifespan(app):
        state = app.state
        state.adapter = None
        state.active = None
        state.status = "loading"
        state.closing = False
        state.records_path = None
        if run_dir is not None:
            raw = Path(run_dir).expanduser()
            output = (raw if raw.is_absolute() else ROOT / raw).resolve()
            results_root = (ROOT / "experiments/results").resolve()
            if results_root not in output.parents:
                raise ValueError("--run-dir must be a NEW child of experiments/results")
            output.mkdir(parents=True, exist_ok=False)
            def write_new(name, value):
                with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
                    json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write("\n")
            write_new("run.json", {
                "schema": "a0_api_session_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": config, "prompt_version": selected_prompt,
                "server_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "formal_quality_certified": False,
                "timing_scope": "request admission to inference completion; excludes model loading and record writing",
                "character_count": "Python len(answer), including punctuation, whitespace and newlines",
                "record_scope": "inference completion; a timed-out HTTP request may complete later",
            })
            state.records_path = output / "results.jsonl"
            with state.records_path.open("x", encoding="utf-8"):
                pass
            LOG.info("API session output=%s", output)
        # One worker, and admission control before submitting: no inference queue.
        state.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="a0")

        async def load_once():
            try:
                def construct():
                    from experiments.shared.a0_adapter import A0Adapter
                    adapter = A0Adapter(ROOT, config)
                    if 'faq' in config:
                        from baseline_api.faq import FAQFallback
                        adapter = FAQFallback(ROOT, config, adapter)
                    if adapter.identity["prompt_version"] != selected_prompt or adapter.pipeline.prompt_version != selected_prompt:
                        raise ValueError("loaded A0 prompt differs from selected API configuration")
                    if state.records_path is not None:
                        write_new("identity.json", adapter.identity)
                    return adapter
                adapter = await asyncio.get_running_loop().run_in_executor(state.pool, construct)
                if not state.closing:
                    state.adapter, state.status = adapter, "ready"
                    LOG.info("A0 ready pid=%s prompt=%s generation=%s",
                             os.getpid(), adapter.identity["prompt_version"], adapter.identity["generation"])
            except asyncio.CancelledError:
                raise
            except Exception:
                state.status = "failed"
                LOG.exception("A0 startup failed; fix resources and restart API")

        loading = asyncio.create_task(load_once())
        LOG.info("API pid=%s; loading local A0 once; HTTP health available while loading", os.getpid())
        try:
            yield
        finally:
            state.closing, state.status = True, "stopping"
            if not loading.done():
                loading.cancel()
                try:
                    await loading
                except asyncio.CancelledError:
                    pass
            # Threads/CUDA cannot be forcibly cancelled safely. Do not unload a
            # model while an outstanding forward still uses it.
            if state.active is not None and not state.active.done():
                LOG.warning("Shutdown has outstanding inference; GPU thread may need to finish before process exits")
            # Admission control leaves no queued requests to cancel.
            state.pool.shutdown(wait=False)

    app = FastAPI(title="資科萬事通 A0 Baseline API", version="1.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        request_id = uuid.uuid4().hex
        LOG.warning("request=%s validation_error", request_id)
        return failure(422, "invalid_request", request_id)

    @app.get("/health")
    async def health():
        state = app.state
        ready = state.status == "ready" and not state.closing
        busy = state.active is not None
        return JSONResponse(status_code=200 if ready else 503,
                            content={"service": "a0_baseline", "status": state.status,
                                     "prompt_version": selected_prompt,
                                     "loaded_prompt_version": state.adapter.identity["prompt_version"] if ready else None,
                                     "models_ready": ready, "busy": busy,
                                     "accepting_requests": ready and not busy})

    @app.post("/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest):
        request_id = uuid.uuid4().hex
        if not body.question.strip():
            return failure(422, "empty_question", request_id)
        state = app.state
        if state.status != "ready" or state.closing:
            return failure(503, "model_not_ready", request_id, retry=True)
        if state.active is not None:
            return failure(503, "busy", request_id, retry=True)

        start = time.perf_counter()
        started_utc = datetime.now(timezone.utc).isoformat()

        def record_outcome(answer=None, timings=None, error=None):
            if state.records_path is None:
                return
            record = {
                "request_id": request_id, "started_utc": started_utc, "question": body.question,
                "prompt_version": state.adapter.identity["prompt_version"],
                "status": "success" if error is None else "failed",
                "answer": answer, "answer_char_count": len(answer) if answer is not None else None,
                "exceeds_100_characters": len(answer) > 100 if answer is not None else None,
                "elapsed_seconds": time.perf_counter() - start,
                "timings": timings, "error": str(error) if error is not None else None,
                "error_type": type(error).__name__ if error is not None else None,
            }
            # The sole inference worker writes before completing its future.
            # Recording failure therefore cannot masquerade as a successful call.
            with state.records_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

        def infer():
            try:
                try:
                    answer, timings = state.adapter.answer(body.question)
                except ValueError as exc:
                    if str(exc).startswith(("query exceeds BGE experiment limit", "full A0 prompt has", "full prompt has")):
                        raise InputRejected(str(exc)) from exc
                    raise
                if not isinstance(answer, str) or not answer.strip():
                    raise RuntimeError("A0 returned an empty answer")
                if config.get("prompt_version") == "baseline_senior_v2" and len(answer) > 100:
                    # Observe compliance only: never slice an answer or drop conditions.
                    LOG.warning("request=%s answer_length=%s exceeds 100 characters; answer preserved",
                                request_id, len(answer))
            except Exception as exc:
                record_outcome(error=exc)
                raise
            record_outcome(answer=answer, timings=timings)
            return answer, timings

        # No await between busy check and submission: admission is atomic on
        # this app's event loop. A timed-out HTTP request never releases a still
        # executing GPU worker to another request.
        future = asyncio.get_running_loop().run_in_executor(state.pool, infer)
        state.active = future

        def finished(result):
            try:
                _, timings = result.result()
                LOG.info("request=%s inference_done elapsed=%.3f retrieval=%s generation=%s input_tokens=%s",
                         request_id, time.perf_counter() - start, timings["retrieval_seconds"],
                         timings["generation_seconds"], timings["input_tokens"])
            except asyncio.CancelledError:
                LOG.warning("request=%s cancelled during API shutdown", request_id)
            except InputRejected:
                LOG.warning("request=%s rejected by A0 input guard", request_id, exc_info=True)
            except Exception:
                if not state.closing:
                    state.status = "failed"  # OOM/device faults require explicit restart.
                LOG.exception("request=%s inference_failed", request_id)
            finally:
                if state.active is result:
                    state.active = None

        future.add_done_callback(finished)
        done, _ = await asyncio.wait({future}, timeout=config["request_timeout_seconds"])
        if not done:
            LOG.warning("request=%s HTTP timeout; inference remains busy until GPU returns", request_id)
            return failure(504, "inference_timeout", request_id)
        try:
            answer, timings = future.result()
        except InputRejected:
            return failure(422, "input_limit_exceeded", request_id)
        except Exception:
            return failure(500, "inference_failed", request_id)
        routing = {key: timings.get(key) for key in ('answer_source', 'faq_top1_score', 'faq_top2_score',
                   'faq_accepted', 'faq_rejection_reason', 'faq_question_id', 'faq_acceptance_mode',
                   'faq_acceptance_reason', 'faq_human_verified', 'faq_score_margin')}
        routing['answer_source'] = routing['answer_source'] or 'rag'
        return ChatResponse(answer=answer, success=True, request_id=request_id,
                            latency_seconds=time.perf_counter() - start, **routing)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="GPU-local JSON, paths relative to project root")
    parser.add_argument("--run-dir", help="optional NEW child of experiments/results for identity and per-question records")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    import uvicorn
    # Single process only: multiple workers/reload would duplicate GPU models.
    uvicorn.run(create_app(args.config, args.run_dir), host=args.host, port=args.port, workers=1, reload=False)


if __name__ == "__main__":
    main()
