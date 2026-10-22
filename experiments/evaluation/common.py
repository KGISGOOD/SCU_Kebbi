"""Stdlib data contracts, immutable atomic records and process-safe run locks."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import uuid
from importlib import metadata

from experiments.a3_faq_sft.common import (ROOT, DataError, read_json, read_rows,
                                           resolve, sha, snapshot)

ARCHITECTURES = ("A0", "A1", "A2-1", "A2-2", "A3", "A4-1", "A4-2", "A5")
FAQ_ARCHITECTURES = ("A1", "A2-1", "A2-2")
SMOKE_IDS = ("P01", "P22", "P26")


def identifier():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def read_run(run):
    meta = read_json(run / "run.json")
    if meta.get("run_contract_sha256") != digest({k: v for k, v in meta.items() if k != "run_contract_sha256"}):
        raise DataError("run.json changed; do not edit a frozen run contract")
    return meta


def atomic_json(path, value):
    """Never append/replace a result: hard-link a fully fsynced unique temp file.

    Same-filesystem hard links are supported by NTFS and normal POSIX filesystems.
    If not supported, fail clearly instead of silently weakening atomicity.
    Orphan .tmp files after a crash are ignored; final *.json is always complete.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)  # exclusive destination: EEXIST instead of overwrite
    finally:
        if temp.exists():
            temp.unlink()  # only this call's newly created temporary file


@contextmanager
def lock(path):
    """OS lock releases on process death; persistent lock file needs no deletion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def dataset(root, manifest):
    rows, sources = snapshot(root, resolve(root, manifest))
    result = [{"faq_id": row["faq_id"], "question": row["messages"][1]["content"],
               "reference_answer": row["messages"][2]["content"], "split": row["split"],
               "source": row["source_file"], "source_row": row["source_row"]} for row in rows]
    if len(result) != 60 or len({r["faq_id"] for r in result}) != 60:
        raise DataError("expected exactly 60 unique original FAQ rows")
    return result, sources


def environment():
    names = ("torch", "transformers", "sentence-transformers", "peft", "numpy", "faiss-cpu",
             "faiss-gpu", "langchain-core", "langchain-community", "langchain-text-splitters",
             "huggingface-hub", "accelerate", "jieba", "packaging")
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "host": platform.node(), "packages": versions}


def code_hashes(root):
    result = {}
    for name in ("shared", "baselines", "a2_faq_finetuning", "a3_faq_sft",
                 "a5_rag_finetuned_llm", "evaluation"):
        for path in sorted((root / "experiments" / name).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json"):
                result[path.relative_to(root).as_posix()] = sha(path)
    return result


def read_attempts(run):
    records = []
    meta = read_run(run)
    questions = {q["faq_id"]: q for q in meta["dataset"]}
    for path in sorted((run / "attempts").glob("*/*.json")):
        record = read_json(path)
        if (record.get("architecture") not in ARCHITECTURES or
                record.get("status") not in ("success", "error") or not record.get("attempt_id")
                or type(record.get("attempt_number")) is not int or record["attempt_number"] < 1):
            raise DataError("invalid committed attempt: %s" % path)
        if record.get("record_sha256") != digest({k: v for k, v in record.items() if k != "record_sha256"}):
            raise DataError("committed attempt content changed: %s" % path)
        source = questions.get(record.get("faq_id"))
        if (record.get("run_id") != meta["run_id"] or record["architecture"] not in meta["architectures"]
                or not source or any(record.get(k) != v for k, v in source.items())):
            raise DataError("attempt does not belong to this run/question snapshot: %s" % path)
        records.append(record)
    return records


def model_fields(meta, architecture):
    key = {"A1": "bge", "A2-1": "a2_1_model", "A2-2": "a2_2_model"}.get(architecture, "base")
    resource = meta["resources"][key]
    value = resource.get("value")
    name_key = {"bge": "bge_model", "a2_1_model": "a2_1_model", "a2_2_model": "a2_2_model"}.get(key, "base_model")
    provenance = value.get("provenance") if value else None
    if architecture in ("A3", "A5"):
        adapter = meta["resources"]["a3_adapter"].get("value")
        provenance = adapter.get("provenance") if adapter else None
    return {"model_name": meta["config"][name_key], "model_version_or_path": value,
            "model_provisional": provenance.get("provisional") if provenance else None}


def latest_results(run):
    latest = {}
    for row in sorted(read_attempts(run), key=lambda r: (r["architecture"], r["faq_id"], r["attempt_number"])):
        key = (row["architecture"], row["faq_id"])
        if key in latest and latest[key]["attempt_number"] == row["attempt_number"]:
            raise DataError("duplicate attempt number: %r" % (key,))
        if key in latest and latest[key]["status"] == "success":
            raise DataError("duplicate attempt after successful completion: %r" % (key,))
        latest[key] = row
    return latest


def selected_questions(run_meta):
    ids = run_meta["question_ids"]
    by_id = {row["faq_id"]: row for row in run_meta["dataset"]}
    return [by_id[faq_id] for faq_id in ids]


def pending(run, run_meta, architecture, retry_failed=False):
    latest = latest_results(run)
    return [row for row in selected_questions(run_meta)
            if (architecture, row["faq_id"]) not in latest or
            (retry_failed and latest[(architecture, row["faq_id"])]["status"] == "error")]


def save_result(run, run_meta, architecture, question, payload):
    import time
    attempt_id = uuid.uuid4().hex
    directory = run / "attempts" / architecture
    prior = [read_json(p) for p in directory.glob(question["faq_id"] + "_*.json")]
    if any(row["status"] == "success" for row in prior):
        raise DataError("successful answer already committed; use a NEW run to regenerate")
    number = max((row["attempt_number"] for row in prior), default=0) + 1
    record = {"run_id": run_meta["run_id"], "architecture": architecture,
              **question, "attempt_id": attempt_id, "attempt_number": number, "created_ns": time.time_ns(),
              "generated_answer": "", "status": "error", "error_message": None,
              "latency_seconds": None, "model_name": None, "model_version_or_path": None,
              "model_provisional": None, "input_tokens": None, "output_tokens": None,
              "stop_reason": None, "reached_token_limit": None, **payload}
    if record["status"] == "success" and not record["generated_answer"].strip():
        raise DataError("empty answer must not be committed as success")
    if architecture in FAQ_ARCHITECTURES:
        record["faq_id_correct"] = (record.get("predicted_faq_id") == question["faq_id"]
                                    if record["status"] == "success" else None)
    record["record_sha256"] = digest(record)
    atomic_json(directory / ("%s_%04d_%s.json" % (question["faq_id"], number, attempt_id)), record)
    return record
