"""Explicit human Train approval, separate from evaluation Gold. No model imports."""
from datetime import datetime

from experiments.a3_faq_sft.common import DataError, read_json, sha


def verify(path, schema, train, manifest, ids):
    before = sha(path)
    value = read_json(path)
    approved = value.get("approved_faq_ids")
    if (value.get("schema_version") != schema
            or value.get("train_sha256") != sha(train)
            or value.get("dataset_manifest_sha256") != sha(manifest)
            or not isinstance(approved, list) or any(not isinstance(i, str) for i in approved)
            or len(approved) != len(ids) or set(approved) != set(ids)
            or any(not isinstance(value.get(k), str) or not value[k].strip()
                   for k in ("reviewer", "reviewed_at", "evidence"))):
        raise DataError("Train approval requires exact IDs, raw-byte hashes and genuine reviewer/time/evidence")
    if datetime.fromisoformat(value["reviewed_at"].replace("Z", "+00:00")).tzinfo is None:
        raise DataError("approval reviewed_at requires timezone")
    decisions = value.get("decisions")
    if (not isinstance(decisions, list) or len(decisions) != len(ids)
            or any(not isinstance(d, dict) or not isinstance(d.get("faq_id"), str)
                   or d.get("decision") != "approved"
                   or not isinstance(d.get("evidence"), str) or not d["evidence"].strip()
                   for d in decisions)
            or {d["faq_id"] for d in decisions} != set(ids)):
        raise DataError("every Train ID needs an explicit approved decision with applicable evidence")
    if sha(path) != before:
        raise DataError("approval changed during read")
    return value, before


def template(schema, train, manifest, ids):
    return {"schema_version": schema, "train_sha256": sha(train),
            "dataset_manifest_sha256": sha(manifest), "approved_faq_ids": [],
            "reviewer": "", "reviewed_at": "", "evidence": "",
            "decisions": [{"faq_id": i, "decision": "needs_review", "evidence": ""} for i in ids]}
