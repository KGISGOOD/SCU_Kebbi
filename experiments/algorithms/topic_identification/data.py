"""Strict scripted-dialogue input validation and split separation."""
from __future__ import annotations

import json
from experiments.algorithms.semantic_cache.cache import fingerprint, normalize_question


def case_signature(case):
    return fingerprint({"history": [{"role": m["role"], "content": normalize_question(m["content"])}
                                     for m in case["history"]], "query": normalize_question(case["query"])})


def load_cases(path, split, allow_unreviewed):
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank JSONL line")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        for key in ("scenario_id", "family_id", "query", "category", "split", "review_status"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{path}:{line_number}: invalid {key}")
        if row["split"] != split or type(row.get("gold_shift")) is not int or row["gold_shift"] not in (0, 1):
            raise ValueError(f"{row['scenario_id']}: invalid split/gold_shift")
        if row["review_status"] not in {"needs_review", "approved"}:
            raise ValueError("unknown review_status")
        if row["review_status"] != "approved" and not allow_unreviewed:
            raise ValueError("dialogue labels need human review; --allow-unreviewed is development only")
        history = row.get("history")
        if not isinstance(history, list) or not history or len(history) % 2:
            raise ValueError(f"{row['scenario_id']}: complete user/assistant history pairs required")
        for i, turn in enumerate(history):
            if not isinstance(turn, dict) or turn.get("role") != ("user" if i % 2 == 0 else "assistant"):
                raise ValueError("history must alternate user/assistant, ending with assistant")
            for key in ("turn_id", "content"):
                if not isinstance(turn.get(key), str) or not turn[key].strip():
                    raise ValueError(f"invalid history {key}")
        ids = [m["turn_id"] for m in history]
        required = row.get("required_history_ids")
        if (len(ids) != len(set(ids)) or not isinstance(required, list)
                or any(not isinstance(k, str) or k not in ids for k in required)
                or len(required) != len(set(required))):
            raise ValueError("invalid turn IDs or required_history_ids")
        for content in [row["query"]] + [m["content"] for m in history]:
            if "<|" in content or "__TI_CONTENT_" in content:
                raise ValueError("dialogue content cannot contain template control tokens/markers")
        cases.append(row)
    if (not cases or len({r["scenario_id"] for r in cases}) != len(cases)
            or len({case_signature(r) for r in cases}) != len(cases)):
        raise ValueError("empty or duplicate scenarios")
    if {r["gold_shift"] for r in cases} != {0, 1}:
        raise ValueError("both Same Topic and Topic Shift are required")
    return cases


def split_manifest(cases):
    return {"scenario_ids": [r["scenario_id"] for r in cases],
            "family_ids": sorted({r["family_id"] for r in cases}),
            "case_signatures": [case_signature(r) for r in cases],
            "query_signatures": [fingerprint(normalize_question(r["query"])) for r in cases]}


def check_separation(calibration_manifest, evaluation):
    current = split_manifest(evaluation)
    for key in ("scenario_ids", "family_ids", "case_signatures", "query_signatures"):
        if set(current[key]) & set(calibration_manifest[key]):
            raise ValueError(f"calibration/evaluation overlap detected: {key}")
