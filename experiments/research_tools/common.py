"""Portable, strict IO and immutable dataset identity. Standard library only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from experiments.datasets.unified_research_v1.validation import read_json, read_rows, sha

ROOT = Path(__file__).resolve().parents[2]
DATA = "experiments/datasets/unified_research_v1"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def tool_identity():
    return {"python": sys.version,
            "code_sha256": {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}


def locate(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_rows(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_dataset(root, name="unified_research_v1"):
    if name == "experiment_suites_v1":
        from .suites import load
        return load(root)
    if name != "unified_research_v1":
        raise ValueError("unknown research dataset")
    directory = root / DATA
    manifest = read_json(directory / "dataset_manifest.json")
    hashes = {"dataset_manifest.json": sha(directory / "dataset_manifest.json")}
    for name, expected in manifest["artifact_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or sha(path) != expected:
            raise ValueError(f"dataset artifact changed/missing: {name}; do not silently merge stale reviews")
        hashes[name] = expected
    registry = read_json(directory / "source_registry.json")
    checked = {}
    for key, source in registry["sources"].items():
        path = locate(root, source["path"])
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"source missing/outside project: {key}")
        if str(path) not in checked:
            checked[str(path)] = sha(path)
        actual = checked[str(path)]
        if actual != source["sha256"]:
            raise ValueError(f"source hash mismatch {key}: expected={source['sha256']} actual={actual}")
    questions = sum((read_rows(directory / f"questions_{split}.jsonl")
                     for split in ("historical", "calibration", "evaluation")), [])
    if len({r["question_id"] for r in questions}) != len(questions):
        raise ValueError("duplicate question IDs")
    return {"identity": fingerprint(hashes), "hashes": hashes, "questions": questions,
            "dataset_name": "unified_research_v1",
            "turns": read_rows(directory / "conversations.jsonl"),
            "workloads": read_rows(directory / "cache_workloads.jsonl"),
            "fills": read_rows(directory / "cache_initial_fill.jsonl"),
            "relations": read_rows(directory / "question_relations.jsonl"), "registry": registry}


def new_output(root, value):
    path = locate(root, value)
    # No writes into datasets, code, model/index resources or input review bundles.
    if not path.is_relative_to(root / "experiments/results/research_tools"):
        raise ValueError("output must be a NEW directory below experiments/results/research_tools")
    path.mkdir(parents=True, exist_ok=False)
    return path


def verify_bundle(directory, manifest_name):
    if (directory / "failed.json").exists() or read_json(directory / "completed.json").get("status") != "completed":
        raise ValueError("cannot use a partial/failed bundle")
    manifest = read_json(directory / manifest_name)
    for name, expected in manifest["files_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or sha(path) != expected:
            raise ValueError(f"bundle file changed: {name}")
    return manifest
