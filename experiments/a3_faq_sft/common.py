"""A3 standard-library IO and immutable source validation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "experiments/datasets/faq_finetuning/faq_split_manifest.json"
DATA = "experiments/datasets/faq_sft"
BASE = "meta-llama/Llama-3.2-3B-Instruct"
SYSTEM = "你是東吳大學資料科學系的智慧問答助理。請使用繁體中文回答；不知道時請明確說明，不要編造資訊。"
SOURCES = (
    ("parents", "P", "畢業專題_爬蟲/type2/資料擴充 - 家長 FAQ 資料集（30 題）.csv"),
    ("highschool", "S", "畢業專題_爬蟲/type2/資料擴充 - 高中生 FAQ 資料集（30 題）.csv"),
)


class DataError(ValueError):
    pass


def resolve(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def _constant(value):
    raise DataError("nonstandard JSON value: %s" % value)


def decode(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_constant)


def read_json(path):
    value = decode(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise DataError("expected JSON object: %s" % path)
    return value


def read_rows(path):
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line, text in enumerate(handle, 1):
            try:
                value = decode(text)
                if not isinstance(value, dict):
                    raise DataError("expected object")
                rows.append(value)
            except ValueError as exc:
                raise DataError("%s:%d: %s" % (path, line, exc)) from exc
    return rows


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hashes(directory, exclude=()):
    return {p.relative_to(directory).as_posix(): sha(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p.relative_to(directory).as_posix() not in exclude}


def snapshot(root, manifest):
    """Only this function reads CSV answers; never called by inference."""
    manifest_key = manifest.relative_to(root).as_posix()  # require portable in-project source
    inputs = [relative for _, _, relative in SOURCES] + [manifest_key]
    before = {name: sha(root / name) for name in inputs}
    split = read_json(manifest)["split_scheme"]
    expected = {
        "train": {"P%02d" % n for n in range(1, 22)} | {"S%02d" % n for n in range(1, 22)},
        "validation": {"P%02d" % n for n in range(22, 26)} | {"S%02d" % n for n in range(22, 27)},
        "test": {"P%02d" % n for n in range(26, 31)} | {"S%02d" % n for n in range(27, 31)},
    }
    for name, ids in expected.items():
        values = split.get(name)
        if (not isinstance(values, list) or any(not isinstance(v, str) for v in values)
                or len(values) != len(ids) or set(values) != ids):
            raise DataError("manifest %s must match existing A2 intent IDs" % name)
    all_rows = {}
    for group, prefix, relative in SOURCES:
        with (root / relative).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            headers = reader.fieldnames or []
            if len(set(headers)) != len(headers) or not {"編號", "問題內容", "ai答覆"} <= set(headers):
                raise DataError("invalid CSV headers: %s" % relative)
            group_ids = set()
            for number, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise DataError("%s:%d inconsistent columns" % (relative, number))
                faq_id, question, answer = row["編號"], row["問題內容"], row["ai答覆"]
                if (faq_id not in {"%s%02d" % (prefix, n) for n in range(1, 31)}
                        or faq_id in all_rows or not question.strip() or not answer.strip()):
                    raise DataError("%s:%d invalid/duplicate ID or empty Q/A: %r" % (relative, number, faq_id))
                group_ids.add(faq_id)
                split_name = next(name for name, ids in expected.items() if faq_id in ids)
                all_rows[faq_id] = {
                    "faq_id": faq_id, "split": split_name, "source_group": group,
                    "source_file": relative, "source_row": number, "review_status": "needs_review",
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": question},
                                 {"role": "assistant", "content": answer}],
                }
            if len(group_ids) != 30:
                raise DataError("%s: expected 30 IDs" % relative)
    if len({r["messages"][1]["content"] for r in all_rows.values()}) != 60:
        raise DataError("duplicate source questions require manual review")
    if before != {name: sha(root / name) for name in inputs}:
        raise DataError("sources changed during read")
    return [all_rows[key] for key in sorted(all_rows)], before


def validate_data(root, directory):
    manifest = read_json(directory / "dataset_manifest.json")
    if manifest.get("schema_version") not in {"a3_sft_data_v1", "a3_sft_data_v2"}:
        raise DataError("unknown A3 dataset schema")
    source_manifest = resolve(root, manifest["intent_manifest"])
    expected_rows, hashes = snapshot(root, source_manifest)
    if hashes != manifest.get("source_sha256") or manifest.get("system_prompt") != SYSTEM:
        raise DataError("source hashes or system prompt differ from dataset manifest")
    if manifest["schema_version"] == "a3_sft_data_v2":
        from .revised_data import revisions
        revised, revision_hashes = revisions(root, resolve(root, manifest["answer_revisions"]), expected_rows)
        if revision_hashes != manifest.get("revision_inputs_sha256"):
            raise DataError("revised answers or evidence changed; prepare a NEW derived dataset")
        expected_rows = [revised.get(row["faq_id"], row) for row in expected_rows]
    actual = {}
    for split in ("train", "validation", "test"):
        path = directory / (split + ".jsonl")
        if sha(path) != manifest.get("files_sha256", {}).get(path.name):
            raise DataError("prepared file changed: %s" % path)
        rows = read_rows(path)
        expected = [row for row in expected_rows if row["split"] == split]
        # Exact original snapshot (v1), or exact source-linked Train revision (v2).
        if rows != expected:
            raise DataError("%s must exactly match its source/revision contract and fixed intent split" % path)
        actual[split] = rows
    if manifest.get("counts") != {name: len(rows) for name, rows in actual.items()}:
        raise DataError("dataset manifest counts differ from actual splits")
    return actual, manifest
