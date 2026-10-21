"""Strict, standard-library-only data IO shared by the A2 commands."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path("experiments/datasets/faq_finetuning")
CSV_SOURCES = (
    ("parents", "P", Path("畢業專題_爬蟲/type2/資料擴充 - 家長 FAQ 資料集（30 題）.csv")),
    ("highschool", "S", Path("畢業專題_爬蟲/type2/資料擴充 - 高中生 FAQ 資料集（30 題）.csv")),
)
GROUPS = {
    "seen_test": ("seen", "train", 2),
    "unseen_validation": ("unseen", "validation", 3),
    "unseen_test": ("unseen", "test", 3),
}
EXPECTED_SPLITS = {
    "train": ["P%02d" % n for n in range(1, 22)] + ["S%02d" % n for n in range(1, 22)],
    "validation": ["P%02d" % n for n in range(22, 26)] + ["S%02d" % n for n in range(22, 27)],
    "test": ["P%02d" % n for n in range(26, 31)] + ["S%02d" % n for n in range(27, 31)],
}


class DataError(ValueError):
    """Invalid or missing input; callers must not silently skip it."""


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, value: Any) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def reject_constant(value):
    raise DataError("non-standard JSON constant: %s" % value)


def decode_json(text: str):
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = decode_json(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise DataError("%s: %s" % (path, exc)) from exc
    if not isinstance(value, dict):
        raise DataError("%s: expected JSON object" % path)
    return value


def read_jsonl(path: Path, issues: List[dict]) -> List[dict]:
    rows = []
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, text in enumerate(handle, 1):
                try:
                    if not text.strip():
                        raise DataError("blank JSONL line")
                    row = decode_json(text)
                    if not isinstance(row, dict):
                        raise DataError("expected JSON object")
                    rows.append(row)
                except ValueError as exc:
                    issues.append({
                        "severity": "error", "code": "jsonl_parse",
                        "location": "%s:%d" % (path, line_number),
                        "message": str(exc), "text": text.rstrip("\r\n"),
                    })
    except (OSError, UnicodeError) as exc:
        issues.append({"severity": "error", "code": "input_read",
                       "location": str(path), "message": str(exc)})
    return rows


def load_source_faqs(root: Path) -> List[dict]:
    """Use actual UTF-8 CSV headers and already-prefixed IDs, preserving question."""
    faqs = []
    for group, prefix, relative in CSV_SOURCES:
        path = root / relative
        group_rows = []
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, strict=True)
                headers = reader.fieldnames or []
                if len(headers) != len(set(headers)):
                    raise DataError("duplicate CSV column names")
                missing = {"編號", "問題內容", "ai答覆"} - set(headers)
                if missing:
                    raise DataError("missing CSV columns: %s" % sorted(missing))
                for number, row in enumerate(reader, 2):
                    if None in row or any(value is None for value in row.values()):
                        raise DataError("CSV row %d has inconsistent columns" % number)
                    faq_id, question = row["編號"], row["問題內容"]
                    expected = {"%s%02d" % (prefix, n) for n in range(1, 31)}
                    if faq_id not in expected:
                        raise DataError("row %d invalid ID %r; expected %s01-%s30; do not add prefix twice"
                                        % (number, faq_id, prefix, prefix))
                    if not question.strip():
                        raise DataError("row %d %s empty question" % (number, faq_id))
                    group_rows.append({"faq_id": faq_id, "question": question,
                                       "source_group": group})
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            raise DataError("%s: %s" % (path, exc)) from exc
        ids = [row["faq_id"] for row in group_rows]
        expected = {"%s%02d" % (prefix, n) for n in range(1, 31)}
        if len(ids) != 30 or len(set(ids)) != 30 or set(ids) != expected:
            raise DataError("%s: expected 30 unique IDs; actual=%r" % (path, ids))
        faqs.extend(group_rows)
    questions = [row["question"] for row in faqs]
    if len(set(questions)) != 60:
        raise DataError("CSV questions are not unique; source tracing/gold requires human review")
    return sorted(faqs, key=lambda row: row["faq_id"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_input_hashes(hashes):
    """Reject edits between validation, consumer reads and export."""
    for name, digest in hashes.items():
        if sha256_file(Path(name)) != digest:
            raise DataError("input changed after snapshot: %s" % name)


def checkpoint_files(path):
    """Portable relative names; the completion marker cannot hash itself."""
    return {item.relative_to(path).as_posix(): sha256_file(item)
            for item in sorted(path.rglob("*"))
            if item.is_file() and item.relative_to(path).as_posix() != "a2_checkpoint_complete.json"}


def read_checkpoint_provenance(path, require_a2=False):
    """Fail closed for A2 exports, including interrupted saves and wrong run roots.

    Original ST/HF model directories without A2 markers remain supported.
    Older A2 exports need a separate integrity review; no implicit upgrade.
    """
    names = ("a2_training_provenance.json", "a2_checkpoint_pending.json",
             "a2_checkpoint_complete.json")

    def is_a2_run(directory):
        config = directory / "run_config.json"
        return (config.is_file() and
                read_json(config).get("schema_version") == "a2_training_provenance_v1")

    in_a2_run = (is_a2_run(path.parent) or
                 (path.parent.name == "checkpoints" and is_a2_run(path.parent.parent)))
    detected = path.is_dir() and (in_a2_run or is_a2_run(path) or
                                  any((path / name).exists() for name in names))
    if not detected and not require_a2:
        return None
    if not (path / "modules.json").is_file():
        raise DataError("not a complete A2 SentenceTransformer model directory: %s" % path)
    completion = read_json(path / "a2_checkpoint_complete.json")
    files = checkpoint_files(path)
    if (completion.get("schema_version") != "a2_checkpoint_complete_v1" or
            completion.get("files_sha256") != files):
        raise DataError("A2 checkpoint incomplete or files changed: %s" % path)
    provenance = read_json(path / "a2_training_provenance.json")
    if (provenance.get("schema_version") != "a2_training_provenance_v1" or
            provenance.get("mode") not in ("a2_1", "a2_2") or
            type(provenance.get("provisional")) is not bool or
            type(provenance.get("completed_epoch")) is not int or
            type(provenance.get("optimizer_steps")) is not int or
            provenance["completed_epoch"] < 1 or provenance["optimizer_steps"] < 1):
        raise DataError("invalid A2 training provenance: %s" % path)
    return provenance


def write_json_exclusive(path: Path, value: Any) -> None:
    """Never replace data, reports or historical results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl_exclusive(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def add_data_arguments(parser) -> None:
    parser.add_argument("--project-root", type=Path, default=default_root())
    parser.add_argument("--manifest", default=str(DATA_DIR / "faq_split_manifest.json"))
    parser.add_argument("--training", default=str(DATA_DIR / "faq_training_candidates.jsonl"))
    parser.add_argument("--queries", default=str(DATA_DIR / "faq_eval_queries.jsonl"))
    parser.add_argument("--candidates", default=str(DATA_DIR / "faq_retrieval_candidates.jsonl"))
    parser.add_argument("--diagnostics", default="experiments/datasets/scu_faq_matching_test_v1.jsonl")


def data_paths(args) -> Dict[str, Path]:
    root = args.project_root.expanduser().resolve()
    return {name: resolve_path(root, getattr(args, name))
            for name in ("manifest", "training", "queries", "candidates", "diagnostics")}
