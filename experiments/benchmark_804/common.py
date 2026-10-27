from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
from importlib import metadata
import uuid

ROOT = Path(__file__).resolve().parents[2]
INPUT = "experiments/benchmark_804/input/資科萬事通_804題測試分配.jsonl"
GUIDE = "experiments/benchmark_804/input/資科萬事通_測試包使用說明.md"
ARCHS = ("A0", "A1", "A2-1", "A2-2", "A3", "A4-1", "A4-2", "A5")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").split("\n") if s.strip()]


def path(root, value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else root / p).resolve()


def save(target, value, replace=False):
    """Atomic commit; preserve old attempts. Only derived state/report pointers replace."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not replace:
        raise FileExistsError(target)
    tmp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    if target.exists() and not replace:
        tmp.unlink()
        raise FileExistsError(target)
    os.replace(str(tmp), str(target))


def environment():
    versions = {}
    for name in ("torch", "transformers", "sentence-transformers", "peft", "faiss-cpu", "faiss-gpu", "numpy", "langchain-core"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "system": platform.system(),
            "machine": platform.machine(), "packages": versions}


def code_hashes(root):
    folders = ("benchmark_804", "algorithms", "evaluation", "shared", "baselines",
               "a2_faq_finetuning", "a3_faq_sft", "a5_rag_finetuned_llm", "research_tools")
    return {p.relative_to(root).as_posix(): sha(p) for folder in folders
            for p in sorted((root / "experiments" / folder).rglob("*.py")) if "__pycache__" not in p.parts}


def verify(identity):
    for name, expected in identity["files_sha256"].items():
        if sha(name) != expected:
            raise ValueError("InputChanged: " + name)
    if environment() != identity["environment"]:
        raise ValueError("InputChanged: Python/package/system/architecture differs")


class Records:
    """Immutable attempt files plus atomic pointer manifest; no append-tail ambiguity."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pointer = self.directory / "index.json"
        self.index = read(self.pointer) if self.pointer.exists() else {}

    def get(self, key):
        item = self.index.get(key)
        if item is None:
            return None
        p = self.directory / item["file"]
        if sha(p) != item["sha256"]:
            raise ValueError("result evidence changed: " + str(p))
        return read(p)

    def put(self, key, record):
        filename = fingerprint(key)[:20] + "_" + uuid.uuid4().hex[:12] + ".json"
        save(self.directory / filename, record)
        self.index[key] = {"file": filename, "sha256": sha(self.directory / filename)}
        save(self.pointer, self.index, replace=True)

    def all(self):
        return [self.get(k) for k in self.index]
