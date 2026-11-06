"""Stable identities and conservative exact-key normalization; no model imports."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import struct
import sys
import unicodedata
from pathlib import Path
from importlib import metadata

NORMALIZATION_VERSION = "nfc_strip_collapse_whitespace_v1"

def normalize_question(query):
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    normalized = " ".join(unicodedata.normalize("NFC", query).split())
    if not normalized:
        raise ValueError("query must not be empty after normalization")
    return normalized

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()

def environment():
    """Strict v2 identity shared by runner/worker; OS build is diagnostic only."""
    names = ("torch", "transformers", "sentence-transformers", "peft", "numpy", "faiss-cpu",
             "faiss-gpu", "langchain-core", "langchain-community", "langchain-text-splitters",
             "huggingface-hub", "accelerate", "jieba", "packaging")
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"schema_version": "stable_environment_v2",
            "python": platform.python_version(),
            "python_implementation": sys.implementation.name,
            "python_executable": str(Path(sys.executable).resolve()),
            "python_prefix": str(Path(sys.prefix).resolve()),
            "os_name": os.name, "sys_platform": sys.platform,
            "machine": platform.machine(), "pointer_bits": struct.calcsize("P") * 8,
            "host": platform.node(), "packages": versions}
