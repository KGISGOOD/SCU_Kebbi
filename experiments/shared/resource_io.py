"""Strict standard-library resource IO extracted from the former training module."""
import hashlib
import json
from pathlib import Path

class DataError(ValueError):
    pass


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


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


