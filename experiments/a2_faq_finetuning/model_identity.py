"""Read-only local checkpoint identity and installed package metadata."""
from importlib import metadata
from pathlib import Path
from .common import DataError, sha256_file


def package_versions():
    result = {}
    for name in ("torch", "transformers", "sentence-transformers", "numpy", "peft",
                 "faiss-cpu", "faiss-gpu", "huggingface-hub", "packaging"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def local_model_identity(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return None
    files = [{"name": p.relative_to(directory).as_posix(), "bytes": p.stat().st_size,
              "sha256": sha256_file(p)} for p in sorted(directory.rglob("*")) if p.is_file()]
    if not files:
        raise DataError("empty original model directory: " + str(directory))
    return {"path": str(directory.resolve()), "files": files}
