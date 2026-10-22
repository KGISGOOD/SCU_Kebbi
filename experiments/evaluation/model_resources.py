"""Read-only offline model resolution and inference-file identity (no model imports)."""
from pathlib import Path, PurePosixPath

from .common import DataError, digest, read_json, sha


def local_snapshot(root, name, revision=None, snapshot=None):
    if snapshot is not None:
        path = Path(snapshot).expanduser()
        path = path if path.is_absolute() else root / path
        if not path.is_dir():
            raise DataError("explicit snapshot directory missing: %s" % path)
        return path.resolve()
    path = Path(name).expanduser()
    path = path if path.is_absolute() else root / path
    if path.is_dir():
        return path.resolve()
    if name not in ("BAAI/bge-m3", "BAAI/bge-reranker-v2-m3", "meta-llama/Llama-3.2-3B-Instruct"):
        raise DataError("local model path missing: %s" % path)
    # A file-level cache lookup never requests a complete repository, contacts
    # the Hub or downloads. Preserve the snapshot parent, NOT config's blob target.
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache(name, "config.json", revision=revision or "main")
    if not isinstance(cached, str):
        raise DataError("config.json not cached for %s revision=%s; supply an existing snapshot; no download"
                        % (name, revision or "main"))
    return Path(cached).parent.resolve()


def inference_identity(path, sentence_model=False):
    """Validate selected native PyTorch weights, tokenizer and declared ST modules.

    Docs/images/ONNX/other export formats are not dense PyTorch dependencies.
    Loader validation is still required; file hashes do not prove valid tensors.
    """
    selected = set()

    def require(relative, json_file=False, weight=False):
        relative = str(relative)
        portable = PurePosixPath(relative)
        if (not relative or portable.is_absolute() or ".." in portable.parts
                or "\\" in relative or ":" in relative):
            raise DataError("unsafe model dependency path: %r" % relative)
        file = path / relative
        if not file.is_file() or file.stat().st_size == 0:
            raise DataError("inference dependency missing/empty (including broken links): %s" % file)
        with file.open("rb") as handle:
            prefix = handle.read(256)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/"):
            raise DataError("unresolved Git LFS pointer: %s" % file)
        if weight and file.stat().st_size < 1024:
            raise DataError("implausibly small model weight file: %s" % file)
        selected.add(relative)
        return read_json(file) if json_file else None

    def transformer(directory=""):
        prefix = directory + "/" if directory else ""
        require(prefix + "config.json", json_file=True)
        require(prefix + "tokenizer_config.json", json_file=True)
        tokenizer_files = ("tokenizer.json", "sentencepiece.bpe.model", "tokenizer.model", "spiece.model", "vocab.txt")
        available = [f for f in tokenizer_files if (path / prefix / f).is_file()]
        if not available:
            if (path / prefix / "vocab.json").is_file() and (path / prefix / "merges.txt").is_file():
                available = ["vocab.json", "merges.txt"]
            else:
                raise DataError("no usable tokenizer vocabulary at %s" % (path / prefix))
        for filename in available:
            # tokenizer.json is a JSON object, never accept a malformed one.
            require(prefix + filename, json_file=filename.endswith(".json"))
        for filename in ("special_tokens_map.json", "added_tokens.json", "generation_config.json",
                         "sentence_bert_config.json"):
            if (path / prefix / filename).exists():
                require(prefix + filename, json_file=True)
        # Match the native Transformers preference; every indexed shard is required.
        for filename in ("model.safetensors", "model.safetensors.index.json",
                         "pytorch_model.bin", "pytorch_model.bin.index.json"):
            file = path / prefix / filename
            if not file.exists():
                continue
            if filename.endswith(".index.json"):
                index = require(prefix + filename, json_file=True)
                mapping = index.get("weight_map")
                if (not isinstance(mapping, dict) or not mapping
                        or any(not isinstance(v, str) or not v for v in mapping.values())):
                    raise DataError("invalid weight_map: %s" % file)
                for shard in sorted(set(mapping.values())):
                    require(prefix + shard, weight=True)
            else:
                require(prefix + filename, weight=True)
            break
        else:
            raise DataError("no native PyTorch model weights/index: %s" % (path / prefix))

    if sentence_model:
        # Require the actual pooling recipe, never silently fall back to a newly
        # invented mean-pooling model when a partial BGE cache lacks modules.json.
        require("modules.json")
        import json
        modules = json.loads((path / "modules.json").read_text(encoding="utf-8-sig"))
        if not isinstance(modules, list) or not modules:
            raise DataError("invalid modules.json: %s" % path)
        types = []
        for module in modules:
            if not isinstance(module, dict) or not isinstance(module.get("path"), str):
                raise DataError("invalid SentenceTransformer module: %r" % module)
            kind, directory = module.get("type"), module["path"]
            if kind == "sentence_transformers.models.Transformer":
                transformer(directory)
            elif kind == "sentence_transformers.models.Pooling":
                require((directory + "/" if directory else "") + "config.json", json_file=True)
            elif kind != "sentence_transformers.models.Normalize":
                raise DataError("unreviewed SentenceTransformer module type: %r" % kind)
            types.append(kind)
        if (types.count("sentence_transformers.models.Transformer") != 1
                or types.count("sentence_transformers.models.Pooling") != 1):
            raise DataError("BGE-M3 must declare one Transformer and one Pooling module")
        if (path / "config_sentence_transformers.json").exists():
            require("config_sentence_transformers.json", json_file=True)
    else:
        transformer()
    files = {relative: sha(path / relative) for relative in sorted(selected)}
    return {"path": str(path), "files_sha256": files, "content_sha256": digest(files),
            "identity_scope": "native PyTorch inference dependencies; not entire Hub repository",
            "loader_validation_required": True}


def cached_model(root, name, revision=None, snapshot=None):
    path = local_snapshot(root, name, revision, snapshot)
    identity = inference_identity(path, sentence_model=name == "BAAI/bge-m3")
    if name != "BAAI/bge-m3":
        # Preserve A3's complete local-base file identity contract. Avoiding a
        # full-repository download check does not change Adapter provenance rules.
        identity["inference_files_sha256"] = identity["files_sha256"]
        files = {p.relative_to(path).as_posix(): sha(p) for p in sorted(path.rglob("*")) if p.is_file()}
        identity.update(files_sha256=files, content_sha256=digest(files),
                        identity_scope="all existing local files; required inference dependencies validated")
    return {"model_argument": name, "revision_requested": revision,
            "snapshot_override": snapshot,
            **identity}
