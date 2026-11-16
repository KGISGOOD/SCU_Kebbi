"""Index validation and original-versus-finetuned provenance checks; no training."""
from pathlib import Path
from .resource_io import DataError, read_json, sha
TOP_K=5
EMBEDDING_MODEL="BAAI/bge-m3"
CHUNK_SIZE=512
CHUNK_OVERLAP=128
sha256_file=sha

def validate_loaded_store(store, embedding_dim, version):
    import faiss

    index = store.index
    if (index is None or not index.is_trained or index.d != embedding_dim
            or not isinstance(index, faiss.IndexFlat) or index.metric_type != faiss.METRIC_INNER_PRODUCT):
        raise DataError("A5 requires the A0 normalized dense IndexFlatIP with matching BGE-M3 dimension")
    count = index.ntotal
    mapping = store.index_to_docstore_id
    docs = store.docstore
    if (count < TOP_K or not isinstance(docs, list) or len(docs) != count
            or not isinstance(mapping, dict) or set(mapping) != set(range(count))
            or any(type(i) is not int or not 0 <= i < len(docs) for i in mapping.values())
            or len(set(mapping.values())) != count):
        raise DataError("A0 index/document mapping is incomplete, empty or inconsistent")
    if any(not isinstance(doc.page_content, str) or not doc.page_content.strip()
           or not isinstance(doc.metadata, dict) for doc in docs):
        raise DataError("A0 index contains invalid/empty document content")
    if {doc.metadata.get("source_group") for doc in docs} != {"type1", "type2"}:
        raise DataError("index must contain A0 Type1/Type2 documents, not an A1/A2 FAQ-only index")
    if version is not None:
        expected = {"embedding_model": EMBEDDING_MODEL, "embedding_dimension": embedding_dim,
                    "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
                    "normalization": "L2", "index_type": "IndexFlatIP", "chunk_count": count}
        for key, value in expected.items():
            if version.get(key) != value:
                raise DataError("A0 version.json mismatch for %s: expected %r, got %r"
                                % (key, value, version.get(key)))
        if version.get("data_fingerprint") != store._compute_data_fingerprint(docs):
            raise DataError("A0 version fingerprint does not match loaded document content")
        sources = version.get("source_directories")
        if (not isinstance(sources, list) or len(sources) != 2
                or any(not isinstance(source, str) for source in sources)
                or {Path(source.replace("\\", "/")).name for source in sources} != {"type1", "type2"}):
            raise DataError("A0 version.json must identify Type1 and Type2 source directories")


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

