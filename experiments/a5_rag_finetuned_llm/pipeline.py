"""Read existing A0 index, retrieve Top-5, generate with the actual A3 adapter."""
from __future__ import annotations

from pathlib import Path
import warnings

from experiments.a3_faq_sft.common import DataError, read_json, resolve, sha
from experiments.a3_faq_sft.model_loader import (
    configure_offline, load_adapter_for_chat, verify_adapter)
from .config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, TOP_K, GenerationSettings
from .prompts import PROMPT_VERSION, build_messages


def index_inputs(directory):
    """Fail before a shared constructor can create a missing directory."""
    files = [directory / "index.faiss", directory / "index.pkl"]
    if not directory.is_dir() or any(not path.is_file() for path in files):
        raise DataError("A0 index missing: %s. Restore the existing A0 index from the GPU project/backup. "
                        "If none exists, separately review the A0 build_index instructions; A5 never builds it."
                        % directory)
    version = directory / "version.json"
    if version.exists():
        if not version.is_file():
            raise DataError("version.json is not a file")
        files.append(version)
    return {path.name: sha(path) for path in files}


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


class A5Pipeline:
    def __init__(self, *, root: Path, index_dir: Path, base_model: str, adapter: Path,
                 device="cuda", precision="fp16", embedding_device=None, offline=False,
                 allow_unreviewed=False, generation=None):
        self.generation = generation or GenerationSettings()
        self.generation.validate()
        self.root = root.expanduser().resolve()
        self.index_dir = resolve(self.root, index_dir)
        self.adapter = resolve(self.root, adapter)
        self.base_model = base_model
        self.index_hashes = index_inputs(self.index_dir)
        # Check missing/incomplete/provisional adapters before allocating BGE/LLM.
        initial_provenance = verify_adapter(self.adapter)
        if initial_provenance["provisional"] and not allow_unreviewed:
            raise DataError("A3 provisional smoke adapter requires --allow-unreviewed")
        configure_offline(offline)
        import torch
        from experiments.shared.embeddings import BGEM3Embedder
        from experiments.shared.vector_store import FAISSVectorStore
        from experiments.shared.retrieval import Retriever

        self.device = device
        self.embedding_device = embedding_device or device
        if device not in ("cpu", "cuda") or self.embedding_device not in ("cpu", "cuda"):
            raise DataError("LLM/embedding device must be cpu or cuda")
        if "cuda" in (device, self.embedding_device) and not torch.cuda.is_available():
            raise DataError("CUDA requested but unavailable")
        self.embedder = BGEM3Embedder(model_name=EMBEDDING_MODEL, device=self.embedding_device,
                                    batch_size=32, normalize_embeddings=True)
        # Same wrapper/default 512-token encoding protocol as A0, not A2 fine-tuned embeddings.
        self.embedder.model.eval()
        self.store = FAISSVectorStore(embedding_dim=self.embedder.embedding_dim,
                                      index_path=str(self.index_dir), use_hnsw=False)
        # Shared constructor calls mkdir(exist_ok=True), but index_inputs requires
        # an EXISTING directory and both files; no new index directory is created.
        if not self.store.load():
            raise DataError("cannot load A0 FAISS/index.pkl; index is never rebuilt by A5")
        version_file = self.index_dir / "version.json"
        self.index_version = read_json(version_file) if version_file.is_file() else None
        validate_loaded_store(self.store, self.embedder.embedding_dim, self.index_version)
        if self.index_hashes != index_inputs(self.index_dir):
            raise DataError("A0 index files changed during loading")
        if self.index_version is None:
            warnings.warn("A0 version.json missing: loaded index shape/mapping checked, but embedding "
                          "source/chunk512/128/L2 provenance cannot be certified; provisional integration only.",
                          RuntimeWarning)
        self.retriever = Retriever(vector_store=self.store, embedder=self.embedder, top_k=TOP_K)
        self.model, self.tokenizer, self.provenance = load_adapter_for_chat(
            self.root, base_model, self.adapter, device, precision, offline, allow_unreviewed)
        if self.provenance != initial_provenance:
            raise DataError("A3 adapter provenance changed during pipeline loading")
        self.provisional = (self.provenance["provisional"] or allow_unreviewed or self.index_version is None)
        random_seed = self.generation.seed
        import random
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.benchmark = False

    def answer(self, query):
        import math
        import torch

        if not isinstance(query, str) or not query.strip():
            raise DataError("query must be nonempty text")
        # Normal query embedding -> FAISS search; no gold ID or FAQ answer lookup.
        with torch.inference_mode():
            retrieved = self.retriever.retrieve(query)
        if len(retrieved) != TOP_K or any(not math.isfinite(float(score)) for _, score in retrieved):
            raise DataError("retrieval did not return five valid documents/scores; no base-only fallback")
        docs = [doc for doc, _ in retrieved]
        scores = [float(score) for _, score in retrieved]
        messages = build_messages(query, docs)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        length = inputs["input_ids"].shape[1]
        context_limit = self.model.config.max_position_embeddings
        if length > self.generation.max_input_tokens or length + self.generation.max_new_tokens > context_limit:
            raise DataError("full Top-5 RAG prompt has %d tokens; exceeds input/context limit. "
                            "Raise --max-input-tokens within model capacity; no document/question is silently truncated."
                            % length)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        kwargs = {"max_new_tokens": self.generation.max_new_tokens,
                  "do_sample": self.generation.do_sample,
                  "repetition_penalty": self.generation.repetition_penalty,
                  "pad_token_id": self.tokenizer.pad_token_id}
        if self.generation.do_sample:
            kwargs.update(temperature=self.generation.temperature, top_p=self.generation.top_p)
        if self.generation.eos_policy == "tokenizer":
            if self.tokenizer.eos_token_id is None:
                raise DataError("tokenizer EOS missing")
            kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        # Default retains A3/base generation_config EOS/end-of-turn IDs.
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        answer = self.tokenizer.decode(output[0, length:], skip_special_tokens=True).strip()
        return {"answer": answer, "documents": docs, "scores": scores,
                "input_tokens": length, "provisional": self.provisional,
                "base_model": self.base_model, "adapter": str(self.adapter),
                "prompt_version": PROMPT_VERSION, "index_version_available": self.index_version is not None}
