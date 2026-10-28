"""Offline resource construction around the existing A0 RAGPipeline.

The existing Prompt, retrieval, vector store and text generation are reused.
Only local-only loading, length checks and measurement are added here.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import time


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(directory):
    from experiments.algorithms.semantic_cache.cache import fingerprint
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"empty resource directory: {directory}")
    return fingerprint({p.relative_to(directory).as_posix(): file_hash(p) for p in files})


def knowledge_hashes(directory):
    """A0 load() consumes these two files; version.json is not required."""
    return {name: file_hash(directory / name) for name in ("index.faiss", "index.pkl")}


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measured(function, *args, **kwargs):
    sync()
    start = time.perf_counter()
    result = function(*args, **kwargs)
    sync()
    return result, time.perf_counter() - start


class A0Adapter:
    def __init__(self, root, config):
        # Set before importing Hugging Face. Explicit local_files_only is also
        # used below, so failure never falls back to network model retrieval.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        self.root, self.config = Path(root).resolve(), dict(config)
        paths = {}
        for name in ("model", "bge", "knowledge_index"):
            raw = Path(config[name]).expanduser()
            path = (raw if raw.is_absolute() else self.root / raw).resolve()
            if not path.is_dir():
                raise ValueError(f"{name}: existing LOCAL directory required: {path}")
            paths[name] = path
        for name in ("model", "bge"):
            if not (paths[name] / "config.json").is_file():
                raise ValueError(f"{name}: config.json missing; do not pass a hub ID")
        for name in ("index.faiss", "index.pkl"):
            if not (paths["knowledge_index"] / name).is_file():
                raise ValueError(f"existing A0 knowledge index missing {name}; never auto-build")
        print("Hashing immutable local resources (including model weights)...", flush=True)
        self.paths = paths
        self.source_hashes = {
            name: knowledge_hashes(path) if name == "knowledge_index" else tree_hash(path)
            for name, path in paths.items()
        }

        import torch
        import faiss
        import importlib.metadata
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from experiments.shared.embeddings import BGEM3Embedder
        from experiments.shared.llm_loader import LLMLoader
        from experiments.shared.vector_store import FAISSVectorStore
        from experiments.baselines.rag_baseline.pipeline import RAGPipeline
        from experiments.baselines.rag_baseline import prompts

        device = config["device"]
        if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
            raise ValueError("device must be cpu or available cuda")
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[config["precision"]]
        if device == "cpu" and dtype != torch.float32:
            raise ValueError("use precision=fp32 for CPU")
        self.embedder = BGEM3Embedder.__new__(BGEM3Embedder)
        self.embedder.model_name, self.embedder.device = str(paths["bge"]), device
        self.embedder.batch_size, self.embedder.normalize_embeddings = 1, True
        self.embedder.model = SentenceTransformer(str(paths["bge"]), device=device,
                                                  local_files_only=True, trust_remote_code=False)
        self.embedder.model.max_seq_length = 512
        self.embedder.model.eval()
        self.embedder.embedding_dim = self.embedder.model.get_sentence_embedding_dimension()
        store = FAISSVectorStore(self.embedder.embedding_dim, str(paths["knowledge_index"]))
        if not store.load():
            raise ValueError("existing A0 index failed to load; no rebuild")
        if (not isinstance(store.index, faiss.IndexFlat) or store.index.metric_type != faiss.METRIC_INNER_PRODUCT
                or store.index.d != self.embedder.embedding_dim or store.index.ntotal <= 0
                or len(store.docstore) != store.index.ntotal
                or set(store.index_to_docstore_id) != set(range(store.index.ntotal))
                or set(store.index_to_docstore_id.values()) != set(range(len(store.docstore)))):
            raise ValueError("A0 index metric/dimension/document count/mapping invalid")
        # Existing index.pkl is trusted project input, never an uploaded pickle.
        self.llm = LLMLoader.__new__(LLMLoader)
        llm = self.llm
        llm.model_id, llm.device, llm.torch_dtype = str(paths["model"]), device, dtype
        llm.max_new_tokens = config["max_new_tokens"]
        llm.strict_max_input_tokens = config.get("max_input_tokens", 2048)
        llm.temperature, llm.top_p, llm.do_sample, llm.repetition_penalty = 1.0, 1.0, False, 1.0
        llm.tokenizer = AutoTokenizer.from_pretrained(str(paths["model"]), local_files_only=True,
                                                     trust_remote_code=False, use_fast=True)
        if llm.tokenizer.pad_token is None:
            llm.tokenizer.pad_token = llm.tokenizer.eos_token
        llm.model = AutoModelForCausalLM.from_pretrained(str(paths["model"]), local_files_only=True,
                                                       trust_remote_code=False, torch_dtype=dtype)
        llm.model.to(device).eval()
        if (type(llm.strict_max_input_tokens) is not int or llm.strict_max_input_tokens <= 0
                or llm.strict_max_input_tokens + llm.max_new_tokens > llm.model.config.max_position_embeddings):
            raise ValueError("A0 full input budget plus generation exceeds model context window")
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config["seed"])
        self.pipeline = RAGPipeline(store, self.embedder, llm, top_k=5)
        self.identity = {
            "schema": "semantic_cache_a0_v2", "context_scope": "single_turn",
            "resources": self.source_hashes,
            "resource_paths": {k: str(v) for k, v in paths.items()},
            "prompt_version": prompts.PROMPT_VERSION,
            "code": self.code_hash(),
            "generation": {"do_sample": False, "max_new_tokens": llm.max_new_tokens,
                           "repetition_penalty": 1.0, "seed": config["seed"],
                           "precision": config["precision"], "device": device,
                           "input_limit": llm.strict_max_input_tokens, "truncation": False, "top_k": 5},
            "embedding": {"normalization": "L2", "max_seq_length": 512},
            "packages": {p: importlib.metadata.version(p) for p in
                         ("torch", "transformers", "sentence-transformers", "numpy", "langchain-core")},
            "faiss_version": getattr(faiss, "__version__", "unknown"),
            "model_config": llm.model.config.to_dict(),
            "generation_config": llm.model.generation_config.to_dict(),
            "tokenizer": {"bos_token_id": llm.tokenizer.bos_token_id,
                          "eos_token_id": llm.tokenizer.eos_token_id,
                          "pad_token_id": llm.tokenizer.pad_token_id},
        }

    def code_hash(self):
        from experiments.algorithms.semantic_cache.cache import fingerprint
        directories = ("experiments/algorithms", "experiments/baselines/rag_baseline", "experiments/shared")
        return fingerprint({p.relative_to(self.root).as_posix(): file_hash(p)
                            for folder in directories for p in sorted((self.root / folder).rglob("*.py"))})

    def verify_sources(self):
        for name, path in self.paths.items():
            current = knowledge_hashes(path) if name == "knowledge_index" else tree_hash(path)
            if current != self.source_hashes[name]:
                raise ValueError(f"resource changed during run: {name}")
        if self.code_hash() != self.identity["code"]:
            raise ValueError("source code changed during run")

    def validate_query(self, query):
        # Reject rather than silently truncate the single-turn cache key.
        # Use tokenizer directly; SentenceTransformer.tokenize may truncate.
        full = self.embedder.model.tokenizer(query, truncation=False)["input_ids"]
        if len(full) > 512:
            raise ValueError("query exceeds BGE experiment limit of 512 tokens")

    def embed(self, query):
        self.validate_query(query)
        return self.embedder.encode(query)

    def answer(self, query):
        from experiments.baselines.rag_baseline.prompts import build_prompt, create_context_from_documents
        self.validate_query(query)
        (docs, scores), retrieval = measured(self.pipeline.retrieve, query)
        def prepare():
            prompt = build_prompt(create_context_from_documents(docs), query)
            tokens = self.llm.tokenizer(prompt, truncation=False)["input_ids"]
            context_limit = self.llm.model.config.max_position_embeddings
            if len(tokens) > self.llm.strict_max_input_tokens or len(tokens) + self.llm.max_new_tokens > context_limit:
                raise ValueError(f"full A0 prompt has {len(tokens)} tokens; input limit={self.llm.strict_max_input_tokens}, "
                                 f"max_new_tokens={self.llm.max_new_tokens}, model context={context_limit}; no truncation allowed")
            return len(tokens)
        input_tokens, preparation = measured(prepare)
        answer, generation = measured(self.pipeline.generate_answer, query, docs)
        return answer, {"retrieval_seconds": retrieval, "prompt_check_seconds": preparation,
                        "generation_seconds": generation, "input_tokens": input_tokens,
                        "documents": [{"rank": i, "score": float(score),
                                       "metadata": doc.metadata, "content": doc.page_content}
                                      for i, (doc, score) in enumerate(zip(docs, scores), 1)]}
