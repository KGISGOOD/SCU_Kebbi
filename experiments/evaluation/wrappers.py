"""Minimal adapters around actual existing algorithms; only question enters answer()."""
from __future__ import annotations

import math
import time

from .common import DataError, environment, resolve


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measured(function, *args, **kwargs):
    sync()
    start = time.perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        sync()
        measured.seconds = time.perf_counter() - start


def documents(docs, scores):
    if len(docs) != len(scores) or any(not math.isfinite(float(s)) for s in scores):
        raise DataError("retrieval scores/documents invalid")
    return [{"rank": i, "document_id": doc.metadata.get("document_id"),
             "source": doc.metadata.get("source_file"), "score": float(score),
             "content": doc.page_content, "metadata": doc.metadata}
            for i, (doc, score) in enumerate(zip(docs, scores), 1)]


class GenerationCapture:
    """Observe the existing generate call, preserving prompt and output tensors.

    Greedy decoding is the requested evaluation setting. Remove sampling-only
    kwargs leaked by legacy LLMLoader when do_sample=False, not its prompt logic.
    """
    def __init__(self, model, tokenizer, max_new_tokens):
        self.last = {}
        self.model = model
        self.tokenizer = tokenizer
        original = model.generate

        def generate(*args, **kwargs):
            from copy import deepcopy
            kwargs["do_sample"] = False
            kwargs["max_new_tokens"] = max_new_tokens
            removed = {key: kwargs.pop(key) for key in ("temperature", "top_p", "top_k") if key in kwargs}
            effective_config = deepcopy(kwargs.get("generation_config", model.generation_config))
            effective_config.do_sample = False
            # Neutral defaults avoid inherited sampling-only settings conflicting
            # with greedy validation; EOS/pad/other normal model settings remain.
            effective_config.temperature, effective_config.top_p, effective_config.top_k = 1.0, 1.0, 50
            kwargs["generation_config"] = effective_config
            inputs = kwargs.get("input_ids", args[0] if args else None)
            self.last = {"generation_kwargs": {k: v.to_dict() if k == "generation_config" else v for k, v in kwargs.items()
                                               if k not in ("input_ids", "attention_mask")},
                         "removed_sampling_kwargs": removed}
            if inputs is None or inputs.ndim != 2 or inputs.shape[0] != 1:
                raise DataError("evaluation expects a single query input_ids tensor")
            input_ids = inputs[0].detach().cpu().tolist()
            self.last.update(input_tokens=len(input_ids), input_token_ids=input_ids,
                             effective_prompt=tokenizer.decode(input_ids, skip_special_tokens=False))
            sync()
            started = time.perf_counter()
            try:
                output = original(*args, **kwargs)
            finally:
                sync()
                self.last["generation_seconds"] = time.perf_counter() - started
            sequences = output.sequences if hasattr(output, "sequences") else output
            new = sequences[0, len(input_ids):].detach().cpu().tolist()
            eos = kwargs.get("eos_token_id", model.generation_config.eos_token_id)
            eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
            hit_eos = bool(new and new[-1] in eos_ids)
            self.last.update(output_tokens=len(new), output_token_ids=new,
                             raw_decoded_output=tokenizer.decode(new, skip_special_tokens=False),
                             reached_token_limit=len(new) >= max_new_tokens,
                             stop_reason="eos" if hit_eos else "token_limit" if len(new) >= max_new_tokens else "other",
                             stop_reason_source="inferred_from_last_token_and_output_length",
                             actual_model_dtype=str(next(model.parameters()).dtype))
            return output

        model.generate = generate


class Architecture:
    def __init__(self, architecture, root, config, resources):
        import torch
        from experiments.a3_faq_sft.model_loader import configure_offline
        configure_offline(True)
        self.arch = architecture
        self.root, self.config, self.resources = root, config, resources
        self.capture = None
        self.partial = {}
        self.provisional = False
        self.model_name = config["base_model"]
        self.model_version = resources.get("base", {}).get("value")
        self.metadata = {"architecture": architecture, "generation": {"do_sample": False,
                         "max_new_tokens": config["max_new_tokens"], "single_turn": True},
                         "packages": environment(), "hardware": {
                             "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
                             "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                             "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None},
                         "seed": config["seed"], "device_requested": config["device"]}
        if config["device"] == "cuda" and not torch.cuda.is_available():
            raise DataError("CUDA requested but unavailable")

        if architecture in ("A2-1", "A2-2"):
            from experiments.a2_faq_finetuning.faq_answering import FAQAnswerer
            key = "a2_1_model" if architecture == "A2-1" else "a2_2_model"
            self.engine = FAQAnswerer(
                root=root, model=config[key], model_tag=architecture,
                candidates=resolve(root, config["a2_candidates"]), device=config["device"],
                max_length=config["a2_max_length"], offline=True,
                allow_unreviewed=config["allow_unreviewed"])
            self.model_name, self.model_version = config[key], resources[key]["value"]
            self.provisional = resources[key]["value"]["provenance"]["provisional"]
            self.metadata.update(embedding_max_length=config["a2_max_length"], precision="float32",
                                 candidate_count=60, retrieval="exact normalized dot product; FAQ ID tie-break",
                                 prompt_version=None, training_provenance=resources[key]["value"]["provenance"])
        elif architecture == "A1":
            from experiments.baselines.faq_semantic_matching.config import settings
            from experiments.baselines.faq_semantic_matching.matcher import FAQMatcher
            settings.embedding_model = resources["bge"]["value"]["path"]
            self.engine = FAQMatcher(faq_dataset_path=str(resolve(root, config["a1_canonical"])),
                                     faq_vector_store_dir=str(resolve(root, config["a1_index"])))
            self.engine.embedder.model.to(config["device"]).eval()
            self.engine.embedder.device = config["device"]
            if not self.engine.load_index() or len(self.engine.faqs) != 60:
                raise DataError("A1 failed to load complete 60-question canonical/index; never auto-build")
            store = self.engine.vector_store
            if (store.index.ntotal != 60 or store.index.d != self.engine.embedder.embedding_dim
                    or len(store.docstore) != 60 or set(store.index_to_docstore_id) != set(range(60))
                    or set(store.index_to_docstore_id.values()) != set(range(60))):
                raise DataError("A1 index/mapping incomplete")
            import faiss
            if not isinstance(store.index, faiss.IndexFlat) or store.index.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise DataError("A1 expected normalized IndexFlatIP")
            canonical = {row["faq_id"]: row for row in self.engine.faqs}
            seen = set()
            for doc in store.docstore:
                faq_id = doc.metadata.get("faq_id")
                if faq_id not in canonical or faq_id in seen:
                    raise DataError("A1 index has duplicate/unknown FAQ ID")
                seen.add(faq_id)
                if doc.page_content != canonical[faq_id]["question"] or doc.metadata.get("answer") != canonical[faq_id]["answer"]:
                    raise DataError("A1 index/canonical Q/A mismatch: %s" % faq_id)
            self.model_name, self.model_version = config["bge_model"], resources["bge"]["value"]
            self.metadata.update(candidate_count=60, precision="float32", embedding_max_length=512,
                                 threshold=settings.similarity_threshold, prompt_version=None,
                                 retrieval="existing FAQMatcher.match -> FAISS Top-1 (Top-2 internal)")
        elif architecture == "A3":
            from experiments.a3_faq_sft.model_loader import load_adapter_for_chat
            self.model, self.tokenizer, provenance = load_adapter_for_chat(
                root, config["base_model"], resolve(root, config["a3_adapter"]),
                config["device"], config["precision"], True, config["allow_unreviewed"])
            self.provenance = provenance
            self.provisional = provenance["provisional"]
            self.metadata.update(prompt_version="A3 saved system + native chat template", precision=config["precision"],
                                 training_provenance=provenance, adapter=resources["a3_adapter"]["value"], max_input_tokens=512)
            self.capture = GenerationCapture(self.model, self.tokenizer, config["max_new_tokens"])
        elif architecture == "A5":
            from experiments.a5_rag_finetuned_llm.pipeline import A5Pipeline
            from experiments.a5_rag_finetuned_llm.config import GenerationSettings
            from experiments.shared import embeddings
            from unittest.mock import patch
            original_embedder = embeddings.BGEM3Embedder
            pinned_path = resources["bge"]["value"]["path"]

            def pinned_embedder(*, model_name="BAAI/bge-m3", **kwargs):
                if model_name != "BAAI/bge-m3":
                    raise DataError("unexpected A5 embedding model: %s" % model_name)
                return original_embedder(model_name=pinned_path, **kwargs)

            # A5 imports this class inside its constructor and offers no model-path
            # parameter. Scope only the dependency injection to this isolated worker
            # constructor; preserve A5's model-ID/index checks and all retrieval math.
            with patch.object(embeddings, "BGEM3Embedder", new=pinned_embedder):
                self.engine = A5Pipeline(
                    root=root, index_dir=resolve(root, config["dense_index"]), base_model=config["base_model"],
                    adapter=resolve(root, config["a3_adapter"]), device=config["device"], precision=config["precision"],
                    offline=True, allow_unreviewed=config["allow_unreviewed"],
                    generation=GenerationSettings(do_sample=False, max_new_tokens=config["max_new_tokens"],
                                                  seed=config["seed"], max_input_tokens=config["a5_max_input_tokens"]))
            if self.engine.embedder.model_name != pinned_path:
                raise DataError("A5 did not load the preflight-pinned BGE snapshot")
            context_window = self.engine.model.config.max_position_embeddings
            if config["a5_max_input_tokens"] + config["max_new_tokens"] > context_window:
                raise DataError("A5 configured input cap + max_new_tokens exceeds loaded model context: %d + %d > %d"
                                % (config["a5_max_input_tokens"], config["max_new_tokens"], context_window))
            self.metadata["model_context_window"] = context_window
            self.provisional = self.engine.provenance["provisional"]
            self.metadata.update(prompt_version="a5_a0_baseline_basic_v1_native_chat_v1", precision=config["precision"],
                                 adapter=resources["a3_adapter"]["value"], dense_top_k=5,
                                 pipeline_provisional=self.engine.provisional,
                                 max_input_tokens=config["a5_max_input_tokens"], truncation=False)
            self.capture = GenerationCapture(self.engine.model, self.engine.tokenizer, config["max_new_tokens"])
            original = self.engine.retriever.retrieve

            def retrieval(query):
                values = measured(original, query)
                self.partial.update(retrieval_seconds=measured.seconds,
                                    retrieved_documents=documents([d for d, _ in values], [s for _, s in values]))
                return values
            self.engine.retriever.retrieve = retrieval
        else:
            self._load_rag()
        self.metadata["model_provisional"] = self.provisional
        if architecture in ("A0", "A1", "A4-1", "A4-2", "A5"):
            self.metadata["embedding_model_version"] = resources["bge"]["value"]
        self.metadata["model_name"] = self.model_name
        self.metadata["model_version_or_path"] = self.model_version
        if self.capture:
            self.metadata["actual_model_dtype"] = str(next(self.capture.model.parameters()).dtype)
            self.metadata["actual_model_device_map"] = {
                key: str(value) for key, value in (getattr(self.capture.model, "hf_device_map", None) or {}).items()}
        if architecture == "A4-2":
            self.metadata["reranker_dtype"] = str(next(self.engine.reranker.model.parameters()).dtype)
            self.metadata["reranker_version"] = resources["reranker"]["value"]

    def _load_rag(self):
        import torch
        from experiments.shared.embeddings import BGEM3Embedder
        from experiments.shared.vector_store import FAISSVectorStore
        from experiments.shared.llm_loader import LLMLoader
        from experiments.a5_rag_finetuned_llm.pipeline import validate_loaded_store
        embedder = BGEM3Embedder(model_name=self.resources["bge"]["value"]["path"],
                                device=self.config["device"], normalize_embeddings=True)
        store = FAISSVectorStore(embedding_dim=embedder.embedding_dim,
                                 index_path=str(resolve(self.root, self.config["dense_index"])))
        if not store.load():
            raise DataError("A0/A4 dense index unavailable; no auto-build")
        version_path = resolve(self.root, self.config["dense_index"]) / "version.json"
        from .common import read_json
        validate_loaded_store(store, embedder.embedding_dim, read_json(version_path) if version_path.is_file() else None)
        llm = LLMLoader(model_id=self.resources["base"]["value"]["path"], device=self.config["device"],
                        torch_dtype={"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[self.config["precision"]],
                        max_new_tokens=self.config["max_new_tokens"], do_sample=False)
        llm.model.eval()
        self.capture = GenerationCapture(llm.model, llm.tokenizer, self.config["max_new_tokens"])
        if self.arch == "A0":
            from experiments.baselines.rag_baseline.pipeline import RAGPipeline
            self.engine = RAGPipeline(store, embedder, llm, top_k=5)
            self.metadata.update(prompt_version="baseline_basic_v1", dense_top_k=5)
        else:
            from experiments.baselines.hybrid_rag_reranker.pipeline import HybridRAGRerankerPipeline
            from experiments.baselines.hybrid_rag_reranker.config import settings
            from experiments.baselines.hybrid_rag_reranker.build_bm25_index import chunk_documents
            from experiments.shared.data_loader import load_documents_from_type1_type2
            raw, stats = load_documents_from_type1_type2(str(self.root / "畢業專題_爬蟲/type1"),
                                                        str(self.root / "畢業專題_爬蟲/type2"))
            if stats["failed_parses"]:
                raise DataError("A4 source parsing failures: %r" % stats)
            chunks = chunk_documents(raw, chunk_size=512, chunk_overlap=128)
            fresh = sorted((d.metadata.get("document_id"), d.page_content) for d in chunks)
            dense = sorted((d.metadata.get("document_id"), d.page_content) for d in store.docstore)
            if fresh != dense:
                raise DataError("A4 current BM25 chunks differ from existing dense index; no rebuild or silent mixing")
            settings.llm_device = self.config["device"]
            if self.arch == "A4-2":
                settings.reranker_model = self.resources["reranker"]["value"]["path"]
            self.engine = HybridRAGRerankerPipeline(store, embedder, llm, chunks, top_k=10,
                                                    use_reranker=self.arch == "A4-2")
            self.metadata.update(prompt_version="a4_hybrid_reranker_v1", dense_top_k=10, bm25_top_k=10,
                                 rrf_k=settings.rrf_k, rerank_top_k=10 if self.arch == "A4-2" else None,
                                 llm_top_k=self.engine.llm_top_k, source_stats=stats,
                                 bm25_dense_content_equal=True, bm25_build="in_memory_no_persisted_index_required")
        self.metadata.update(precision=self.config["precision"], max_input_tokens=2048,
                             truncation=True, tokenizer_add_special_tokens="legacy default True")

    def answer(self, query):
        """No faq_id/reference_answer argument exists, so they cannot enter prompts."""
        self.partial = {}
        if self.capture:
            self.capture.last = {}
        if self.arch == "A1":
            answer, score, faq, top2, _ = self.engine.match(query, return_faq=True)
            return {"generated_answer": answer or "", "predicted_faq_id": faq.get("faq_id") if faq else None,
                    "matched_question": faq.get("question") if faq else None, "similarity_score": score,
                    "faq_top2": top2, "query_equals_matched_question": bool(faq and query == faq["question"])}
        if self.arch in ("A2-1", "A2-2"):
            result = self.engine.answer(query)
            return {"generated_answer": result["answer"] or "", "predicted_faq_id": result["faq_id"],
                    "matched_question": result["question"], "similarity_score": result["similarity"],
                    "query_equals_matched_question": query == result["question"]}
        if self.arch == "A3":
            import torch
            messages = [{"role": "system", "content": self.provenance["system_prompt"]},
                        {"role": "user", "content": query}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
            length = inputs["input_ids"].shape[1]
            if length > 512 or length + self.config["max_new_tokens"] > self.model.config.max_position_embeddings:
                raise DataError("A3 input exceeds its normal token limit")
            inputs = {k: v.to(self.config["device"]) for k, v in inputs.items()}
            with torch.inference_mode():
                result = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.config["max_new_tokens"],
                                             pad_token_id=self.tokenizer.pad_token_id)
            answer = self.tokenizer.decode(result[0, length:], skip_special_tokens=True).strip()
        elif self.arch == "A5":
            result = self.engine.answer(query)
            answer = result["answer"]
        else:
            if self.arch == "A0":
                docs, scores = measured(self.engine.retrieve, query)
                self.partial.update(retrieval_seconds=measured.seconds,
                                    retrieved_documents=documents(docs, scores))
            else:
                bm, bs, de, ds, hy, hs = measured(self.engine.retrieve, query)
                self.partial.update(retrieval_seconds=measured.seconds,
                                    bm25_documents=documents(bm, bs), dense_documents=documents(de, ds),
                                    hybrid_documents=documents(hy, hs))
                docs, scores = measured(self.engine.rerank_results, query, hy, hs, top_k=self.engine.top_k)
                self.partial.update(reranking_seconds=measured.seconds,
                                    final_ranked_documents=documents(docs, scores))
                docs, scores = docs[:self.engine.llm_top_k], scores[:self.engine.llm_top_k]
                self.partial["retrieved_documents"] = documents(docs, scores)
            if not docs:
                raise DataError("no retrieved documents; refusing empty-context run")
            answer = measured(self.engine.generate_answer, query, docs, do_sample=False,
                              max_new_tokens=self.config["max_new_tokens"])
            self.partial["generation_stage_seconds"] = measured.seconds
        return {"generated_answer": answer, **self.partial,
                **(self.capture.last if self.capture else {})}
