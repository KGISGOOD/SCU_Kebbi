"""Data-contract adapters only. Retrieval, matching and generation reuse project code."""
from __future__ import annotations

from .common import path


def architecture_config(root, config, allow_provisional):
    m = config["models"]
    return {"base_model": str(path(root, m["base"])), "bge_model": str(path(root, m["bge"])),
            "a3_adapter": str(path(root, m["a3"])), "dense_index": config["dense_index"],
            "device": config["device"], "precision": config["qa_precision"], "seed": config["seed"],
            "max_new_tokens": config["max_new_tokens"], "a5_max_input_tokens": config["max_input_tokens"],
            "allow_unreviewed": allow_provisional}


def a0_config(config, precision=None):
    return {"model": config["models"]["base"], "bge": config["models"]["bge"],
            "knowledge_index": config["dense_index"], "device": config["device"],
            "precision": precision or config["qa_precision"], "seed": config["seed"],
            "max_input_tokens": config["max_input_tokens"], "max_new_tokens": config["max_new_tokens"]}


def sd_config(config, integrated=False):
    return dict(a0_config(config, config["int_precision"] if integrated else config["sd_precision"]),
                draft_model=config["models"]["draft"], standard_draft_tokens=4,
                prompt_lookup_tokens=4, max_matching_ngram_size=2)


class QAEngine:
    def __init__(self, architecture, root, config, resources, canonical, allow_provisional):
        self.architecture = architecture
        self.capture = None
        if architecture not in {"A1", "A2-1", "A2-2"}:
            from experiments.evaluation.wrappers import Architecture
            self.engine = Architecture(architecture, root, architecture_config(root, config, allow_provisional), resources)
            self.metadata = self.engine.metadata
            if architecture in {"A0", "A4-1", "A4-2"}:
                llm = self.engine.engine.llm_loader
                context = llm.model.config.max_position_embeddings
                if config["max_input_tokens"] + config["max_new_tokens"] > context:
                    raise ValueError("configured full prompt plus generation exceeds QA model context window")
                llm.strict_max_input_tokens = config["max_input_tokens"]
                self.metadata.update(max_input_tokens=config["max_input_tokens"], truncation=False,
                                     model_context_window=context)
            return
        from experiments.shared.embeddings import BGEM3Embedder
        key = {"A1": "bge", "A2-1": "a2_1", "A2-2": "a2_2"}[architecture]
        embedder = BGEM3Embedder(model_name=str(path(root, config["models"][key])),
                                device=config["device"], batch_size=16, normalize_embeddings=True)
        embedder.model.max_seq_length = 512
        embedder.model.float().eval()
        self.metadata = {"architecture": architecture, "candidate_count": len(canonical),
                         "candidate_source": "frozen V2 canonical intents, never 804 or Gold",
                         "model": resources[key], "precision": "float32", "max_length": 512,
                         "threshold": config["faq_threshold"], "generation_seconds": "not_applicable"}
        if architecture == "A1":
            from experiments.baselines.faq_semantic_matching.matcher import FAQMatcher
            from experiments.baselines.faq_semantic_matching.config import settings
            from experiments.shared.vector_store import FAISSVectorStore
            settings.similarity_threshold = config["faq_threshold"]
            matcher = FAQMatcher.__new__(FAQMatcher)
            matcher.faqs, matcher.embedder = canonical, embedder
            matcher.faq_documents = matcher._faqs_to_documents(canonical)
            # Build only the same existing normalized IndexFlatIP IN MEMORY.
            # No persisted index constructor/save and no test queries in candidates.
            store = FAISSVectorStore.__new__(FAISSVectorStore)
            store.embedding_dim, store.use_hnsw = embedder.embedding_dim, False
            store.build(embedder.encode([r["question"] for r in canonical]), matcher.faq_documents)
            matcher.vector_store, matcher.index_built = store, True
            self.engine = matcher
        else:
            from experiments.a2_faq_finetuning.faq_answering import FAQAnswerer
            engine = FAQAnswerer.__new__(FAQAnswerer)
            engine.corpus, engine.answers = canonical, {r["faq_id"]: r for r in canonical}
            engine.model_tag, engine.model_name = architecture, str(path(root, config["models"][key]))
            engine.provisional = resources[key]["value"]["provenance"]["provisional"]
            engine.threshold, engine.device, engine.embedder = config["faq_threshold"], config["device"], embedder
            engine.ids = tuple(r["faq_id"] for r in canonical)
            engine._vectors = engine._encode([r["question"] for r in canonical])
            self.engine = engine

    def answer(self, query):
        if self.architecture == "A1":
            answer, score, faq, top2, seconds = self.engine.match(query, return_faq=True)
            return {"generated_answer": answer or "", "predicted_faq_id": faq["faq_id"] if faq else None,
                    "similarity_score": score, "faq_top2": top2, "retrieval_seconds": seconds,
                    "generation_seconds": None, "generation_metric_status": "not_applicable"}
        if self.architecture in {"A2-1", "A2-2"}:
            result = self.engine.answer(query)
            return {"generated_answer": result["answer"] or "", "predicted_faq_id": result["faq_id"],
                    "matched_question": result["question"], "similarity_score": result["similarity"],
                    "generation_seconds": None, "generation_metric_status": "not_applicable"}
        return self.engine.answer(query)


def stable_engine_identity(engine):
    """Remove only observed load-duration/memory metadata from Cache namespace.

    Actual full identity is saved separately; model/config/code/Prompt all remain.
    Without this adapter, reloading an INT worker changes its Cache namespace.
    """
    engine.identity = {k: v for k, v in engine.identity.items()
                       if k not in {"loading_seconds", "fp32_memory_after_loading"}}
