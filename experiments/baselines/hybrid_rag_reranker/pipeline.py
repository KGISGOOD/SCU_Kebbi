"""Hybrid RAG + Reranker pipeline (A4).

Combines BM25 retrieval, dense retrieval, RRF fusion, optional reranking,
and LLM generation for the A4 baseline.
"""

from __future__ import annotations

import time
from typing import List, Tuple, Optional, Dict, Any

from langchain_core.documents import Document

from .config import settings
from .bm25_retrieval import BM25Retriever, HybridRetriever
from .fusion import reciprocal_rank_fusion
from .reranker import Reranker, rerank_documents
from experiments.shared.embeddings import BGEM3Embedder
from experiments.shared.vector_store import FAISSVectorStore
from experiments.shared.retrieval import Retriever
from experiments.shared.llm_loader import LLMLoader
from experiments.shared.evaluation import timer


class HybridRAGRerankerPipeline:
    """Hybrid RAG + Reranker pipeline."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: BGEM3Embedder,
        llm_loader: LLMLoader,
        documents: List[Document],
        top_k: Optional[int] = None,
        use_reranker: bool = False,
    ):
        """Initialize the Hybrid RAG + Reranker pipeline.

        Args:
            vector_store: FAISS vector store instance (dense retrieval)
            embedder: BGE-M3 embedder instance
            llm_loader: LLM loader instance
            documents: List of Document objects (same for BM25 and dense)
            top_k: Number of documents to retrieve (overrides config)
            use_reranker: Whether to use reranker after fusion
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_loader = llm_loader
        self.top_k = top_k if top_k is not None else settings.top_k
        self.use_reranker = use_reranker
        self.llm_top_k = settings.reranker_top_k  # Number of documents to pass to LLM after fusion/reranking

        # Initialize retrievers
        self.dense_retriever = Retriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            top_k=self.top_k,
        )

        self.bm25_retriever = BM25Retriever(
            documents=documents,
            k1=1.5,
            b=0.75,
            epsilon=0.25,
        )

        self.hybrid_retriever = HybridRetriever(
            bm25_retriever=self.bm25_retriever,
            dense_retriever=self.dense_retriever,
            rrf_k=settings.rrf_k,
        )

        # Initialize reranker if needed
        if self.use_reranker:
            self.reranker = Reranker(
                model_name=settings.reranker_model,
                device=settings.llm_device,  # Reuse device setting
                batch_size=settings.reranker_batch_size,
                max_length=settings.reranker_max_length,
            )
        else:
            self.reranker = None

        # Initialize LLM loader (already passed in)
        self.llm_loader = llm_loader

    def retrieve(
        self, query: str
    ) -> Tuple[
        List[Document], List[float],  # BM25 results
        List[Document], List[float],  # Dense results
        List[Document], List[float]   # Hybrid (RRF) results
    ]:
        """Retrieve documents using hybrid BM25 + dense with RRF fusion.

        Args:
            query: Query string

        Returns:
            Tuple of (bm25_docs, bm25_scores, dense_docs, dense_scores, hybrid_docs, hybrid_scores)
        """
        return self.hybrid_retriever.retrieve(query, top_k=self.top_k)

    def rerank_results(
        self,
        query: str,
        documents: List[Document],
        scores: List[float],
        top_k: Optional[int] = None
    ) -> Tuple[List[Document], List[float]]:
        """Re-rank documents using the reranker.

        Args:
            query: Query string
            documents: List of Document objects to re-rank
            scores: Original scores (not used in reranking)
            top_k: Number of top documents to return (None for all)

        Returns:
            Tuple of (reranked_documents, rerank_scores)
        """
        if not self.use_reranker or self.reranker is None:
            # Return original documents and scores if reranker not used
            if top_k is not None:
                return documents[:top_k], scores[:top_k]
            return documents, scores

        return self.reranker.rerank(query, documents, top_k=top_k)

    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Document],
        **llm_kwargs,
    ) -> str:
        """Generate answer from query and retrieved documents.

        Args:
            query: User query
            retrieved_docs: Retrieved documents
            **llm_kwargs: Additional arguments for LLM generation

        Returns:
            Generated answer string
        """
        # Import here to avoid circular imports
        from .prompts import build_prompt, create_context_from_documents

        # Limit to top-k documents for LLM input to avoid excessive length and ensure concise answer
        llm_docs = retrieved_docs[:self.llm_top_k]

        # Create context from documents
        context = create_context_from_documents(llm_docs)
        # Build prompt
        prompt = build_prompt(context, query)
        # Generate answer
        answer = self.llm_loader.generate(prompt, **llm_kwargs)
        return answer

    def __call__(
        self,
        query: str,
        return_context: bool = False,
        **llm_kwargs,
    ) -> Tuple[
        str,  # answer
        Optional[List[Document]],  # retrieved_documents
        Optional[List[float]],     # retrieval_scores
        Dict[str, Any]             # retrieval_info
    ]:
        """Run the full Hybrid RAG + Reranker pipeline.

        Args:
            query: User query
            return_context: Whether to return retrieved documents and scores
            **llm_kwargs: Additional arguments for LLM generation

        Returns:
            Tuple of (answer, retrieved_documents, scores, retrieval_info) if return_context=True,
            otherwise just (answer, None, None, {})
        """
        start_time = time.time()

        # Retrieve using hybrid approach
        retrieval_start = time.perf_counter()
        bm25_docs, bm25_scores, dense_docks, dense_scores, hybrid_docs, hybrid_scores = self.retrieve(query)
        retrieval_end = time.perf_counter()
        retrieval_time = retrieval_end - retrieval_start

        # Apply reranker if enabled
        reranking_start = time.perf_counter()
        if self.use_reranker:
            final_docs, final_scores = self.rerank_results(
                query, hybrid_docs, hybrid_scores, top_k=self.top_k
            )
        else:
            final_docs, final_scores = hybrid_docs, hybrid_scores
        reranking_end = time.perf_counter()
        reranking_time = reranking_end - reranking_start

        # Generate answer
        generation_start = time.perf_counter()
        answer = self.generate_answer(query, final_docs, **llm_kwargs)
        generation_end = time.perf_counter()
        generation_time = generation_end - generation_start

        end_time = time.time()
        total_time = end_time - start_time

        if settings.verbose:
            print(f"Query processed in {total_time:.2f}s")
            print(f"  Retrieval: {retrieval_time:.3f}s")
            print(f"  Reranking: {reranking_time:.3f}s")
            print(f"  Generation: {generation_time:.3f}s")

        # Prepare retrieval info
        retrieval_info = {
            "bm25_docs": bm25_docs,
            "bm25_scores": bm25_scores,
            "dense_docs": dense_docs,
            "dense_scores": dense_scores,
            "hybrid_docs": hybrid_docs,
            "hybrid_scores": hybrid_scores,
            "final_docs": final_docs,
            "final_scores": final_scores,
            "retrieval_time": retrieval_time,
            "reranking_time": reranking_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "use_reranker": self.use_reranker,
        }

        if return_context:
            return answer, final_docs, final_scores, retrieval_info
        else:
            return answer, None, None, retrieval_info


def create_pipeline_from_config(
    use_reranker: bool = False,
    documents: Optional[List[Document]] = None
) -> HybridRAGRerankerPipeline:
    """Create a Hybrid RAG + Reranker pipeline instance from the global settings.

    Args:
        use_reranker: Whether to use reranker after fusion
        documents: List of Document objects (required for BM25 retriever)

    Returns:
        Initialized HybridRAGRerankerPipeline
    """
    if documents is None:
        raise ValueError("Documents must be provided for BM25 retriever initialization")

    # Initialize embedder
    embedder = BGEM3Embedder(
        model_name=settings.embedding_model,
        device=settings.llm_device,  # Reuse device setting for consistency
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=settings.embedding_normalize,
    )

    # Initialize or load vector store (dense)
    vector_store = FAISSVectorStore(
        embedding_dim=embedder.embedding_dim,
        index_path=settings.dense_vector_store_dir,
        use_hnsw=settings.use_hnsw,
        hnsw_M=settings.hnsw_M,
        hnsw_efConstruction=settings.hnsw_efConstruction,
    )

    # Try to load existing index
    if not vector_store.load():
        if settings.verbose:
            print(
                f"No existing index found at {settings.dense_vector_store_dir}. "
                "Please run build_index.py first."
            )
        # We still create the vector store object; it will be empty until built
        # The pipeline will fail if used without building

    # Initialize LLM loader
    llm_loader = LLMLoader(
        model_id=settings.llm_model_id,
        device=settings.llm_device,
        load_in_8bit=settings.llm_load_in_8bit,
        load_in_4bit=settings.llm_load_in_4bit,
        max_new_tokens=settings.llm_max_new_tokens,
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        do_sample=settings.llm_do_sample,
        repetition_penalty=settings.llm_repetition_penalty,
        trust_remote_code=settings.llm_trust_remote_code,
    )

    # Create pipeline
    pipeline = HybridRAGRerankerPipeline(
        vector_store=vector_store,
        embedder=embedder,
        llm_loader=llm_loader,
        documents=documents,
        top_k=settings.top_k,
        use_reranker=use_reranker,
    )

    return pipeline