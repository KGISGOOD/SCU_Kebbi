"""RAG pipeline for the baseline.

Combines retrieval, prompt construction, and LLM generation.
"""

from __future__ import annotations

import time
from typing import List, Tuple, Optional

from langchain_core.documents import Document

from .config import settings
from .prompts import build_prompt, create_context_from_documents
from experiments.shared.evaluation import timer
from experiments.shared.embeddings import BGEM3Embedder
from experiments.shared.vector_store import FAISSVectorStore
from experiments.shared.retrieval import Retriever
from experiments.shared.llm_loader import LLMLoader


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: BGEM3Embedder,
        llm_loader: LLMLoader,
        top_k: Optional[int] = None,
    ):
        """Initialize the RAG pipeline.

        Args:
            vector_store: FAISS vector store instance.
            embedder: BGE-M3 embedder instance.
            llm_loader: LLM loader instance.
            top_k: Number of documents to retrieve (overrides config).
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_loader = llm_loader
        self.top_k = top_k if top_k is not None else settings.top_k
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            top_k=self.top_k,
        )

    def retrieve(
        self, query: str
    ) -> Tuple[List[Document], List[float]]:
        """Retrieve documents for a query.

        Args:
            query: Query string.

        Returns:
            Tuple of (documents, scores).
        """
        # Use the retriever which correctly converts query embedding to numpy array
        retriever_results = self.retriever.retrieve(query)
        documents = [doc for doc, _ in retriever_results]
        scores = [score for _, score in retriever_results]
        return documents, scores

    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Document],
        **llm_kwargs,
    ) -> str:
        """Generate answer from query and retrieved documents.

        Args:
            query: User query.
            retrieved_docs: Retrieved documents.
            **llm_kwargs: Additional arguments for LLM generation.

        Returns:
            Generated answer string.
        """
        # Create context from documents
        context = create_context_from_documents(retrieved_docs)
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
    ) -> Tuple[str, Optional[List[Document]], Optional[List[float]]]:
        """Run the full RAG pipeline.

        Args:
            query: User query.
            return_context: Whether to return retrieved documents and scores.
            **llm_kwargs: Additional arguments for LLM generation.

        Returns:
            Tuple of (answer, retrieved_documents, scores) if return_context=True,
            otherwise just (answer, None, None).
        """
        start_time = time.time()

        # Retrieve
        retrieved_docs, scores = self.retrieve(query)

        # Generate answer
        answer = self.generate_answer(query, retrieved_docs, **llm_kwargs)

        end_time = time.time()
        if settings.verbose:
            print(f"Query processed in {end_time - start_time:.2f}s")

        if return_context:
            return answer, retrieved_docs, scores
        else:
            return answer, None, None


def create_pipeline_from_config() -> RAGPipeline:
    """Create a RAG pipeline instance from the global settings.

    Returns:
        Initialized RAGPipeline.
    """
    # Initialize embedder
    embedder = BGEM3Embedder(
        model_name=settings.embedding_model,
        device=settings.llm_device,  # Reuse device setting for consistency
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=settings.embedding_normalize,
    )

    # Initialize or load vector store
    vector_store = FAISSVectorStore(
        embedding_dim=embedder.embedding_dim,
        index_path=settings.vector_store_dir,
        use_hnsw=settings.use_hnsw,
        hnsw_M=settings.hnsw_M,
        hnsw_efConstruction=settings.hnsw_efConstruction,
    )

    # Try to load existing index; if not available, we'll need to build it separately
    if not vector_store.load():
        if settings.verbose:
            print(
                f"No existing index found at {settings.vector_store_dir}. "
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
    pipeline = RAGPipeline(
        vector_store=vector_store,
        embedder=embedder,
        llm_loader=llm_loader,
        top_k=settings.top_k,
    )

    return pipeline


if __name__ == "__main__":
    # Simple test (will fail if index not built or model not available)
    try:
        pipeline = create_pipeline_from_config()
        print("Pipeline created successfully")
        if pipeline.vector_store.get_document_count() > 0:
            print(f"Vector store contains {pipeline.vector_store.get_document_count()} documents")
            # Test retrieval
            docs, scores = pipeline.retrieve("測試問題")
            print(f"Retrieved {len(docs)} documents for test query")
        else:
            print("Vector store is empty - please build index first")
    except Exception as e:
        print(f"Failed to create pipeline: {e}")