"""Retrieval module for dense vector search using FAISS.

Provides a retriever that encodes queries and searches the vector store.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

from langchain_core.documents import Document

from .embeddings import BGEM3Embedder
from .vector_store import FAISSVectorStore


class Retriever:
    """Dense retriever using BGE-M3 embeddings and FAISS vector store."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: BGEM3Embedder,
        top_k: int = 5,
    ):
        """Initialize the retriever.

        Args:
            vector_store: FAISS vector store instance.
            embedder: BGE-M3 embedder instance.
            top_k: Number of documents to retrieve.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """Retrieve top-k documents for a query.

        Args:
            query: Query string.

        Returns:
            List of (document, score) tuples, sorted by score descending.
        """
        # Embed the query
        query_embedding = self.embedder.embed_query(query)
        query_array = np.array([query_embedding])

        # Search the vector store
        documents, scores = self.vector_store.similarity_search(query_array, k=self.top_k)

        # Return as list of tuples
        return list(zip(documents, scores))


# Import numpy here to avoid issues if not installed
import numpy as np