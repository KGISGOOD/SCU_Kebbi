"""Reranker module for Hybrid RAG + Reranker (A4).

Implements a cross-encoder reranker using BGE Reranker model to re-rank
retrieved documents based on query-document relevance.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document

try:
    import torch
    from torch.nn.functional import softmax
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False


class Reranker:
    """Cross-encoder reranker using BGE Reranker model."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        """Initialize the reranker.

        Args:
            model_name: Hugging Face model identifier for the reranker
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for encoding
            max_length: Maximum sequence length for tokenizer
        """
        if not RERANKER_AVAILABLE:
            raise ImportError(
                "Reranker requires torch and transformers. "
                "Please install them with: pip install torch transformers"
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            raise RuntimeError(f"Failed to load reranker model {model_name}: {e}") from e

    def compute_scores(self, query: str, documents: List[Document]) -> List[float]:
        """Compute relevance scores for query-document pairs.

        Args:
            query: Query string
            documents: List of Document objects to re-rank

        Returns:
            List of relevance scores (one per document)
        """
        if not documents:
            return []

        # Process in batches
        all_scores = []
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]

            # Prepare query-document pairs for the tokenizer
            # For BGE reranker, we pass query and document as separate text pairs
            queries = [query] * len(batch_docs)
            texts = [doc.page_content for doc in batch_docs]

            # Tokenize batch - correctly handle query-document pairs
            inputs = self.tokenizer(
                queries,
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                # For BGE reranker, the model outputs a single score per pair (relevance score)
                # The logits shape is (batch_size, 1) for binary classification or (batch_size,) for regression
                # BGE reranker-v2-m3 outputs a single logit per pair
                logits = outputs.logits
                if logits.dim() > 1 and logits.shape[1] == 1:
                    logits = logits.squeeze(-1)  # Shape: (batch_size,)

                # We'll use raw logits as scores for simplicity
                # Higher score means more relevant
                scores = logits.cpu().numpy().tolist()

            all_scores.extend(scores)

        return all_scores

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> Tuple[List[Document], List[float]]:
        """Re-rank documents based on query relevance.

        Args:
            query: Query string
            documents: List of Document objects to re-rank
            top_k: Number of top documents to return (None for all)

        Returns:
            Tuple of (reranked_documents, rerank_scores)
        """
        if not documents:
            return [], []

        # Compute relevance scores
        scores = self.compute_scores(query, documents)

        # Create document-score pairs and sort by score descending
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Get top-k if specified
        if top_k is not None:
            doc_score_pairs = doc_score_pairs[:top_k]

        # Unzip documents and scores
        reranked_documents = [doc for doc, _ in doc_score_pairs]
        rerank_scores = [score for _, score in doc_score_pairs]

        return reranked_documents, rerank_scores


def rerank_documents(
    query: str,
    documents: List[Document],
    scores: List[float],
    top_k: int = 5,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 512
) -> Tuple[List[Document], List[float]]:
    """Convenience function to rerank documents.

    Args:
        query: Query string
        documents: List of Document objects to re-rank
        scores: Original retrieval scores (not used in reranking, but kept for compatibility)
        top_k: Number of top documents to return after reranking
        model_name: Hugging Face model identifier for the reranker
        device: Device to use ('cuda', 'cpu', or None for auto)
        batch_size: Batch size for encoding
        max_length: Maximum sequence length for tokenizer

    Returns:
        Tuple of (reranked_documents, rerank_scores)
    """
    reranker = Reranker(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_length=max_length
    )
    return reranker.rerank(query, documents, top_k=top_k)