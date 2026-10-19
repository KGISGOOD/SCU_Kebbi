"""Fusion module for Hybrid RAG + Reranker (A4).

Implements Reciprocal Rank Fusion (RRF) and other fusion methods
for combining BM25 and dense retrieval results.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document


def reciprocal_rank_fusion(
    bm25_docs: List[Document],
    bm25_scores: List[float],
    dense_docs: List[Document],
    dense_scores: List[float],
    k: int = 60,
    top_k: int = 10
) -> Tuple[List[Document], List[float]]:
    """Apply Reciprocal Rank Fusion to combine rankings.

    Args:
        bm25_docs: BM25 retrieved documents
        bm25_scores: BM25 scores (not used in RRF ranking, but kept for compatibility)
        dense_docs: Dense retrieved documents
        dense_scores: Dense scores (not used in RRF ranking, but kept for compatibility)
        k: RRF parameter (default: 60)
        top_k: Number of final documents to return

    Returns:
        Tuple of (fused_documents, fused_scores)
    """
    # Create document ID mapping for deduplication
    # We'll use document_id from metadata if available, otherwise fallback to content hash
    def get_doc_id(doc: Document) -> str:
        doc_id = doc.metadata.get("document_id")
        if doc_id:
            return doc_id
        # Fallback: hash of content
        return str(hash(doc.page_content))

    # Create rank mappings
    bm25_ranks: Dict[str, int] = {}
    for rank, doc in enumerate(bm25_docs, start=1):
        doc_id = get_doc_id(doc)
        bm25_ranks[doc_id] = rank

    dense_ranks: Dict[str, int] = {}
    for rank, doc in enumerate(dense_docs, start=1):
        doc_id = get_doc_id(doc)
        dense_ranks[doc_id] = rank

    # Get all unique documents from both rankings
    all_doc_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())

    # Compute RRF scores
    doc_rrf_scores: Dict[str, float] = {}
    for doc_id in all_doc_ids:
        score = 0.0
        bm25_rank = bm25_ranks.get(doc_id)
        dense_rank = dense_ranks.get(doc_id)

        if bm25_rank is not None:
            score += 1.0 / (k + bm25_rank)
        if dense_rank is not None:
            score += 1.0 / (k + dense_rank)

        doc_rrf_scores[doc_id] = score

    # Sort documents by RRF score descending
    sorted_doc_ids = sorted(
        doc_rrf_scores.keys(),
        key=lambda x: doc_rrf_scores[x],
        reverse=True
    )

    # Get top-k documents
    top_doc_ids = sorted_doc_ids[:top_k]

    # Create mapping from doc_id to document
    id_to_doc: Dict[str, Document] = {}
    for doc in bm25_docs + dense_docs:
        doc_id = get_doc_id(doc)
        if doc_id not in id_to_doc:
            id_to_doc[doc_id] = doc

    # Get final documents and scores
    fused_docs = [id_to_doc[doc_id] for doc_id in top_doc_ids if doc_id in id_to_doc]
    fused_scores = [doc_rrf_scores[doc_id] for doc_id in top_doc_ids if doc_id in id_to_doc]

    return fused_docs, fused_scores


def score_based_fusion(
    bm25_docs: List[Document],
    bm25_scores: List[float],
    dense_docs: List[Document],
    dense_scores: List[float],
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
    top_k: int = 10
) -> Tuple[List[Document], List[float]]:
    """Apply score-based fusion (weighted combination) to combine results.

    Args:
        bm25_docs: BM25 retrieved documents
        bm25_scores: BM25 scores
        dense_docs: Dense retrieved documents
        dense_scores: Dense scores
        bm25_weight: Weight for BM25 scores (default: 0.5)
        dense_weight: Weight for dense scores (default: 0.5)
        top_k: Number of final documents to return

    Returns:
        Tuple of (fused_documents, fused_scores)
    """
    # Create document ID mapping for deduplication
    def get_doc_id(doc: Document) -> str:
        doc_id = doc.metadata.get("document_id")
        if doc_id:
            return doc_id
        # Fallback: hash of content
        return str(hash(doc.page_content))

    # Collect all unique documents with their scores
    doc_scores: Dict[str, float] = {}

    # Add BM25 scores
    for doc, score in zip(bm25_docs, bm25_scores):
        doc_id = get_doc_id(doc)
        if doc_id not in doc_scores:
            doc_scores[doc_id] = 0.0
        doc_scores[doc_id] += bm25_weight * score

    # Add dense scores
    for doc, score in zip(dense_docs, dense_scores):
        doc_id = get_doc_id(doc)
        if doc_id not in doc_scores:
            doc_scores[doc_id] = 0.0
        doc_scores[doc_id] += dense_weight * score

    # Sort documents by combined score descending
    sorted_doc_ids = sorted(
        doc_scores.keys(),
        key=lambda x: doc_scores[x],
        reverse=True
    )

    # Get top-k documents
    top_doc_ids = sorted_doc_ids[:top_k]

    # Create mapping from doc_id to document
    id_to_doc: Dict[str, Document] = {}
    for doc in bm25_docs + dense_docs:
        doc_id = get_doc_id(doc)
        if doc_id not in id_to_doc:
            id_to_doc[doc_id] = doc

    # Get final documents and scores
    fused_docs = [id_to_doc[doc_id] for doc_id in top_doc_ids if doc_id in id_to_doc]
    fused_scores = [doc_scores[doc_id] for doc_id in top_doc_ids if doc_id in id_to_doc]

    return fused_docs, fused_scores