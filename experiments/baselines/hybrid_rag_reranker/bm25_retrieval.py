"""BM25 retrieval for Hybrid RAG + Reranker (A4).

Implements BM25 retrieval using the same chunk set as the dense retriever
for fair comparison. Uses a standard BM25 implementation with Chinese
text segmentation using jieba for proper word segmentation.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any, Optional

from langchain_core.documents import Document
import numpy as np

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


def _chinese_tokenize(text: str) -> List[str]:
    """Proper Chinese tokenization for BM25 using jieba.

    This function uses jieba for Chinese word segmentation while preserving
    English words and numbers. For better domain-specific results, consider
    adding custom dictionary terms to jieba.

    Args:
        text: Input text to tokenize

    Returns:
        List of tokens (words)
    """
    if not JIEBA_AVAILABLE:
        # Fallback to simple tokenization if jieba is not available
        import re
        # Convert to lowercase for consistency (only as fallback)
        text = text.lower()

        # Split by common punctuation and whitespace
        # This handles both English and Chinese punctuation
        tokens = re.split(r'[\s\-\_\.\,\!\?\;\:\~\@\#\$\%\^\&\*\(\)\+\=\[\]\{\}\|\/\\\"\'\n\r\t]+', text)

        # Filter out empty tokens and very short tokens (length < 2)
        tokens = [token for token in tokens if token and len(token) >= 2]
        return tokens

    # Use jieba for proper Chinese word segmentation
    # jieba.lcut returns a list of words
    tokens = jieba.lcut(text)

    # Filter out empty tokens and very short tokens (length < 1)
    # We keep single character tokens as they might be meaningful in Chinese
    # but filter out only truly empty ones
    tokens = [token for token in tokens if token.strip()]

    return tokens


class BM25Retriever:
    """BM25 retriever that works with the same document set as dense retrieval."""

    def __init__(
        self,
        documents: List[Document],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """Initialize the BM25 retriever.

        Args:
            documents: List of Document objects to index
            k1: BM25 parameter (default: 1.5)
            b: BM25 parameter (default: 0.75)
            epsilon: Parameter for handling negative IDF values (default: 0.25)
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Preprocess documents: tokenize and compute document frequencies
        self._preprocess()

    def _preprocess(self):
        """Preprocess documents for BM25: tokenize and compute document frequencies."""
        # Tokenize all documents
        self.document_tokens: List[List[str]] = []
        for doc in self.documents:
            tokens = _chinese_tokenize(doc.page_content)
            self.document_tokens.append(tokens)

        # Compute document frequencies
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        for tokens in self.document_tokens:
            # Count each term once per document
            seen_terms = set(tokens)
            for term in seen_terms:
                self.doc_freqs[term] += 1

        # Compute inverse document frequencies
        self.doc_count = len(self.documents)
        self.idf: Dict[str, float] = {}
        for term, freq in self.doc_freqs.items():
            # Standard BM25 IDF formula
            self.idf[term] = math.log(
                (self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0
            )

        # Compute average document length
        self.doc_lens: List[int] = [len(tokens) for tokens in self.document_tokens]
        self.avg_doc_len: float = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0

    def get_scores(self, query: str) -> List[float]:
        """Compute BM25 scores for all documents given a query.

        Args:
            query: Query string

        Returns:
            List of BM25 scores (one per document)
        """
        query_tokens = _chinese_tokenize(query)
        scores = [0.0] * self.doc_count

        for term in query_tokens:
            if term not in self.idf:
                continue

            idf = self.idf[term]
            term_freqs = [tokens.count(term) for tokens in self.document_tokens]

            for doc_idx, (tf, doc_len) in enumerate(zip(term_freqs, self.doc_lens)):
                # BM25 formula
                numerator = self.k1 * tf
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score = idf * (numerator / denominator)
                scores[doc_idx] += score

        return scores

    def retrieve(self, query: str, top_k: int = 10) -> Tuple[List[Document], List[float]]:
        """Retrieve top-k documents for a query using BM25.

        Args:
            query: Query string
            top_k: Number of documents to return

        Returns:
            Tuple of (documents, scores) sorted by score descending, with non-positive scores filtered out
        """
        scores = self.get_scores(query)

        # Get indices of top-k scores
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Get documents and scores for top indices, filtering out non-positive scores
        filtered_docs = []
        filtered_scores = []
        for idx in top_indices:
            if scores[idx] > 0:
                filtered_docs.append(self.documents[idx])
                filtered_scores.append(scores[idx])

        return filtered_docs, filtered_scores


class HybridRetriever:
    """Hybrid retriever that combines BM25 and dense retrieval."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: Any,  # Will be Retriever from shared.retrieval
        rrf_k: int = 60,
    ):
        """Initialize the hybrid retriever.

        Args:
            bm25_retriever: BM25Retriever instance
            dense_retriever: Dense retriever instance (from shared.retrieval.Retriever)
            rrf_k: RRF parameter (default: 60)
        """
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> Tuple[
        List[Document],
        List[float],
        List[Document],
        List[float],
        List[Document],
        List[float],
    ]:
        """Retrieve documents using hybrid BM25 + dense with RRF fusion.

        Args:
            query: Query string
            top_k: Number of final documents to return after fusion

        Returns:
            Tuple of:
                - bm25_docs: BM25 retrieved documents
                - bm25_scores: BM25 scores
                - dense_docs: Dense retrieved documents
                - dense_scores: Dense scores
                - fused_docs: Fused documents after RRF
                - fused_scores: Fused scores after RRF
        """
        # Get BM25 results
        bm25_docs, bm25_scores = self.bm25_retriever.retrieve(query, top_k=top_k)

        # Get dense results
        dense_results = self.dense_retriever.retrieve(query)
        dense_docs = [doc for doc, _ in dense_results]
        dense_scores = [score for _, score in dense_results]

        # Apply RRF fusion
        fused_docs, fused_scores = self._apply_rrf(
            bm25_docs, bm25_scores,
            dense_docs, dense_scores,
            top_k=top_k
        )

        return bm25_docs, bm25_scores, dense_docs, dense_scores, fused_docs, fused_scores

    def _apply_rrf(
        self,
        bm25_docs: List[Document],
        bm25_scores: List[float],
        dense_docs: List[Document],
        dense_scores: List[float],
        top_k: int
    ) -> Tuple[List[Document], List[float]]:
        """Apply Reciprocal Rank Fusion to combine rankings.

        Args:
            bm25_docs: BM25 retrieved documents
            bm25_scores: BM25 scores
            dense_docs: Dense retrieved documents
            dense_scores: Dense scores
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
        doc_rrf_scores: Dict[str, float] = defaultdict(float)
        for doc_id in all_doc_ids:
            bm25_rank = bm25_ranks.get(doc_id)
            dense_rank = dense_ranks.get(doc_id)

            if bm25_rank is not None:
                doc_rrf_scores[doc_id] += 1.0 / (self.rrf_k + bm25_rank)
            if dense_rank is not None:
                doc_rrf_scores[doc_id] += 1.0 / (self.rrf_k + dense_rank)

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