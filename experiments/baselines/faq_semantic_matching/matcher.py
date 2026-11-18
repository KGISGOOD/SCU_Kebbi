"""FAQ matcher for the FAQ semantic matching baseline.

Uses BGE-M3 embedding and FAISS IndexFlatIP to match questions to FAQ answers.
"""

from __future__ import annotations

import json
import os
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from langchain_core.documents import Document

from experiments.shared.embeddings import BGEM3Embedder
from experiments.shared.vector_store import FAISSVectorStore
from .config import settings


class FAQMatcher:
    """FAQ matcher that uses semantic similarity to find the best matching FAQ."""



    def _faqs_to_documents(self, faqs: List[Dict[str, Any]]) -> List[Document]:
        """Convert FAQ dictionaries to LangChain Documents for embedding.

        We embed the 'question' field of each FAQ.

        Args:
            faqs: List of FAQ dictionaries.

        Returns:
            List of Document objects.
        """
        documents = []
        for faq in faqs:
            doc = Document(
                page_content=faq['question'],
                metadata={
                    "faq_id": faq.get("faq_id"),
                    "question": faq.get("question"),
                    "answer": faq.get("answer"),
                    "source_file": faq.get("source_file"),
                    "source_type": faq.get("source_type"),
                }
            )
            documents.append(doc)
        return documents



    def match(self, question: str,
              return_faq: bool = False, *, query_vector=None) -> Tuple[Optional[str], float,
                                                 Optional[Dict[str, Any]],
                                                 List[Dict[str, Any]],
                                                 float]:
        """Match a question to the best FAQ.

        Args:
            question: The input question to match.
            return_faq: Whether to return the full FAQ dictionary for the match.

        Returns:
            Tuple of (matched_answer, similarity_score, matched_faq, top_2_faqs, matching_time)
            matched_answer: The answer string if similarity passes threshold, otherwise None.
            similarity_score: Float similarity score of the top-1 FAQ (always returned).
            matched_faq: The full FAQ dictionary for top-1 if return_faq is True, otherwise None.
            top_2_faqs: List of up to two dictionaries, each containing faq_id, question,
                        answer, and similarity_score (float). Empty list if no results.
            matching_time: Time in seconds to embed the question and search the index.
        """
        if not self.index_built:
            print("FAQ index not built. Please build or load the index first.")
            return None, 0.0, None, [], 0.0

        if not question or not question.strip():
            print("Empty question received.")
            return None, 0.0, None, [], 0.0

        start_time = time.perf_counter()

        # Embed the question
        query_embedding = (np.array(self.embedder.embed_query(question), dtype=np.float32)
                           if query_vector is None else np.asarray(query_vector, dtype=np.float32).reshape(-1))

        # Search the vector store for top-k similar FAQs
        # We set top_k to 2 to get the top-1 and top-2 for analysis
        top_k = max(2, settings.top_k)  # Ensure at least 2 for margin calculation
        docs, sims = self.vector_store.similarity_search(query_embedding, top_k)

        end_time = time.perf_counter()
        matching_time = end_time - start_time

        if not docs:
            print("No results returned from vector store.")
            return None, 0.0, None, [], matching_time

        # Extract the top-1 and top-2 results
        top1_doc, top1_score = docs[0], sims[0]
        top2_doc, top2_score = (docs[1], sims[1]) if len(docs) > 1 else (None, 0.0)

        # Get the FAQ ID from the metadata
        top1_faq_id = top1_doc.metadata.get("faq_id")
        top2_faq_id = top2_doc.metadata.get("faq_id") if top2_doc else None

        # Find the full FAQ dictionaries for top-1 and top-2
        top1_faq = None
        top2_faq = None
        for faq in self.faqs:
            if faq.get("faq_id") == top1_faq_id:
                top1_faq = faq
            if top2_faq_id and faq.get("faq_id") == top2_faq_id:
                top2_faq = faq

        # Prepare the top-2 list for return (each as a dict with faq_id, question, answer, score)
        top2_faqs = []
        if top1_faq:
            top2_faqs.append({
                "faq_id": top1_faq.get("faq_id"),
                "question": top1_faq.get("question"),
                "answer": top1_faq.get("answer"),
                "similarity_score": float(top1_score)
            })
        if top2_faq:
            top2_faqs.append({
                "faq_id": top2_faq.get("faq_id"),
                "question": top2_faq.get("question"),
                "answer": top2_faq.get("answer"),
                "similarity_score": float(top2_score)
            })

        # Determine match status based on threshold
        threshold = settings.similarity_threshold
        is_matched = (threshold is None) or (top1_score >= threshold)

        # Prepare the return values
        matched_answer = top1_faq.get("answer") if is_matched and top1_faq else None
        similarity_score = float(top1_score)  # always return similarity score
        matched_faq = top1_faq if return_faq else None  # always return top-1 FAQ if requested

        return matched_answer, similarity_score, matched_faq, top2_faqs, matching_time
