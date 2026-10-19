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
from .faq_loader import load_all_faq, save_faq_as_jsonl


class FAQMatcher:
    """FAQ matcher that uses semantic similarity to find the best matching FAQ."""

    def __init__(self, faq_dataset_path: Optional[str] = None,
                 faq_vector_store_dir: Optional[str] = None):
        """Initialize the FAQ matcher.

        Args:
            faq_dataset_path: Path to the FAQ JSONL dataset.
            faq_vector_store_dir: Path to the FAQ vector store directory.
        """
        # Use settings if not provided
        self.faq_dataset_path = faq_dataset_path or settings.faq_dataset_path
        self.faq_vector_store_dir = faq_vector_store_dir or settings.faq_vector_store_dir

        # Load FAQ data
        self.faqs: List[Dict[str, Any]] = self._load_faqs()
        self.faq_documents: List[Document] = self._faqs_to_documents(self.faqs)

        # Initialize embedder and vector store
        self.embedder = BGEM3Embedder(
            model_name=settings.embedding_model,
            # Reuse device settings from the RAG baseline config if available, else auto-detect
            # We'll try to import the RAG baseline settings for device, but if not available, use None
            device=None,  # We'll let the embedder auto-detect or we can set later
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=settings.embedding_normalize,
        )

        self.vector_store = FAISSVectorStore(
            embedding_dim=self.embedder.embedding_dim,
            index_path=self.faq_vector_store_dir,
            use_hnsw=settings.use_hnsw,
            hnsw_M=settings.hnsw_M,
            hnsw_efConstruction=settings.hnsw_efConstruction,
        )

        # Flag to track if the index has been loaded/built
        self.index_built = False

    def _load_faqs(self) -> List[Dict[str, Any]]:
        """Load FAQs from the JSONL dataset file.

        Returns:
            List of FAQ dictionaries.
        """
        if not os.path.exists(self.faq_dataset_path):
            print(f"FAQ dataset not found at {self.faq_dataset_path}. "
                  "Please run the faq_loader to generate it first.")
            return []

        faqs = []
        try:
            with open(self.faq_dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        faq = json.loads(line)
                        faqs.append(faq)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON line: {e}")
            print(f"Loaded {len(faqs)} FAQs from {self.faq_dataset_path}")
        except Exception as e:
            print(f"Error loading FAQs from {self.faq_dataset_path}: {e}")
        return faqs

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
                    "review_status": faq.get("review_status"),
                }
            )
            documents.append(doc)
        return documents

    def build_index(self) -> None:
        """Build the FAISS index from the FAQ questions.

        This computes embeddings for all FAQ questions and builds the index.
        """
        if not self.faq_documents:
            print("No FAQ documents to build index from.")
            return

        print(f"Building FAQ index from {len(self.faq_documents)} FAQs...")
        start_time = time.time()

        # Compute embeddings for FAQ questions
        questions = [doc.page_content for doc in self.faq_documents]
        embeddings = self.embedder.encode(questions)

        # Build index
        self.vector_store.build(embeddings, self.faq_documents)

        # Save the index
        self.vector_store.save()

        end_time = time.time()
        print(f"FAQ index built and saved in {end_time - start_time:.2f} seconds.")
        self.index_built = True

    def load_index(self) -> bool:
        """Load the FAISS index from disk.

        Returns:
            True if index loaded successfully, False otherwise.
        """
        try:
            success = self.vector_store.load()
            if success:
                print(f"Loaded FAQ index from {self.faq_vector_store_dir}")
                self.index_built = True
                return True
            else:
                print(f"No existing FAQ index found at {self.faq_vector_store_dir}")
                self.index_built = False
                return False
        except Exception as e:
            print(f"Error loading FAQ index: {e}")
            self.index_built = False
            return False

    def match(self, question: str,
              return_faq: bool = False) -> Tuple[Optional[str], Optional[float],
                                                 Optional[Dict[str, Any]],
                                                 Optional[List[Dict[str, Any]]]]:
        """Match a question to the best FAQ.

        Args:
            question: The input question to match.
            return_faq: Whether to return the full FAQ dictionary for the match.

        Returns:
            Tuple of (matched_answer, similarity_score, matched_faq, [top_2_faqs])
            If no match is found (or index not built), returns (None, None, None, None).
            If return_faq is True, the matched_faq will be the full FAQ dictionary.
            The second element in the tuple is the similarity score of the top match.
            The fourth element is a list of the top 2 FAQs (each as a dict) for analysis.
        """
        if not self.index_built:
            print("FAQ index not built. Please build or load the index first.")
            return None, None, None, None

        if not question or not question.strip():
            print("Empty question received.")
            return None, None, None, None

        start_time = time.time()

        # Embed the question
        query_embedding = np.array(self.embedder.embed_query(question), dtype=np.float32)

        # Search the vector store for top-k similar FAQs
        # We set top_k to 2 to get the top-1 and top-2 for analysis
        top_k = max(2, settings.top_k)  # Ensure at least 2 for margin calculation
        docs, sims = self.vector_store.similarity_search(query_embedding, top_k)

        end_time = time.time()
        matching_time = end_time - start_time

        if not docs:
            print("No results returned from vector store.")
            return None, None, None, None

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
        is_matched = threshold is not None and top1_score >= threshold

        # Prepare the return values
        matched_answer = top1_faq.get("answer") if is_matched and top1_faq else None
        similarity_score = float(top1_score) if is_matched else None
        matched_faq = top1_faq if return_faq and is_matched else None

        # If not matched, we still want to return the top-2 for analysis? The spec says:
        #   matched：通過門檻，回傳標準 FAQ 答案。
        #   no_match：未通過門檻，回傳固定無匹配訊息。
        #   error：程式執行或資料異常.
        # So we return the answer only if matched, otherwise we return a fixed message?
        # But the function returns the answer, and the caller can decide what to do.
        # We'll return the answer as None if not matched, and the caller can use the top2_faqs to show alternatives.

        return matched_answer, similarity_score, matched_faq, top2_faqs, matching_time
