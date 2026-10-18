"""FAISS vector store with version management and persistence.

Handles building, saving, loading, and validating FAISS indices.
"""

from __future__ import annotations

import json
import os
import pickle
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import faiss
import numpy as np
from langchain_core.documents import Document

from .embeddings import BGEM3Embedder


class FAISSVectorStore:
    """FAISS vector store with metadata persistence and versioning."""

    def __init__(
        self,
        embedding_dim: int,
        index_path: str,
        use_hnsw: bool = False,
        hnsw_M: int = 32,
        hnsw_efConstruction: int = 200,
    ):
        """Initialize the vector store.

        Args:
            embedding_dim: Dimension of embeddings.
            index_path: Directory to save/load index files.
            use_hnsw: Whether to use HNSW index (for larger datasets).
            hnsw_M: HNSW parameter.
            hnsw_efConstruction: HNSW parameter.
        """
        self.embedding_dim = embedding_dim
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.use_hnsw = use_hnsw
        self.hnsw_M = hnsw_M
        self.hnsw_efConstruction = hnsw_efConstruction

        self.index: Optional[faiss.Index] = None
        self.docstore: List[Document] = []  # List of documents
        self.index_to_docstore_id: Dict[int, int] = {}  # FAISS index -> docstore index

    def _get_index_files(self) -> Tuple[Path, Path]:
        """Get paths for index and docstore files."""
        index_file = self.index_path / "index.faiss"
        docstore_file = self.index_path / "index.pkl"
        return index_file, docstore_file

    def _get_version_file(self) -> Path:
        """Get path for version file."""
        return self.index_path / "version.json"

    def _compute_data_fingerprint(self, documents: List[Document]) -> str:
        """Compute a fingerprint of the documents for version validation."""
        # Sort by document_id for consistent fingerprinting
        sorted_docs = sorted(documents, key=lambda d: d.metadata.get("document_id", ""))
        # Concatenate document_id and page_content
        content_to_hash = ""
        for doc in sorted_docs:
            doc_id = doc.metadata.get("document_id", "")
            content_to_hash += f"{doc_id}:{doc.page_content}\n"
        return hashlib.md5(content_to_hash.encode("utf-8")).hexdigest()

    def _save_version_info(
        self,
        documents: List[Document],
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        source_directories: List[str],
    ) -> None:
        """Save version information to version.json."""
        version_data = {
            "embedding_model": embedding_model,
            "embedding_dimension": self.embedding_dim,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "source_directories": source_directories,
            "document_count": len(set(d.metadata.get("document_id", "") for d in documents)),
            "chunk_count": len(documents),
            "data_fingerprint": self._compute_data_fingerprint(documents),
            "index_type": "IndexFlatIP" if not self.use_hnsw else "IndexHNSWFlat",
            "normalization": "L2",
        }
        version_file = self._get_version_file()
        with open(version_file, "w", encoding="utf-8") as f:
            json.dump(version_data, f, indent=2, ensure_ascii=False)

    def _load_version_info(self) -> Optional[Dict[str, Any]]:
        """Load version information from version.json."""
        version_file = self._get_version_file()
        if not version_file.exists():
            return None
        try:
            with open(version_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def is_valid(
        self,
        documents: List[Document],
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        source_directories: List[str],
    ) -> bool:
        """Check if the existing index is valid for the given parameters and data."""
        version_info = self._load_version_info()
        if version_info is None:
            return False

        # Check basic parameters
        if version_info.get("embedding_model") != embedding_model:
            return False
        if version_info.get("embedding_dimension") != self.embedding_dim:
            return False
        if version_info.get("chunk_size") != chunk_size:
            return False
        if version_info.get("chunk_overlap") != chunk_overlap:
            return False
        # Check source directories (order-independent)
        if set(version_info.get("source_directories", [])) != set(source_directories):
            return False

        # Check data fingerprint
        current_fingerprint = self._compute_data_fingerprint(documents)
        if version_info.get("data_fingerprint") != current_fingerprint:
            return False

        return True

    def build(self, embeddings: np.ndarray, documents: List[Document]) -> None:
        """Build the FAISS index from embeddings and documents.

        Args:
            embeddings: Numpy array of shape (n, embedding_dim).
            documents: List of Document objects corresponding to embeddings.
        """
        if len(embeddings) != len(documents):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) does not match "
                f"number of documents ({len(documents)})"
            )

        # Normalize embeddings (if not already normalized) for inner product = cosine
        faiss.normalize_L2(embeddings)

        # Create index
        if self.use_hnsw and len(embeddings) >= 1000:  # Use HNSW for larger datasets
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, self.hnsw_M)
            self.index.hnsw.efConstruction = self.hnsw_efConstruction
            self.index.hnsw.efSearch = 64  # default for search
        else:
            # Use IndexFlatIP for inner product (cosine similarity when normalized)
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Add to index
        self.index.add(embeddings.astype(np.float32))
        self.docstore = documents
        # Create mapping from index position to docstore position
        self.index_to_docstore_id = {i: i for i in len(documents)}

    def save(self) -> None:
        """Save the index and docstore to disk."""
        if self.index is None:
            raise RuntimeError("Index has not been built yet")

        index_file, docstore_file = self._get_index_files()
        # Save FAISS index
        faiss.write_index(self.index, str(index_file))
        # Save docstore and mapping
        with open(docstore_file, "wb") as f:
            pickle.dump(
                {
                    "docstore": self.docstore,
                    "index_to_docstore_id": self.index_to_docstore_id,
                },
                f,
            )

    def load(self) -> bool:
        """Load the index and docstore from disk.

        Returns:
            True if loaded successfully, False otherwise.
        """
        index_file, docstore_file = self._get_index_files()
        if not index_file.exists() or not docstore_file.exists():
            return False

        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            # Load docstore and mapping
            with open(docstore_file, "rb") as f:
                data = pickle.load(f)
                self.docstore = data["docstore"]
                self.index_to_docstore_id = data["index_to_docstore_id"]
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    def similarity_search(
        self, query_embedding: np.ndarray, k: int = 4
    ) -> Tuple[List[Document], List[float]]:
        """Search for similar documents.

        Args:
            query_embedding: Query embedding vector (1D array).
            k: Number of results to return.

        Returns:
            Tuple of (documents, scores) where scores are similarity scores.
        """
        if self.index is None:
            raise RuntimeError("Index has not been loaded or built yet")

        # Normalize query embedding
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding.reshape(1, -1))

        # Search
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)

        # Convert indices to documents
        documents: List[Document] = []
        similarities: List[float] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue
            doc_idx = self.index_to_docstore_id.get(int(idx))
            if doc_idx is not None and doc_idx < len(self.docstore):
                documents.append(self.docstore[doc_idx])
                similarities.append(float(score))

        return documents, similarities

    def get_document_count(self) -> int:
        """Get the number of documents in the store."""
        return len(self.docstore) if self.docstore else 0


if __name__ == "__main__":
    # Simple test
    store = FAISSVectorStore(embedding_dim=384, index_path="test_index")
    print(f"Vector store initialized at {store.index_path}")