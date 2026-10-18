"""Embedding provider for BGE-M3 model.

Loads the BAAI/bge-m3 model and generates normalized embeddings.
"""

from __future__ import annotations

import os
from typing import List, Union, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


class BGEM3Embedder:
    """Embedding wrapper for BGE-M3 model with L2 normalization."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ):
        """Initialize the embedder.

        Args:
            model_name: Hugging Face model identifier.
            device: Device to use ('cuda', 'cpu', or None for auto).
            batch_size: Batch size for encoding.
            normalize_embeddings: Whether to L2-normalize embeddings.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model.max_seq_length = 512  # BGE-M3 supports 512
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}") from e

        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(
        self, sentences: Union[str, List[str]], batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Encode sentences to embeddings.

        Args:
            sentences: Single string or list of strings.
            batch_size: Override default batch size.

        Returns:
            Numpy array of embeddings (normalized if enabled).
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        if batch_size is None:
            batch_size = self.batch_size

        # Encode with sentence-transformers
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (LangChain compatible)."""
        embeddings = self.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query (LangChain compatible)."""
        embedding = self.encode([text])
        return embedding[0].tolist()


def get_embedder(
    model_name: str = "BAAI/bge-m3",
    device: Optional[str] = None,
    batch_size: int = 32,
) -> BGEM3Embedder:
    """Factory function to get an embedder instance."""
    return BGEM3Embedder(model_name=model_name, device=device, batch_size=batch_size)


if __name__ == "__main__":
    # Simple test
    embedder = BGEM3Embedder()
    test_texts = ["Hello world", "這是測試"]
    embeddings = embedder.encode(test_texts)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"First embedding (first 5 dims): {embeddings[0][:5]}")
    # Check normalization
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Embedding norms: {norms}")