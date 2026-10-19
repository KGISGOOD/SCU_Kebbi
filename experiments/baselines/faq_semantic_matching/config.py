"""Configuration for FAQ semantic matching.

Centralizes all hyperparameters, paths, and settings for the FAQ matching baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class FAQSettings:
    """Configuration settings for FAQ semantic matching."""

    # Paths (relative to SCU_Kebbi root)
    datasets_dir: str = field(
        default_factory=lambda: str(_BASE_DIR / "experiments" / "datasets")
    )
    faq_dataset_path: str = field(
        default_factory=lambda: str(_BASE_DIR / "experiments" / "datasets" / "faq" / "faq_canonical_v1.jsonl")
    )
    faq_vector_store_dir: str = field(
        default_factory=lambda: str(_BASE_DIR / "experiments" / "vector_stores" / "faq_bge_m3_v1")
    )
    results_dir: str = field(
        default_factory=lambda: str(_BASE_DIR / "experiments" / "results" / "faq_semantic_matching")
    )
    type2_dir: str = field(
        default_factory=lambda: str(_BASE_DIR / "畢業專題_爬蟲" / "type2")
    )

    # Embedding model (reuse BGE-M3)
    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = 32
    embedding_normalize: bool = True  # L2 normalization for cosine similarity via Inner Product

    # Vector store
    use_hnsw: bool = False  # Use IndexFlatIP for reproducibility
    hnsw_M: int = 32
    hnsw_efConstruction: int = 200
    # Similarity search parameters
    top_k: int = 1  # We only need top-1 for matching, but compute top-2 for analysis
    similarity_threshold: Optional[float] = None  # To be set after calibration; None means no threshold

    # Text splitting (for metadata compatibility)
    chunk_size: int = 512
    chunk_overlap: int = 128
    # Optional fingerprint for data versioning
    data_fingerprint: Optional[str] = None

    # Misc
    verbose: bool = True

    def __post_init__(self):
        """Ensure directories exist."""
        for dir_path in [
            self.faq_vector_store_dir,
            self.results_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = FAQSettings()