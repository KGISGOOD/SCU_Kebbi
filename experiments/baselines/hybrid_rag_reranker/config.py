"""Configuration for Hybrid RAG + Reranker (A4).

Centralizes all hyperparameters, paths, and settings for the A4 baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class Settings:
    """Configuration settings for the Hybrid RAG + Reranker baseline."""

    # Paths (relative to SCU_Kebbi root)
    type1_dir: str = field(
        default_factory=lambda: str(_BASE_DIR / "畢業專題_爬蟲" / "type1")
    )
    type2_dir: str = field(
        default_factory=lambda: str(_BASE_DIR / "畢業專題_爬蟲" / "type2")
    )
    # BM25 index directory (separate from dense vector store)
    bm25_index_dir: str = field(
        default_factory=lambda: str(_BASE_DIR / "experiments" / "vector_stores" / "hybrid_bm25_v1")
    )
    # Dense vector store directory (reuse same as A0 for fair comparison)
    dense_vector_store_dir: str = field(
        default_factory=lambda: str(_BASE_DIR / "experiments" / "vector_stores" / "type1_type2_bge_m3_v1")
    )
    results_dir: str = field(
        default_factory=lambda: str(_BASE_DIR / "experiments" / "results" / "hybrid_rag_reranker")
    )
    datasets_dir: str = field(
        default_factory=lambda: str(_BASE_DIR / "experiments" / "datasets")
    )

    # Data loading (same as A0 for fair comparison)
    chunk_size: int = 512
    chunk_overlap: int = 128
    remove_duplicates: bool = True

    # Embedding model (same as A0 - BGE-M3)
    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = 32
    embedding_normalize: bool = True

    # Vector store (same as A0 - FAISS IndexFlatIP)
    use_hnsw: bool = False  # Use IndexFlatIP for reproducibility
    hnsw_M: int = 32
    hnsw_efConstruction: int = 200
    # Similarity search parameters
    top_k: int = 10  # Initial retrieval top-k for both BM25 and Dense
    score_threshold: Optional[float] = None  # None means no threshold

    # RRF Fusion parameter
    rrf_k: int = 60  # Default RRF parameter

    # Reranker model (BGE Reranker)
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_batch_size: int = 32
    reranker_max_length: int = 512
    reranker_top_k: int = 5  # Final top-k after reranking

    # LLM settings (same as A0 for fair comparison)
    llm_model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    llm_device: Optional[str] = None  # Auto-detect
    llm_load_in_8bit: bool = False
    llm_load_in_4bit: bool = False
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.7
    llm_top_p: float = 0.9
    llm_do_sample: bool = True
    llm_repetition_penalty: float = 1.0
    llm_trust_remote_code: bool = True

    # Prompt settings (same as A0 for fair comparison)
    prompt_template: str = """基於以下提供的資訊回答問題。
如果提供的資訊中沒有答案，請說「根據現有資料，無法回答此問題」。

{context}

問題：{question}

回答："""

    # Generation
    seed: Optional[int] = 42

    # Misc
    verbose: bool = True
    show_sources: bool = True

    def __post_init__(self):
        """Ensure directories exist."""
        for dir_path in [
            self.bm25_index_dir,
            self.dense_vector_store_dir,
            self.results_dir,
            self.datasets_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()