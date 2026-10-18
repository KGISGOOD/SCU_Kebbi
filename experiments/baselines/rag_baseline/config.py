"""Configuration for the RAG baseline.

Centralizes all hyperparameters, paths, and settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

@dataclass
class Settings:
    """Configuration settings for the RAG baseline."""

    # Paths (relative to SCU_Kebbi root)
    type1_dir: str = field(
        default_factory=lambda: str(_BASE_DIR / "畢業專題_爬蟲" / "type1")
    )
    type2_dir: str = field(
        default_factory=lambda: str(_BASE_DIR / "畢業專題_爬蟲" / "type2")
    )
    vector_store_dir: str = field(
        default_factory=lambda: str(_BASE_DIR / "experiments" / "vector_stores" / "type1_type2_bge_m3_v1")
    )
    results_dir: str = field(
        default_factory=lambda: str(_BASE_DIR / "experiments" / "results" / "rag_baseline")
    )
    datasets_dir: str = field(
        default_factory=lambda: str(_BASE_DIR / "experiments" / "datasets")
    )

    # Data loading
    chunk_size: int = 512
    chunk_overlap: int = 128
    remove_duplicates: bool = True

    # Embedding model
    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = 32
    embedding_normalize: bool = True

    # Vector store
    use_hnsw: bool = False  # Use IndexFlatIP for reproducibility
    hnsw_M: int = 32
    hnsw_efConstruction: int = 200
    # Similarity search parameters
    top_k: int = 5
    score_threshold: Optional[float] = None  # None means no threshold

    # LLM settings
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

    # Prompt settings
    prompt_template: str = """基於以下提供的資訊回答問題。
如果提供的資訊中沒有答案，請說「根據現有資料，無法回答此問題」。

{context}

問題：{question}

回答："""
    # Alternative simple template
    # prompt_template: str = "{context}\n\n{question}"

    # Generation
    seed: Optional[int] = 42

    # Misc
    verbose: bool = True
    show_sources: bool = True

    def __post_init__(self):
        """Ensure directories exist."""
        for dir_path in [
            self.vector_store_dir,
            self.results_dir,
            self.datasets_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()