from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppSettings:
    parent_vector_dir: str = os.getenv("PODCAST_VECTOR_DIR", "./vector_store")
    model_name: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    use_cpu: bool = os.getenv("USE_CPU", "false").lower() == "true"
    top_k: int = int(os.getenv("TOP_K", "5"))
    fetch_k: int = int(os.getenv("FETCH_K", "100"))
    ollama_model: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:14b")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
    ollama_stream: bool = os.getenv("OLLAMA_STREAM", "false").lower() == "true"
    ollama_timeout_sec: int = int(os.getenv("OLLAMA_TIMEOUT", "900"))

