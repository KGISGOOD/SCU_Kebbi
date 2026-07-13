from __future__ import annotations
from typing import Any, Protocol
import torch
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


class EmbeddingsProvider(Protocol):
    def get(self) -> Any:  # returns an embeddings object compatible with FAISS
        ...

class HFEmbeddingsProvider:
    def __init__(self, model_name: str, use_cpu: bool) -> None:
        self._model_name = model_name
        self._use_cpu = use_cpu
        self._emb = None  # cache

    def get(self):
        if self._emb is None:
            device = "cpu" if self._use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
            self._emb = HuggingFaceEmbeddings(model_name=self._model_name, model_kwargs={"device": device})
        return self._emb
