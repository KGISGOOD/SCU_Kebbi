from __future__ import annotations
import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from embeddings import EmbeddingsProvider


class VectorStoreLoader:
    def __init__(self, embeddings: EmbeddingsProvider) -> None:
        print("Vector Store Loader initialize.")
        self._embeddings = embeddings

    def load_faiss(self, path: str) -> Optional[FAISS]:
        if os.path.exists(path):
            try:
                vs = FAISS.load_local(path, self._embeddings.get(), allow_dangerous_deserialization=True)
                print(f"Loaded vector store from {path}")
                return vs
            except Exception as e:  # noqa: BLE001
                print(f"Error loading vector store from {path}: {e}")
        else:
            print(f"Vector store not found at {path}")
        return None

    def load_all_from_dir(self, parent_dir: str) -> List[FAISS]:
        stores: List[FAISS] = []
        for root, _dirs, files in os.walk(parent_dir):
            if "index.faiss" in files and "index.pkl" in files:
                vs = self.load_faiss(root)
                if vs is not None:
                    stores.append(vs)
        return stores