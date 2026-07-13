from __future__ import annotations
import heapq
from typing import List, Sequence, Tuple
from pydantic import Field
from langchain.schema import BaseRetriever, Document
from langchain_community.vectorstores import FAISS

class MultiStoreRetriever(BaseRetriever):
    vectorstores: List[FAISS] = Field(default_factory=list)
    top_k: int = 5
    fetch_k: int = 100

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, vectorstores: Sequence[FAISS], top_k: int, fetch_k: int, **data):
        super().__init__(vectorstores=list(vectorstores), top_k=top_k, fetch_k=fetch_k, **data)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.retrieve(query)

    def retrieve(self, query: str) -> List[Document]:
        all_results: List[Tuple[Document, float]] = []
        for vs in self.vectorstores:
            _ = vs.max_marginal_relevance_search(query, k=self.fetch_k, fetch_k=self.fetch_k)
            scored = vs.similarity_search_with_score(query, k=self.fetch_k)
            all_results.extend(scored)
        best = heapq.nsmallest(self.top_k, all_results, key=lambda x: x[1])
        return [doc for doc, _ in best]