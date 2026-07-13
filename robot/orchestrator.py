from __future__ import annotations
import time
from typing import List, Tuple, Optional
from langchain.schema import BaseRetriever
from llm.ollama import ChatOllamaLLM
from prompts import PromptFactory  # kept for compatibility, not used


class QAOrchestrator:
    def __init__(
        self,
        retriever: BaseRetriever,
        llm_facade: ChatOllamaLLM,
        prompt_factory: PromptFactory,
    ) -> None:
        self._retriever = retriever
        self._llm = llm_facade
        self._prompt_factory = prompt_factory  # kept for compatibility, not used

    def retrieve_only(self, query: str) -> Tuple[str, List[Tuple[str, str]], float]:
        t1 = time.time()
        docs = self._retriever.invoke(query)
        t2 = time.time()
        elapsed = t2 - t1
        uniq: set[Tuple[str, str]] = set()
        ctx_lines: List[str] = []
        for d in docs:
            page = d.page_content.strip()
            ep = d.metadata.get("episode_name", "Unknown Episode")
            pod = d.metadata.get("Podcast_name", "Unknown Podcast")
            ctx_lines.append(f"內容：{page}\n來源：{ep}, {pod}")
            uniq.add((ep, pod))
        src = "\n可參考下方節目集數：\n" + "".join([f"Result {i}: {e}, {p}\n" for i, (e, p) in enumerate(uniq, 1)])
        out = "\n--- 向量資料庫檢索結果 ---\n" + "\n\n".join(ctx_lines[:5]) + "\n\n" + src
        return out, list(uniq), elapsed

    def ask(
        self,
        question: str,
        context: str,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[str, float]:
        """
        Combine the raw question with the retrieved context (no extra prompt template)
        and send it to the LLM.
        """
        # Simple concatenation: question first, then retrieved information
        combined = f"問題：{question}\n\n檢索資訊：{context}"
        t3 = time.time()
        # The LLM facade's _call expects a plain string prompt
        ans = self._llm._call(combined)
        t4 = time.time()
        return ans.strip(), (t4 - t3)