from __future__ import annotations
import time
from typing import List, Tuple
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseRetriever
from llm.ollama import ChatOllamaLLM
from prompts import PromptFactory


class QAOrchestrator:
    def __init__(
        self,
        retriever: BaseRetriever,
        llm_facade: ChatOllamaLLM,
        prompt_factory: PromptFactory,
    ) -> None:
        self._retriever = retriever
        self._llm = llm_facade
        self._prompt_factory = prompt_factory
        self._memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # Debug: show number of vectorstores if MultiStoreRetriever
        if hasattr(retriever, "vectorstores"):
            try:
                print(f"[DEBUG] Loaded {len(retriever.vectorstores)} vectorstores")
            except Exception:
                pass

        self._qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self._llm,  # ChatOllamaLLM implements generate(), compatible
            retriever=self._retriever,
            memory=self._memory,
            combine_docs_chain_kwargs={
                "prompt": self._prompt_factory.context_prompt(),
                "document_variable_name": "context",
                "document_prompt": self._prompt_factory.document_prompt(),
            },
        )

    def clear_memory(self) -> None:
        self._memory.clear()

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
        out = "\n--- 向量資料庫�檢索結果 ---\n" + "\n\n".join(ctx_lines[:5]) + "\n\n" + src
        return out, list(uniq), elapsed

    def ask(self, question: str, history: List[Tuple[str, str]]) -> Tuple[str, float]:
        t3 = time.time()
        resp = self._qa_chain.invoke({"question": question, "chat_history": history})
        t4 = time.time()
        return resp.get("answer", ""), (t4 - t3)