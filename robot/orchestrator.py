from __future__ import annotations
import time
from typing import List, Tuple
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
        # Debug: show number of vectorstores if MultiStoreRetriever
        if hasattr(retriever, "vectorstores"):
            try:
                print(f"[DEBUG] Loaded {len(retriever.vectorstores)} vectorstores")
            except Exception:
                pass

    def retrieve_only(self, query: str) -> Tuple[str, List[Tuple[str, str]], float]:
        t1 = time.time()
        docs = self._retriever.invoke(query)
        t2 = time.time()
        elapsed = t2 - t1
        print(f"[DEBUG retrieve_only] query={query}, num_docs={len(docs)}")
        uniq: set[Tuple[str, str]] = set()
        ctx_lines: List[str] = []
        for d in docs:
            page = d.page_content.strip()
            ep = d.metadata.get("episode_name", "Unknown Episode")
            pod = d.metadata.get("Podcast_name", "Unknown Podcast")
            print(f"[DEBUG retrieve_only] doc metadata: {d.metadata}")
            ctx_lines.append(f"內容：{page}\n來源：{ep}, {pod}")
            uniq.add((ep, pod))
        src = "\n可參考下方節目集數：\n" + "".join([f"Result {i}: {e}, {p}\n" for i, (e, p) in enumerate(uniq, 1)])
        out = "\n--- 向量資料庫檢索結果 ---\n" + "\n\n".join(ctx_lines[:5]) + "\n\n" + src
        print(f"[DEBUG retrieve_only] uniq={uniq}")
        # Debug: context size and per-doc size
        print(f"[DEBUG] context chars = {len(out)}")
        for i, d in enumerate(docs[:5]):
            print(f"[DEBUG] doc {i+1} chars = {len(d.page_content)}")
        return out, list(uniq), elapsed

    def ask(self, question: str, history: List[Tuple[str, str]] = None) -> Tuple[str, float]:
        t_total_start = time.time()
        # RAG time
        t_rag_start = time.time()
        ctx, _, _ = self.retrieve_only(question)
        t_rag_end = time.time()
        rag_time = t_rag_end - t_rag_start

        prompt = self._prompt_factory.context_prompt().format(
            context=ctx,
            question=question
        )
        prompt_chars = len(prompt)

        t_llm_start = time.time()
        answer = self._llm._call(prompt)
        t_llm_end = time.time()
        llm_time = t_llm_end - t_llm_start

        t_total_end = time.time()
        total_time = t_total_end - t_total_start

        print(f"[PERF] RAG time: {rag_time:.3f} s")
        print(f"[PERF] Prompt chars: {prompt_chars}")
        print(f"[PERF] LLM time: {llm_time:.3f} s")
        print(f"[PERF] Total time: {total_time:.3f} s")

        return answer, 0.0