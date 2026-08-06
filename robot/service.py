from __future__ import annotations
from typing import List, Tuple
from opencc import OpenCC
from orchestrator import QAOrchestrator
from relevance import RelevancePolicy


class ChatService:
    def __init__(self, orchestrator: QAOrchestrator, relevance: RelevancePolicy, retrieve_only: bool) -> None:
        self._orch = orchestrator
        self._rel = relevance
        self._retrieve_only = retrieve_only
        self._cc = OpenCC("s2t")

    def handle(self, message: str, history: List[Tuple[str, str]]):
        try:
            if not self._rel.is_related(message, history):
                self._orch.clear_memory()

            ctx, uniq_sources, vec_time = self._orch.retrieve_only(message)
            if self._retrieve_only:
                return self._cc.convert(ctx + f"\n---\n向量���檢索時間: {vec_time:.2f} 秒")

            ans, llm_time = self._orch.ask(message, history if history else [])
            src = "\n可參考下方節目集數：\n" + "".join([f"Result {i}: {e}, {p}\n" for i, (e, p) in enumerate(uniq_sources, 1)])
            full = ans + "\n\n" + src
            full = self._cc.convert(full)
            timing = f"\n---\n向量���檢索時間: {vec_time:.2f} 秒\nLLM 生成時間: {llm_time:.2f} 秒"
            return full + timing
        except Exception as e:  # noqa: BLE001
            return f"發生���錯���誤: {e}\n很抱歉，我無法處理您的問題。���請再試一次或���換個問題。"