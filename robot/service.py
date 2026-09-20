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

    def handle(self, message: str, history: List[Tuple[str, str]]) -> str:
        try:
            answer, _ = self._orch.ask(message, [])
            return answer
        except Exception as e:
            return f"發生錯誤: {e}"