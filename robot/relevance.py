from __future__ import annotations
from typing import List, Tuple

class RelevancePolicy:
    def is_related(self, new_question: str, history: List[Tuple[str, str]]) -> bool:
        if not history:
            return False
        last_question = history[-1][0]
        k1 = set(last_question.lower().split())
        k2 = set(new_question.lower().split())
        overlap = len(k1.intersection(k2))
        threshold = max(1, int(min(len(k1), len(k2)) * 0.3))
        return overlap >= threshold