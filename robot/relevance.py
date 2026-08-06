from __future__ import annotations
from typing import List, Tuple

class RelevancePolicy:
    def is_related(self, new_question: str, history: List[Tuple[str, str]]) -> bool:
        return True