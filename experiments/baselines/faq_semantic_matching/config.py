"""V6 Matcher top-two search defaults; FAQFallback owns acceptance policy."""
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class FAQSettings:
    top_k: int = 1
    similarity_threshold: Optional[float] = None

settings = FAQSettings()
