"""A0-aligned defaults without importing its mkdir-on-import Settings instance."""
from dataclasses import dataclass
import math

from experiments.a3_faq_sft.common import BASE, DataError

INDEX = "experiments/vector_stores/type1_type2_bge_m3_v1"
ADAPTER = "experiments/models/a3_smoke_v1/final_adapter"
EMBEDDING_MODEL = "BAAI/bge-m3"
TOP_K = 5
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 512
    max_input_tokens: int = 2048
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    seed: int = 42
    eos_policy: str = "model"

    def validate(self):
        if self.max_new_tokens < 1 or self.max_input_tokens < 1 or not 0 <= self.seed < 2**32:
            raise DataError("positive token limits and seed in [0,2**32) required")
        if (not all(math.isfinite(v) for v in (self.temperature, self.top_p, self.repetition_penalty))
                or self.temperature <= 0 or not 0 < self.top_p <= 1 or self.repetition_penalty <= 0):
            raise DataError("invalid temperature/top-p/repetition-penalty")
        if self.eos_policy not in ("model", "tokenizer"):
            raise DataError("eos_policy must be model or tokenizer")


DEFAULT_BASE = BASE
