"""Independent multi-turn A0 adapter and conservative reduction policy."""
from __future__ import annotations

import re
import time
from experiments.algorithms.shared.a0_adapter import sync, measured
from experiments.algorithms.semantic_cache.cache import fingerprint


DEPENDENCY_PATTERN = r"它|他們|她們|這|那|上述|前面|剛才|剛剛|之前|繼續|接著|同樣|比較|相較|如果|所以|還有|也能|也可以|分別|兩者|哪一個|呢|\b(it|they|that|those|also|same)\b"
POLICY_VERSION = "keep_last_pair_and_dependency_guard_v1"


def reduce_history(history, query, shifted):
    """Never reads gold labels or required_history_ids."""
    if not shifted:
        return list(history), "no predicted topic shift"
    if re.search(DEPENDENCY_PATTERN, query, flags=re.IGNORECASE):
        return list(history), "conservative lexical dependency guard"
    # History validation ensures complete alternating user/assistant pairs.
    if len(history) <= 2:
        return list(history), "keep most recent completed pair; no removable older pair"
    return list(history[-2:]), "predicted shift; retain latest completed pair, remove older pairs only"


def multiturn_prompt(docs, query, history):
    from experiments.baselines.rag_baseline.prompts import build_prompt, create_context_from_documents
    base = build_prompt(create_context_from_documents(docs), query)
    marker = "<|start_header_id|>user<|end_header_id|>"
    position = base.find(marker)
    if position < 0:
        raise ValueError("A0 prompt header changed; review multi-turn adapter")
    rendered = "".join(f"<|start_header_id|>{m['role']}<|end_header_id|>\n{m['content']}<|eot_id|>"
                       for m in history)
    return base[:position] + rendered + base[position:]


class ContextExperiment:
    def __init__(self, adapter, features, thresholds, config):
        self.adapter, self.features, self.thresholds, self.config = adapter, features, thresholds, config

    def run(self, case, docs, retrieval_seconds, arm, repeat):
        import torch
        sync()
        start = time.perf_counter()
        detection = None
        decision_seconds = 0.0
        if arm == "full":
            kept, reason, prediction = list(case["history"]), "full-history baseline", None
        else:
            def detect():
                data = self.features.embedding(case) if arm == "embedding" else self.features.attention(case)
                key = "TI0" if arm == "embedding" else "TI3-1"
                return data, int(data["scores"][key] < self.thresholds[key]["threshold"])
            (detection, prediction), decision_seconds = measured(detect)
            kept, reason = reduce_history(case["history"], case["query"], prediction)
        def prepare():
            tokenizer = self.adapter.llm.tokenizer
            original = multiturn_prompt(docs, case["query"], case["history"])
            prompt = multiturn_prompt(docs, case["query"], kept)
            original_tokens = len(tokenizer(original, truncation=False)["input_ids"])
            reduced_tokens = len(tokenizer(prompt, truncation=False)["input_ids"])
            limit = self.adapter.llm.model.config.max_position_embeddings
            if (max(original_tokens, reduced_tokens) > 2048
                    or max(original_tokens, reduced_tokens) + self.config["max_new_tokens"] > limit):
                raise ValueError("full multi-turn baseline exceeds A0 input/context limit; no truncation")
            return prompt, original_tokens, reduced_tokens
        (prompt, original_count, reduced_count), preparation_seconds = measured(prepare)
        torch.manual_seed(self.config["seed"] + repeat)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config["seed"] + repeat)
        answer, generation_seconds = measured(self.adapter.llm.generate, prompt)
        sync()
        execution = time.perf_counter() - start
        kept_ids = [m["turn_id"] for m in kept]
        # Gold dependency annotations are used ONLY after the online policy.
        required = case["required_history_ids"]
        return {"scenario_id": case["scenario_id"], "arm": arm, "repeat_index": repeat,
                "question": case["query"], "answer": answer, "prediction_shift": prediction,
                "topic_detection_seconds": decision_seconds, "detection": detection,
                "retrieval_seconds": retrieval_seconds, "prompt_preparation_seconds": preparation_seconds,
                "generation_seconds": generation_seconds, "mode_execution_seconds": execution,
                "end_to_end_seconds": retrieval_seconds + execution,
                "end_to_end_kind": "controlled sum of once-shared retrieval and independent arm execution",
                "original_prompt_tokens": original_count, "reduced_prompt_tokens": reduced_count,
                "prompt_token_reduction": original_count - reduced_count,
                "kept_history_ids": kept_ids, "removed_history_ids": [m["turn_id"] for m in case["history"] if m["turn_id"] not in kept_ids],
                "policy_reason": reason, "prompt": prompt, "prompt_identity": fingerprint(prompt),
                "annotated_required_context_retained": set(required).issubset(kept_ids),
                "annotation_review_status": case["review_status"],
                "quality_review": {"answer_correct": None, "context_sufficient": None,
                                   "equivalent_quality_to_full": None, "reviewer": None, "notes": None},
                "quality_verified": False, "context_safety_guaranteed": False}
