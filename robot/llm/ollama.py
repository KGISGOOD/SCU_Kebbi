from __future__ import annotations
import json
from typing import Any, List, Optional
import requests
from langchain.llms.base import BaseLLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLMResult

class ChatOllamaLLM(BaseLLM):
    model_name: str
    url: str
    do_stream: bool = False
    timeout_sec: int = 120

    def __init__(self, model_name: str, url: str, stream: bool, timeout_sec: int, **data):
        super().__init__(model_name=model_name, url=url, do_stream=stream, timeout_sec=timeout_sec, **data)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        payload = {"model": self.model_name, "prompt": prompt, "stream": self.do_stream}
        try:
            resp = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout_sec,
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                stream=self.do_stream,
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"[Ollama] connection error: {e}")

        if resp.status_code != 200:
            raise RuntimeError(f"[Ollama] HTTP {resp.status_code}: {resp.text[:500]}")

        if not self.do_stream:
            try:
                obj = resp.json()
                return obj.get("response", "")
            except Exception as e:
                raise RuntimeError(f"[Ollama] invalid JSON: {e}. body={resp.text[:500]}")

        full = ""
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    full += obj.get("response", "")
                except json.JSONDecodeError:
                    continue
        finally:
            resp.close()
        return full

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        gens = []
        for p in prompts:
            gens.append([{"text": self._call(p, stop=stop, run_manager=run_manager, **kwargs)}])
        return LLMResult(generations=gens)

    @property
    def _llm_type(self) -> str:
        return "chat_ollama"


