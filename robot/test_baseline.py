#!/usr/bin/env python
# -*- coding: utf-8 """
"""
Baseline latency benchmark for SCU_Kebbi project.
Measures pure RAG → Llama 3.2 latency without any optimisation.
Does NOT modify any existing baseline files.
"""

import os
import time
from config import AppSettings
from embeddings import HFEmbeddingsProvider
from vectorstores import VectorStoreLoader
from retriever import MultiStoreRetriever
from llm.ollama import ChatOllamaLLM
from prompts import PromptFactory
from orchestrator import QAOrchestrator

def main():
    print("=== Baseline Latency Benchmark ===")
    print("Measuring pure RAG → Llama 3.2 latency (no cache, attention, LoRA, etc.)\n")

    # ----- Initialization (exactly as in baseline_gradio.py) -----
    settings = AppSettings()
    emb = HFEmbeddingsProvider(settings.model_name, settings.use_cpu)
    loader = VectorStoreLoader(emb)
    stores = loader.load_all_from_dir(settings.parent_vector_dir)
    retriever = MultiStoreRetriever(stores, top_k=settings.top_k, fetch_k=settings.fetch_k)
    llm = ChatOllamaLLM(
        model_name=settings.ollama_model,
        url=settings.ollama_url,
        stream=settings.ollama_stream,
        timeout_sec=settings.ollama_timeout_sec,
    )
    prompts = PromptFactory()
    orch = QAOrchestrator(retriever, llm, prompts)

    # ----- Warm-up (not counted) -----
    print("[Warm-up] Running a dummy query to initialise models / FAISS / Ollama...")
    _ , _ = orch.ask("測試", [])
    print("[Warm-up] Completed.\n")

    # ----- Test queries -----
    queries = [
        "巨資學院有哪些特色？",
        "巨資學院有哪些核心課程？",
        "如何申請巨資學院的雙聯學制？",
        "巨資學院有哪些實習機會？",
        "巨資學院的學生社團有哪些？",
    ]

    latencies = []
    answers = []

    for idx, q in enumerate(queries, start=1):
        t0 = time.perf_counter()
        answer, _ = orch.ask(q, [])
        t1 = time.perf_counter()
        latency = t1 - t0
        latencies.append(latency)
        answers.append(answer.strip())
        print(f"[{idx}/{len(queries)}] Q: {q}")
        print(f"      Latency: {latency:.4f} s")
        print(f"      Answer: {answer[:100]}{'...' if len(answer)>100 else ''}")
        print()

    # ----- Statistics -----
    def avg(lst): return sum(lst)/len(lst) if lst else 0.0
    def med(lst):
        s = sorted(lst)
        n = len(s)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return s[n//2]
        else:
            return (s[n//2-1] + s[n//2]) / 2.0

    avg_lat = avg(latencies)
    med_lat = med(latencies)

    print("=== Summary ===")
    print(f"Query count        : {len(queries)}")
    print(f"Average latency    : {avg_lat:.4f} s")
    print(f"Median latency     : {med_lat:.4f} s")
    print("(Warm-up query excluded from statistics)")

if __name__ == "__main__":
    main()