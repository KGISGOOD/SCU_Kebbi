#!/usr/bin/env python
# -*- coding: utf-8 """
"""
Semantic Cache + RAG experiment for SCU_Kebbi project.
Uses existing components (HFEmbeddingsProvider, FAISS, MultiStoreRetriever, QAOrchestrator)
without modifying baseline core logic.
"""
import os
import time
import pickle
import numpy as np
import faiss
from service import ChatService
from config import AppSettings
from embeddings import HFEmbeddingsProvider
from vectorstores import VectorStoreLoader
from retriever import MultiStoreRetriever
from llm.ollama import ChatOllamaLLM
from prompts import PromptFactory
from orchestrator import QAOrchestrator
from relevance import RelevancePolicy

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2 normalize rows of a 2D array (N,D)."""
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / (norm + 1e-10)

def main():
    # Shared questions
    questions = [
        # identical
        "巨資學院有哪些特色？",
        "巨資學院有哪些特色？",
        # semantically similar (paraphrase)
        "請說明巨資學院的主要特色是什麼？",
        "巨資學院的特色有哪些？",
        # unrelated
        "今天台北的天氣如何？",
        "石頭切割機的使用方法是？",
        # another similar pair
        "如何申請巨資學院的雙聯學制？",
        "申請巨資學院雙聯學制需要準備哪些文件？",
        # unrelated
        "Python 列表排序的方法有哪些？",
        "學校的圖書館開放時間是幾點？",
    ]

    # Shared components (Baseline)
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
    relevance = RelevancePolicy()  # not used in this experiment but kept for compatibility
    service = ChatService(orch, relevance, retrieve_only=False)  # not used directly

    # ========== Phase 1: Baseline (no cache) ==========
    print("\n=== Phase 1: Baseline (RAG -> Llama 3.2) ===")
    # Warm-up
    dummy_q = "測試"
    t0 = time.perf_counter()
    _ , _ = orch.ask(dummy_q, [])
    t1 = time.perf_counter()
    print(f"[Warm-up] Completed in {t1 - t0:.3f}s\n")

    baseline_latencies = []
    for idx, q in enumerate(questions, start=1):
        t0 = time.perf_counter()
        answer, _ = orch.ask(q, [])  # baseline RAG+LLM
        t1 = time.perf_counter()
        latency = t1 - t0
        baseline_latencies.append(latency)
        print(f"[{idx:02d}/{len(questions)}] Q: {q}")
        print(f"      Baseline latency={latency:.3f}s")
        print(f"      Answer: {answer[:100]}{'...' if len(answer)>100 else ''}")
        print()
    # Baseline stats
    baseline_avg = np.mean(baseline_latencies) if baseline_latencies else 0.0
    baseline_median = np.median(baseline_latencies) if baseline_latencies else 0.0

    # ========== Phase 2: Semantic Cache + RAG ==========
    print("\n=== Phase 2: Semantic Cache + RAG (fresh cache) ===")
    # Fresh cache (in-memory)
    cache_index = None
    cache_store = []

    # Warm-up (model already loaded, but run a dummy query through orch.ask to ensure readiness)
    t0 = time.perf_counter()
    _ , _ = orch.ask(dummy_q, [])
    t1 = time.perf_counter()
    print(f"[Warm-up] Completed in {t1 - t0:.3f}s\n")

    hit_count = 0
    miss_count = 0
    hit_latencies = []
    miss_latencies = []
    cache_latencies = []  # overall latency per question (hit=0, miss=RAG+LLM)
    for idx, q in enumerate(questions, start=1):
        # Embed and normalize question
        q_vec = np.array(emb.get().embed_query(q)).reshape(1, -1)
        q_vec = l2_normalize(q_vec)

        # Initialize cache index on first use
        if cache_index is None:
            embed_dim = len(q_vec[0])
            cache_index = faiss.IndexFlatIP(embed_dim)
            cache_store = []

        # Search cache
        D, I = cache_index.search(q_vec, 1)
        similarity = float(D[0][0])
        cached_idx = int(I[0][0])

        hit = False
        if similarity >= 0.75 and cached_idx < len(cache_store):
            # Cache HIT
            hit = True
            hit_count += 1
            answer = cache_store[cached_idx][1]  # (question, answer)
            latency = 0.0  # embedding+search negligible
            hit_latencies.append(latency)
            cache_latencies.append(latency)
        else:
            # Cache MISS
            miss = True
            miss_count += 1
            t0 = time.perf_counter()
            answer, _ = orch.ask(q, [])  # baseline RAG+LLM
            t1 = time.perf_counter()
            latency = t1 - t0
            miss_latencies.append(latency)
            cache_latencies.append(latency)

            # Store in cache for future
            a_vec = np.array(emb.get().embed_query(q)).reshape(1, -1)
            a_vec = l2_normalize(a_vec)
            cache_index.add(a_vec)
            cache_store.append((q, answer.strip()))

        status = "HIT" if hit else "MISS"
        print(f"[{idx:02d}/{len(questions)}] Q: {q}")
        print(f"      {status} | latency={latency:.3f}s")
        if not hit:
            print(f"      Baseline latency={latency:.3f}s")
        print(f"      Answer: {answer[:100]}{'...' if len(answer)>100 else ''}")
        print()

    # Cache stats
    total_queries = len(questions)
    hit_rate = hit_count / total_queries if total_queries else 0.0
    cache_avg = np.mean(cache_latencies) if cache_latencies else 0.0
    cache_median = np.median(cache_latencies) if cache_latencies else 0.0
    avg_hit_latency = np.mean(hit_latencies) if hit_latencies else 0.0
    avg_miss_latency = np.mean(miss_latencies) if miss_latencies else 0.0

    # ========== Summary ==========
    print("=== Experiment Results ===")
    print("\n--- Baseline ---")
    print(f"Total queries          : {total_queries}")
    print(f"Average latency        : {baseline_avg:.3f} s")
    print(f"Median latency         : {baseline_median:.3f} s")
    print("\n--- Semantic Cache + RAG ---")
    print(f"Total queries          : {total_queries}")
    print(f"Cache hits             : {hit_count}")
    print(f"Cache misses           : {miss_count}")
    print(f"Hit rate               : {hit_rate:.2%}")
    print(f"Average latency        : {cache_avg:.3f} s")
    print(f"Median latency         : {cache_median:.3f} s")
    print(f"Average hit latency    : {avg_hit_latency:.3f} s")
    print(f"Average miss latency   : {avg_miss_latency:.3f} s")
    print("\n--- Comparison ---")
    print(f"Baseline average latency: {baseline_avg:.3f} s")
    print(f"Semantic Cache + RAG average latency: {cache_avg:.3f} s")
    latency_reduction = ((baseline_avg - cache_avg) / baseline_avg * 100) if baseline_avg > 0 else 0.0
    print(f"Latency reduction      : {latency_reduction:.1f}%")

    # Persist cache from Phase 2
    script_dir = os.path.dirname(__file__)
    cache_index_path = os.path.join(script_dir, "semantic_cache.index")
    cache_store_path = os.path.join(script_dir, "semantic_cache_store.pkl")
    faiss.write_index(cache_index, cache_index_path)
    with open(cache_store_path, "wb") as f:
        pickle.dump(cache_store, f)
    print(f"[Cache] Saved to {cache_index_path} and {cache_store_path}")

if __name__ == "__main__":
    main()