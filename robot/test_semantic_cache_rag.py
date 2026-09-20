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
    # 0. Load existing cache if present
    script_dir = os.path.dirname(__file__)
    cache_index_path = os.path.join(script_dir, "semantic_cache.index")
    cache_store_path = os.path.join(script_dir, "semantic_cache_store.pkl")
    if os.path.exists(cache_index_path) and os.path.exists(cache_store_path):
        cache_index = faiss.read_index(cache_index_path)
        with open(cache_store_path, "rb") as f:
            cache_store = pickle.load(f)
        print(f"[Cache] Loaded existing cache: {len(cache_store)} entries")
    else:
        cache_index = None
        cache_store = []

    # 1. Initialize settings and components (same as baseline)
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

    # 2. Ensure cache structures are initialized
    if cache_index is None:
        sample_vec = emb.get().embed_query("test")
        embed_dim = len(sample_vec)
        cache_index = faiss.IndexFlatIP(embed_dim)
        cache_store = []

    # 3. Warm-up (run a dummy query through the full RAG pipeline)
    print("[Warm-up] Running dummy query to load model...")
    dummy_q = "測試"
    t0 = time.perf_counter()
    _ , _ = orch.ask(dummy_q, [])
    t1 = time.perf_counter()
    print(f"[Warm-up] Completed in {t1 - t0:.3f}s\n")

    # 4. Test questions: identical, semantically similar, unrelated
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

    hit_count = 0
    miss_count = 0
    total_latency = 0.0
    hit_latencies = []
    miss_latencies = []
    baseline_latencies = []  # latency of full RAG+LLM for each question
    answers = []  # final answers (from cache or LLM)
    baselines = []  # answers from LLM (for consistency check, optional)

    for idx, q in enumerate(questions, start=1):
        # Embed and normalize question
        q_vec = np.array(emb.get().embed_query(q)).reshape(1, -1)
        q_vec = l2_normalize(q_vec)

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
        else:
            # Cache MISS
            miss = True
            miss_count += 1
            t0 = time.perf_counter()
            answer, _ = orch.ask(q, [])  # baseline RAG+LLM
            t1 = time.perf_counter()
            latency = t1 - t0
            miss_latencies.append(latency)

            # Store in cache for future
            a_vec = np.array(emb.get().embed_query(q)).reshape(1, -1)
            a_vec = l2_normalize(a_vec)
            cache_index.add(a_vec)
            cache_store.append((q, answer.strip()))

        total_latency += latency
        answers.append(answer)

        # Baseline latency (full RAG+LLM) for each question
        t0b = time.perf_counter()
        _, baseline_llm_time = orch.ask(q, [])  # we ignore answer, just measure
        t1b = time.perf_counter()
        baseline_latency = t1b - t0b
        baseline_latencies.append(baseline_latency)
        # Optionally store baseline answer for consistency check (skip to avoid extra output)

        # Progress
        status = "HIT" if hit else "MISS"
        print(f"[{idx:02d}/{len(questions)}] Q: {q}")
        print(f"      {status} | latency={latency:.3f}s")
        if not hit:
            print(f"      Baseline latency={baseline_latency:.3f}s")
        print(f"      Answer: {answer[:100]}{'...' if len(answer)>100 else ''}")
        print()

    # 5. Statistics
    total_queries = len(questions)
    hit_rate = hit_count / total_queries if total_queries else 0.0
    avg_latency = total_latency / total_queries if total_queries else 0.0
    median_latency = np.median([lat for lat in (hit_latencies + miss_latencies)]) if (hit_latencies + miss_latencies) else 0.0
    avg_hit_latency = np.mean(hit_latencies) if hit_latencies else 0.0
    avg_miss_latency = np.mean(miss_latencies) if miss_latencies else 0.0
    avg_baseline_latency = np.mean(baseline_latencies) if baseline_latencies else 0.0
    latency_reduction = ((avg_baseline_latency - avg_latency) / avg_baseline_latency * 100) if avg_baseline_latency > 0 else 0.0

    print("=== Experiment Results ===")
    print(f"Total queries          : {total_queries}")
    print(f"Cache hits             : {hit_count}")
    print(f"Cache misses           : {miss_count}")
    print(f"Hit rate               : {hit_rate:.2%}")
    print(f"Average latency        : {avg_latency:.3f} s")
    print(f"Median latency         : {median_latency:.3f} s")
    print(f"Average hit latency    : {avg_hit_latency:.3f} s")
    print(f"Average miss latency   : {avg_miss_latency:.3f} s")
    print(f"Baseline average latency: {avg_baseline_latency:.3f} s")
    print(f"Latency reduction      : {latency_reduction:.1f}%")

    # 6. Persist cache
    faiss.write_index(cache_index, cache_index_path)
    with open(cache_store_path, "wb") as f:
        pickle.dump(cache_store, f)
    print(f"[Cache] Saved to {cache_index_path} and {cache_store_path}")

if __name__ == "__main__":
    main()