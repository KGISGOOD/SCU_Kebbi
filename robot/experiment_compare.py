#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment: Baseline vs Simple Semantic Cache
Compares average latency of vanilla RAG+LLM vs a question-level semantic cache.
"""

import os
import time
import numpy as np
import faiss
from typing import List, Tuple

# Reuse existing project modules
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
    """L2-normalize rows of a 2D array (or a 1D vector)."""
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / (norm + 1e-10)

def load_questions(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    # ---------- 0. Load settings and initialise shared components ----------
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
    relevance = RelevancePolicy()
    service = ChatService(orch, relevance, retrieve_only=False)

    # ---------- 1. Prepare questions ----------
    questions_path = os.path.join(os.path.dirname(__file__), "experiment_questions.txt")
    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    questions = load_questions(questions_path)
    if len(questions) == 0:
        raise ValueError("No questions loaded.")
    print(f"Loaded {len(questions)} questions from {questions_path}")

    # ---------- 2. Baseline experiment ----------
    print("\n=== Running Baseline (no cache) ===")
    baseline_latencies: List[float] = []
    baseline_answers: List[str] = []
    for idx, q in enumerate(questions, start=1):
        t0 = time.perf_counter()
        ans = service.handle(q, [])
        t1 = time.perf_counter()
        latency = t1 - t0
        baseline_latencies.append(latency)
        baseline_answers.append(ans)
        print(f"[{idx:03d}/{len(questions)}] Q: {q}")
        print(f"      Latency: {latency:.3f}s")
        print(f"      Answer: {ans[:80]}{'...' if len(ans)>80 else ''}")
        print()

    # ---------- 3. Simple Semantic Cache experiment ----------
    print("\n=== Running Simple Semantic Cache ===")
    cache_index = None  # type: faiss.Index | None
    cache_store: List[Tuple[str, str]] = []  # each entry: (prompt, answer)
    cache_latencies: List[float] = []
    cache_hits: List[bool] = []  # True if hit, False if miss
    for idx, q in enumerate(questions, start=1):
        t0 = time.perf_counter()
        # --- Embedding and cache search ---
        q_vec = np.array(emb.get().embed_query(q)).reshape(1, -1)  # (1, D)
        q_vec = l2_normalize(q_vec)
        hit = False
        if cache_index is not None and cache_index.ntotal > 0:
            D, I = cache_index.search(q_vec, 1)  # D: similarity, I: index
            sim = float(D[0][0])
            idx_cache = int(I[0][0])
            if sim >= 0.75 and idx_cache < len(cache_store):
                # Cache hit
                hit = True
                answer = cache_store[idx_cache][1]
                t1 = time.perf_counter()
                latency = t1 - t0
                cache_latencies.append(latency)
                cache_hits.append(True)
                print(f"[{idx:03d}/{len(questions)}] Q: {q}")
                print(f"      HIT  | latency={latency:.3f}s (sim={sim:.3f})")
                print(f"      Answer: {answer[:80]}{'...' if len(answer)>80 else ''}")
                print()
                continue
        # Cache miss (includes case where index not ready)
        # Obtain answer via baseline path (service.handle does retrieve+LLM)
        ans = service.handle(q, [])
        t1 = time.perf_counter()
        latency = t1 - t0
        cache_latencies.append(latency)
        cache_hits.append(False)
        # Build prompt to store in cache
        ctx, _, _ = orch.retrieve_only(q)
        prompt = prompts.context_prompt().format(context=ctx, question=q)
        # Add to cache
        if cache_index is None:
            # Initialize index with correct dimension
            sample_dim = q_vec.shape[1]
            cache_index = faiss.IndexFlatIP(sample_dim)  # inner product == cosine after L2 norm
        cache_index.add(q_vec)
        cache_store.append((prompt, ans.strip()))
        print(f"[{idx:03d}/{len(questions)}] Q: {q}")
        print(f"      MISS | latency={latency:.3f}s")
        print(f"      Answer: {ans[:80]}{'...' if len(ans)>80 else ''}")
        print()

    # ---------- 4. Statistics ----------
    def stats(lat_list: List[float]) -> Tuple[float, float]:
        if not lat_list:
            return (0.0, 0.0)
        avg = sum(lat_list) / len(lat_list)
        sorted_lst = sorted(lat_list)
        mid = len(sorted_lst) // 2
        median = (sorted_lst[mid] + sorted_lst[-mid-1]) / 2.0 if len(sorted_lst) % 2 == 0 else sorted_lst[mid]
        return (avg, median)

    base_avg, base_median = stats(baseline_latencies)
    cache_avg, cache_median = stats(cache_latencies)

    hit_latencies = [cache_latencies[i] for i, h in enumerate(cache_hits) if h]
    miss_latencies = [cache_latencies[i] for i, h in enumerate(cache_hits) if not h]
    avg_hit = sum(hit_latencies) / len(hit_latencies) if hit_latencies else 0.0
    avg_miss = sum(miss_latencies) / len(miss_latencies) if miss_latencies else 0.0

    hit_count = sum(cache_hits)
    miss_count = len(cache_hits) - hit_count
    hit_rate = hit_count / len(cache_hits) if cache_hits else 0.0

    latency_reduction = base_avg - cache_avg
    latency_reduction_pct = (latency_reduction / base_avg * 100.0) if base_avg != 0 else 0.0

    # ---------- 5. Output results ----------
    print("===== Baseline =====")
    print(f"Total queries: {len(baseline_latencies)}")
    print(f"Average latency: {base_avg:.3f} s")
    print(f"Median latency: {base_median:.3f} s")
    print()
    print("===== Simple Semantic Cache =====")
    print(f"Total queries: {len(cache_latencies)}")
    print(f"Cache hits: {hit_count}")
    print(f"Cache misses: {miss_count}")
    print(f"Hit rate: {hit_rate:.2%}")
    print(f"Average latency: {cache_avg:.3f} s")
    print(f"Median latency: {cache_median:.3f} s")
    print(f"Average hit latency: {avg_hit:.3f} s")
    print(f"Average miss latency: {avg_miss:.3f} s")
    print()
    print("===== Comparison =====")
    print(f"Baseline average latency: {base_avg:.3f} s")
    print(f"SmartCache average latency: {cache_avg:.3f} s")
    print(f"Latency reduction: {latency_reduction:.3f} s")
    print(f"Latency reduction %: {latency_reduction_pct:.2f} %")
    print()
    # Optional: per-category breakdown if we had labels; we can infer from order:
    # Questions order: [ExactDup1, ExactDup2, SemSim1, SemSim2, Unrel1, Unrel2]
    # We'll compute simple stats for each pair.
    def pair_stats(start_idx: int, label: str):
        indices = [start_idx, start_idx+1]
        lat = [cache_latencies[i] for i in indices if i < len(cache_latencies)]
        hits = [cache_hits[i] for i in indices if i < len(cache_hits)]
        hit_rate_pair = sum(hits) / len(hits) if hits else 0.0
        avg_lat_pair = sum(lat) / len(lat) if lat else 0.0
        print(f"{label}: queries=2, hits={sum(hits)}, hit_rate={hit_rate_pair:.2%}, avg_lat={avg_lat_pair:.3f}s")

    print("===== By Category (Semantic Cache) =====")
    if len(questions) >= 6:
        pair_stats(0, "Exact duplicate")
        pair_stats(2, "Semantic similar")
        pair_stats(4, "Unrelated")
    else:
        print("Not enough questions for category breakdown.")

if __name__ == "__main__":
    main()