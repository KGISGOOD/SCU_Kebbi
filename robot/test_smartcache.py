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
    """L2 normalize rows of a 2D array (or 1D array)."""
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / (norm + 1e-10)

def build_prompt(question: str, retriever: MultiStoreRetriever):
    """Retrieve context and build prompt for LLM."""
    retrieved_docs = retriever.get_relevant_documents(question)
    context = "\n".join(retrieved_docs)
    prompt = f"""你是一個友好的助理。請根據以下背景資料回答問題。

背景資料：
{context}

問題：{question}
答案："""
    return prompt, retrieved_docs

def main():
    # ---------- 初始化設定 ----------
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

    # ---------- 讀取測試問題 ----------
    questions_path = os.path.join(os.path.dirname(__file__), "speed_test_questions.txt")
    with open(questions_path, encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    # ---------- 建立快取結構 ----------
    # 取得嵌入維度（假設 BGE‑M3 為 1024，若取不到則直接寫死 1024）
    try:
        embed_dim = emb.get().client.encode(["test"]).shape[1]  # 若 emb.get() 回傳的是 HuggingFaceEmbeddings
    except Exception:
        embed_dim = 1024  # 預設維度

    cache_index = faiss.IndexFlatIP(embed_dim)   # inner product => cosine after L2 norm
    cache_store = []  # list of (prompt, answer)

    hit_count = 0
    total_latency = 0.0
    answers = []

    for idx, q in enumerate(questions, start=1):
        prompt, _ = build_prompt(q, retriever)

        # 嘗試快取命中
        q_vec = emb.get().client.encode([q])
        q_vec = l2_normalize(q_vec)
        D, I = cache_index.search(q_vec, 1)
        similarity = float(D[0][0])
        cache_latency = 0.0  # 向量化+搜尋耗時可忽略不計，若需測量可自行加計時
        if similarity >= 0.75 and int(I[0][0]) < len(cache_store):
            hit = True
            cached_answer = cache_store[int(I[0][0])][1]
            answer = cached_answer
            latency = cache_latency
            hit_count += 1
        else:
            hit = False
            # 未命中 → 呼叫 LLM 產生答案
            t0 = time.perf_counter()
            answer, llm_time = orch.ask(q, [])   # orch.ask 回傳 (answer, llm_time)
            t1 = time.perf_counter()
            latency = t1 - t0  # 包含向檢 + LLM 時間，與原始腳本一致
            # 將結果寫入快取
            a_vec = emb.get().client.encode([q])
            a_vec = l2_normalize(a_vec)
            cache_index.add(a_vec)
            cache_store.append((prompt, answer.strip()))
        total_latency += latency
        answers.append(answer)

        # 顯示進度（可選）
        status = "HIT" if hit else "MISS"
        print(f"[{idx:03d}/{len(questions)}] Q: {q[:30]}... | {status} | latency={latency:.3f}s")

    # ---------- 結果匯總 ----------
    hit_rate = hit_count / len(questions) if questions else 0
    avg_latency = total_latency / len(questions) if questions else 0

    print("\n=== 測試結果 ===")
    print(f"總問題數      : {len(questions)}")
    print(f"快取命中次數 : {hit_count}")
    print(f"命中率       : {hit_rate:.2%}")
    print(f"平均延遲     : {avg_latency:.3f} 秒/題")

    # ---------- （可選）持久化快取 ----------
    cache_index_path = os.path.join(os.path.dirname(__file__), "smartcache.index")
    cache_store_path = os.path.join(os.path.dirname(__file__), "smartcache_store.pkl")
    faiss.write_index(cache_index, cache_index_path)
    with open(cache_store_path, "wb") as f:
        pickle.dump(cache_store, f)
    print(f"快取已儲存至 {cache_index_path} 與 {cache_store_path}")

if __name__ == "__main__":
    main()