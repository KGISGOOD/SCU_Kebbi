import os
import time
import pickle
import numpy as np
import faiss

# —— 您專案已有的模組 ——
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
    """把 shape 為 (N, D) 的矩陣做 L2 正規化（每一行）"""
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / (norm + 1e-10)


def build_prompt(question: str, retriever: MultiStoreRetriever):
    """沿用您原本的檢索＋組 prompt 邏輯"""
    retrieved_docs = retriever.get_relevant_documents(question)
    context = "\n".join(retrieved_docs)
    prompt = f"""你是一個友好的助理。請根據以下背景資料回答問題。

背景資料：
{context}

問題：{question}
答案："""
    return prompt, retrieved_docs


def main():
    # ---------- 0️⃣ 嘗試載入既有快取 ----------
    cache_index_path = os.path.join(os.path.dirname(__file__), "smartcache.index")
    cache_store_path = os.path.join(os.path.dirname(__file__), "smartcache_store.pkl")
    if os.path.exists(cache_index_path) and os.path.exists(cache_store_path):
        cache_index = faiss.read_index(cache_index_path)
        with open(cache_store_path, "rb") as f:
            cache_store = pickle.load(f)
        print(f"已載入既有快取：{len(cache_store)} 條記錄")
    else:
        # 若檔案不存在則先建立空的快取（維度會在後續初始化 emb 時重新設定）
        cache_index = None
        cache_store = []

    # ---------- 1️⃣ 初始化設定 ----------
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

    # ---------- 2️⃣ 確保快取結構已正確初始化 ----------
    # 若剛才載入失敗（cache_index 為 None），則依據 emb 取得維度建立空快取
    if cache_index is None:
        sample_vec = emb.get().embed_query("test")
        embed_dim = len(sample_vec)
        cache_index = faiss.IndexFlatIP(embed_dim)
        cache_store = []

    # ---------- 3️⃣ 讀取測試問題 ----------
    questions_path = os.path.join(os.path.dirname(__file__), "speed_test_questions.txt")
    with open(questions_path, encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    # ---------- 4️⃣ 主迴圈 ----------
    hit_count = 0
    total_latency = 0.0
    answers = []  # 可供後續相似度計算使用

    for idx, q in enumerate(questions, start=1):
        prompt, _ = build_prompt(q, retriever)

        # ---- 向量化問題並 L2 正規化 ----
        q_vec = np.array(emb.get().embed_query(q)).reshape(1, -1)   # (1, D)
        q_vec = l2_normalize(q_vec)

        # ---- 嘗試快取命中 ----
        D, I = cache_index.search(q_vec, 1)          # D: similarity, I: index
        similarity = float(D[0][0])
        cached_idx = int(I[0][0])

        if similarity >= 0.75 and cached_idx < len(cache_store):
            # 命中
            hit = True
            answer = cache_store[cached_idx][1]      # 已存在的答案
            latency = 0.0                            # 向量檢索時間極小，可視為 0
            hit_count += 1
        else:
            # 未命中 → 調用 LLM 產出答案
            hit = False
            t0 = time.perf_counter()
            answer, llm_time = orch.ask(q, [])       # orch.ask 回傳 (answer, llm_time)
            t1 = time.perf_counter()
            latency = t1 - t0                        # 包含檢索 + LLM 時間

            # ---- 寫入快取 ----
            a_vec = np.array(emb.get().embed_query(q)).reshape(1, -1)
            a_vec = l2_normalize(a_vec)
            cache_index.add(a_vec)
            cache_store.append((prompt, answer.strip()))

        total_latency += latency
        answers.append(answer)

        # 即時進度（顯示問題、命中狀態、延遲以及完整答案）
        status = "HIT" if hit else "MISS"
        print(f"[{idx:03d}/{len(questions)}] Q: {q}")
        print(f"      {status} | latency={latency:.3f}s")
        print(f"      Answer: {answer}")
        print()  # empty line for readability

    # ---------- 5️⃣ 結果彙總 ----------
    hit_rate = hit_count / len(questions) if questions else 0
    avg_latency = total_latency / len(questions) if questions else 0

    print("\n=== 測試結果 ===")
    print(f"總問題數      : {len(questions)}")
    print(f"快取命中次數 : {hit_count}")
    print(f"命中率       : {hit_rate:.2%}")
    print(f"平均延遲     : {avg_latency:.3f} 秒/題")

    # ---------- 6️⃣ （可選）持久化快取 ----------
    # 每次執行結束時都把最新的快取寫回磁碟，以供下次使用
    faiss.write_index(cache_index, cache_index_path)
    with open(cache_store_path, "wb") as f:
        pickle.dump(cache_store, f)
    print(f"快取已儲存至 {cache_index_path} 與 {cache_store_path}")


if __name__ == "__main__":
    main()