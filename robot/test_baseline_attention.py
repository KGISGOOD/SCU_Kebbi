#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment measures the computational overhead of Attention Topic Identification
added to the baseline RAG pipeline. The attention result does not affect retrieval or generation.
"""

import os
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- 基線所需的元件（直接復用既有程式） ----
from config import AppSettings
from embeddings import HFEmbeddingsProvider
from vectorstores import VectorStoreLoader
from retriever import MultiStoreRetriever
from llm.ollama import ChatOllamaLLM
from prompts import PromptFactory
from orchestrator import QAOrchestrator
from relevance import RelevancePolicy  # 僅為了符合 import，實際未使用

# ---- 小工具：L2 正規化（與現有快取腳本相同，雖然本實驗不使用） ----
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / (norm + 1e-10)

# ---- 取得 token 在完整 input_ids 中的起始/結束位置（直接搬自 test_attention_topic.py） ----
def get_token_spans(input_ids_list, question_ids):
    q_len = len(question_ids)
    for i in range(len(input_ids_list) - q_len + 1):
        if input_ids_list[i:i+q_len] == question_ids:
            return i, i+q_len
    # fallback：假設在結尾
    return len(input_ids_list)-q_len, len(input_ids_list)

# ---- Attention 計算（完整流程計時） ----
def compute_attention_ratio(history: str, current: str,
                            attn_model, attn_tokenizer, device):
    """
    返回 (attention_ratio, attention_latency_seconds)
    此函式現在包含完整的 Attention Topic Identification 流程：
    1. 建立 messages
    2. apply_chat_template
    3. tokenizer
    4. 移到 device
    5. model forward (output_attentions=True)
    6. 取得 attentions 與做 all‑layers/all‑heads 平均
    7. 找 history / current token spans
    8. 計算 Query→History、Query→Query 平均注意力
    9. 計算 History Attention Ratio
    """
    # ----- 開始計時 -----
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()

    # 1. 建立兩則使用者訊息的 prompt（不加 generation prompt）
    messages = [
        {"role": "user", "content": history},
        {"role": "user", "content": current},
    ]
    prompt = attn_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    # 2. tokenizer 編碼
    inputs = attn_tokenizer(prompt, return_tensors="pt")
    # 3. 移到 device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 4‑5. Attention model forward
    with torch.no_grad():
        outputs = attn_model(**inputs)   # output_attentions=True 在模型載入時已設定

    # 6. 取得 attentions 與做平均
    attentions = outputs.attentions      # tuple length = num_layers
    all_layers_attn = torch.stack(attentions)          # (num_layers, batch, num_heads, seq_len, seq_len)
    all_attn_avg = all_layers_attn.mean(dim=(0,1))     # (batch, seq_len, seq_len)

    # 7. 找 token spans
    input_ids_list = inputs["input_ids"][0].tolist()
    history_ids = attn_tokenizer.encode(history, add_special_tokens=False)
    current_ids = attn_tokenizer.encode(current, add_special_tokens=False)
    h_start, h_end = get_token_spans(input_ids_list, history_ids)
    q_start, q_end = get_token_spans(input_ids_list, current_ids)

    # 8. 確保合法範圍
    h_start = max(0, min(h_start, len(input_ids_list)-1))
    h_end   = max(h_start+1, min(h_end,   len(input_ids_list)))
    q_start = max(0, min(q_start, len(input_ids_list)-1))
    q_end   = max(q_start+1, min(q_end,   len(input_ids_list)))

    # 9. 計算 Query→History 平均注意力
    q_to_h_attn = all_attn_avg[0, q_start:q_end, h_start:h_end]   # (query_len, history_len)
    q_to_h_avg = q_to_h_attn.mean().item()
    # 計算 Query→Query 平均注意力
    q_to_q_attn = all_attn_avg[0, q_start:q_end, q_start:q_end]   # (query_len, query_len)
    q_to_q_avg = q_to_q_attn.mean().item()

    if (q_to_h_avg + q_to_q_avg) != 0:
        ratio = q_to_h_avg / (q_to_h_avg + q_to_q_avg)
    else:
        ratio = 0.0

    # ----- 結束計時 -----
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    attn_latency = time.perf_counter() - start

    return ratio, attn_latency

def main():
    print("=== Baseline + Attention Overhead Experiment ===")
    print("This experiment measures the computational overhead of Attention Topic Identification")
    print("added to the baseline RAG pipeline. The attention result does not affect retrieval or generation.\n")

    # -------- 共用基線元件 --------
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
    relevance = RelevancePolicy()   # 只為了符合 import，實際不使用
    service = None  # 我們直接使用 orch.ask，不需要 ChatService

    # -------- Attention 模型（Hugging Face） --------
    attn_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"[Info] Loading attention model: {attn_model_name}")
    attn_tokenizer = AutoTokenizer.from_pretrained(attn_model_name, trust_remote_code=True)
    attn_model = AutoModelForCausalLM.from_pretrained(
        attn_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        output_attentions=True,   # 必須開啟
    )
    attn_model.eval()
    attn_device = attn_model.device

    # -------- Warm-up --------
    dummy_history = "測試"
    dummy_current = "測試"
    # Attention warm-up (完整流程)
    _ , _ = compute_attention_ratio(dummy_history, dummy_current,
                                    attn_model, attn_tokenizer, attn_device)
    # Baseline LLM warm-up (呼叫一次 RAG+LLM)
    _ , _ = orch.ask("測試", [])

    # -------- 測試資料（history + current_query） --------
    test_pairs = [
        {
            "history": "巨資學院有哪些特色？",
            "current": "那它有哪些課程？",
        },
        {
            "history": "巨資學院的核心課程是什麼？",
            "current": "請問巨資學院有哪些實習機會？",
        },
        {
            "history": "如何申請巨資學院的雙聯學制？",
            "current": "申請需要準備哪些文件？",
        },
        {
            "history": "巨資學院與其他學系的跨領域合作有哪些？",
            "current": "有哪些教授在進行資料科學與金融的結合研究？",
        },
        {
            "history": "巨資學院的學生社團有哪些？",
            "current": "他們通常組織哪些類型的活動？",
        },
    ]

    # ---------- 階段 A：純 Baseline ----------
    print("=== Phase A: Baseline (RAG → Llama 3.2) ===")
    baseline_latencies = []
    baseline_answers = []
    for idx, pair in enumerate(test_pairs, start=1):
        current = pair["current"]
        t0 = time.perf_counter()
        answer, _ = orch.ask(current, [])   # 不傳入 history，保持純 baseline
        t1 = time.perf_counter()
        latency = t1 - t0
        baseline_latencies.append(latency)
        baseline_answers.append(answer.strip())
        print(f"[{idx:02d}/{len(test_pairs)}] Baseline latency: {latency:.4f}s")
        print(f"      Answer: {answer[:100]}{'...' if len(answer)>100 else ''}")
        print()

    # ---------- 階段 B：Baseline + Attention ----------
    print("\n=== Phase B: Baseline + Attention ===")
    attention_latencies = []
    baseline_latencies_b = []   # 基線部分在 B 階段的延遲（應與 A 階段相近）
    total_latencies = []
    attention_ratios = []
    baseline_answers_b = []
    baseline_plus_attention_answers = []   # 應該與 baseline_answers_b 相同

    for idx, pair in enumerate(test_pairs, start=1):
        history = pair["history"]
        current  = pair["current"]

        # ----- Attention 完整計算 -----
        ratio, attn_lat = compute_attention_ratio(
            history, current, attn_model, attn_tokenizer, attn_device
        )
        attention_latencies.append(attn_lat)
        attention_ratios.append(ratio)

        # ----- Baseline RAG + LLM （只使用 current_query） --------
        t0 = time.perf_counter()
        answer, _ = orch.ask(current, [])   # 再次執行相同的 Baseline，只為了公平測量
        t1 = time.perf_counter()
        base_lat = t1 - t0
        baseline_latencies_b.append(base_lat)
        baseline_answers_b.append(answer.strip())

        total_lat = attn_lat + base_lat
        total_latencies.append(total_lat)
        baseline_plus_attention_answers.append(answer.strip())   # 同上

        # ----- 單筆輸出 -----
        print(f"--- Q{idx} ---")
        print(f"History      : {history}")
        print(f"Current Query: {current}")
        print(f"Attention Ratio   : {ratio:.6f}")
        print(f"Attention full computation latency : {attn_lat:.4f} s")
        print(f"Baseline latency  : {base_lat:.4f} s")
        print(f"Total latency     : {total_lat:.4f} s")
        print(f"Baseline answer   : {answer[:100]}{'...' if len(answer)>100 else ''}")
        print(f"Baseline+Att answer: {answer[:100]}{'...' if len(answer)>100 else ''}")
        print()

    # -------- 整體統計 --------
    def avg(lst): return sum(lst)/len(lst) if lst else 0.0
    def med(lst):
        s = sorted(lst)
        n = len(s)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return s[n//2]
        else:
            return (s[n//2-1] + s[n//2])/2.0

    avg_base   = avg(baseline_latencies)
    med_base   = med(baseline_latencies)

    avg_attn   = avg(attention_latencies)
    med_attn   = med(attention_latencies)
    avg_ratio  = avg(attention_ratios)

    avg_total  = avg(total_latencies)
    med_total  = med(total_latencies)

    avg_baseline_in_B = avg(baseline_latencies_b)   # 應與 avg_base 相近
    additional_latency = avg_total - avg_baseline_in_B
    latency_increase_pct = (additional_latency / avg_baseline_in_B * 100) if avg_baseline_in_B > 0 else 0.0

    print("=== Summary ===")
    print("--- Baseline (RAG → Llama 3.2) ---")
    print(f"Average latency : {avg_base:.4f} s")
    print(f"Median latency  : {med_base:.4f} s")
    print()
    print("--- Attention full computation ---")
    print(f"Average latency : {avg_attn:.4f} s")
    print(f"Median latency  : {med_attn:.4f} s")
    print(f"Average Attention Ratio : {avg_ratio:.6f}")
    print()
    print("--- Baseline + Attention ---")
    print(f"Average total latency : {avg_total:.4f} s")
    print(f"Median total latency  : {med_total:.4f} s")
    print()
    print("--- Comparison ---")
    print(f"Baseline average latency          : {avg_base:.4f} s")
    print(f"Baseline + Attention average latency : {avg_total:.4f} s")
    print(f"Additional latency due to Attention : {additional_latency:.4f} s")
    print(f"Latency increase percentage       : {latency_increase_pct:.2f}%")

if __name__ == "__main__":
    main()