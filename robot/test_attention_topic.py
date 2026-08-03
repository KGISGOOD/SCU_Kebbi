#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
測試演算法二：自注意力機制話題識別（整合真實 RAG 檢索器）
目的：透過取得模型 self-attention 分數，計算當前 query token
對歷史 token（系統說明 + 檢索段落）的注意力比例，以判斷是否為話題切換。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---- 引入專案現有的 RAG 組件 ----
try:
    from config import AppSettings
    from embeddings import HFEmbeddingsProvider
    from vectorstores import VectorStoreLoader
    from retriever import MultiStoreRetriever
    from prompts import PromptFactory
except Exception as e:
    # 若無法匯入（例如在獨立測試時），回退到簡易 DummyRetriever 以免整個腳本失敗
    print(f"警告：無法載入專案模組({e})，將使用 DummyRetriever 作為備用。")
    AppSettings = None


def build_prompt_with_retriever(question: str, retriever):
    """使用真實的 Retriever 取得相關文件並組成 Prompt"""
    retrieved_docs = retriever.get_relevant_documents(question)
    context = "\n".join(retrieved_docs)
    prompt = f"""你是一個友好的助理。請根據以下背景資料回答問題。

背景資料：
{context}

問題：{question}
答案："""
    return prompt, retrieved_docs


def main():
    # --- 設定 ---
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # 可自行更換
    TOPIC_SHIFT_THRESHOLD = 0.40  # 經驗值：問題對過去的注意力過低時視為話題切換
    DELTA_THRESHOLD = 0.12        # 變化門檻：比例驟變時也判斷為切換

    # --- 載入模型與 tokenizer ---
    print(f"Loading model {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        output_attentions=True,   # 關鍵：啟用 attentions
    )
    model.eval()

    # --- 初始化真實的 RAG 檢索器（若可用）---
    if AppSettings is not None:
        try:
            settings = AppSettings()
            emb = HFEmbeddingsProvider(settings.model_name, settings.use_cpu)
            loader = VectorStoreLoader(emb)
            stores = loader.load_all_from_dir(settings.parent_vector_dir)
            retriever = MultiStoreRetriever(stores, top_k=settings.top_k, fetch_k=settings.fetch_k)
            print("成功載入專案的 MultiStoreRetriever。")
        except Exception as e:
            print(f"載入專案 Retriever 失敗({e})，改用 DummyRetriever。")
            retriever = None
    else:
        retriever = None

    # 若仍未取得 retriever，則使用簡易的 DummyRetriever（僅純問題作為 prompt）
    if retriever is None:
        class DummyRetriever:
            def get_relevant_documents(self, question):
                return []  # 無檢索內容
        retriever = DummyRetriever()

    # --- 讀取測試問題 ---
    questions_path = os.path.join(os.path.dirname(__file__), "speed_test_questions.txt")
    with open(questions_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    questions = lines

    topic_shift_count = 0
    prev_ratio = None

    for idx, q in enumerate(questions, start=1):
        # 1. 建立 Prompt (透過真實 RAG)
        prompt, _ = build_prompt_with_retriever(q, retriever)

        # 2. 取得模型 self-attention (僅 forward，不產出 token)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        # 3. 取得注意力張量並平均層/head
        attn = torch.stack(outputs.attentions)          # [L, B, H, S, S]
        attn_avg = attn.mean(dim=(0, 1, 2))             # [S, S]，每行和 = 1

        # ------- 取得「純問題」的 token 範圍（不含系統 Prompt 與答案說明） -------
        # 假設 prompt 格式為： 「你是一個友好的助理。請根據以下背景資料回答問題。
        # 背景資料：{context}
        # 問題：{question}
        # 答案：」
        # 我們只要把「問題：{question}」那段切出來。
        # 先Tokenizer整個prompt，再找出「問題：」的起始位置。
        prompt_ids = inputs.input_ids[0].tolist()
        question_token_ids = tokenizer(q, add_special_tokens=False).input_ids
        q_len = len(question_token_ids)

        # 從後往前找 question token 序列在 prompt_ids 中的起始 index
        # 這裡採用簡單的暴力匹配（因為題目通常不重複，且長度較短）
        q_start = None
        for i in range(len(prompt_ids) - q_len + 1):
            if prompt_ids[i:i+q_len] == question_token_ids:
                q_start = i
                break
        if q_start is None:               # 找不到就退而求其次：假設最後 q_len 個 token 是問題
            q_start = len(prompt_ids) - q_len
        q_end = q_start + q_len

        # prompt 長度（不包括問題本身）＝ 系統說明 + 背景資料 + 「問題：」等固定前綴
        prompt_len = q_start                # 這裡把「問題：」之前的所有 token 視為「prompt」

        # ------- 計算問題對 prompt 的注意力 vs 問題對自己的注意力 -------
        # 問題的所有 token 對全部 prompt 的平均注意力
        attn_q_to_prompt = attn_avg[q_start:q_end, :prompt_len].mean().item()
        # 問題的所有 token 對自己的平均注意力（對角線）
        attn_q_to_self   = attn_avg[q_start:q_end, q_start:q_end].mean().item()

        ratio = attn_q_to_prompt / (attn_q_to_prompt + attn_q_to_self) if (attn_q_to_prompt + attn_q_to_self) != 0 else 0.0

        # 4. 判斷話題是否切換（絕對門檻 + 變化門檻）
        topic_shift = False
        if ratio < TOPIC_SHIFT_THRESHOLD:                # 經驗值：問題對過去的注意力過低 → 可能切換
            topic_shift = True
        elif prev_ratio is not None and abs(ratio - prev_ratio) > DELTA_THRESHOLD:   # 變化大 → 可能切換
            topic_shift = True

        if topic_shift:
            topic_shift_count += 1

        print(f"[{idx:03d}/{len(questions)}] Q: {q}")
        print(f"      q→prompt ratio = {ratio:.4f} -> topic_shift = {topic_shift}")
        print()

        prev_ratio = ratio

    # --- 結果彙總 ---
    shift_rate = topic_shift_count / len(questions) if questions else 0
    print("\n=== 測試結果 ===")
    print(f"總問題數      : {len(questions)}")
    print(f"話題切換次數 : {topic_shift_count}")
    print(f"話題切換比率 : {shift_rate:.2%}")


if __name__ == "__main__":
    main()