#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第一階段實驗：觀察 Llama 3.2 3B Instruct 的 Self-Attention
在 prefill 階段，測試 Current Query 對 History 的平均 attention。
不進行任何訓練、不生成回答、不修改既有檔案。
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_token_spans(tokenizer, prompt, question):
    """
    Return (start, end) token indices for `question` inside `prompt`.
    Assumes question appears exactly once in prompt.
    """
    # Tokenize prompt and question separately
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    q_len = len(question_ids)
    # Find the question token sequence in prompt_ids
    for i in range(len(prompt_ids) - q_len + 1):
        if prompt_ids[i:i+q_len] == question_ids:
            return i, i+q_len
    # Fallback: assume question is at the end
    return len(prompt_ids)-q_len, len(prompt_ids)

def main():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"[診斷] 載入 tokenizer 與模型 (output_attentions=True): {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        output_attentions=True,   # 需要 attentions
    )
    model.eval()

    # 定義測試案例（至少 6 組）
    test_cases = [
        # Case 1: 同一話題
        {
            "name": "Case1_same_topic",
            "q1": "巨資學院有哪些特色？",
            "q2": "那它有哪些相關課程？",
            "expected": "same_topic"
        },
        # Case 2: 話題切換
        {
            "name": "Case2_topic_shift",
            "q1": "巨資學院有哪些特色？",
            "q2": "學校的雙聯學制怎麼申請？",
            "expected": "topic_shift"
        },
        # Additional similar cases
        {
            "name": "Case3_same_topic",
            "q1": "巨資學院的核心課程是什麼？",
            "q2": "請問巨資學院有哪些實習機會？",
            "expected": "same_topic"
        },
        {
            "name": "Case4_topic_shift",
            "q1": "巨資學院的核心課程是什麼？",
            "q2": "今天台北的天氣如何？",
            "expected": "topic_shift"
        },
        {
            "name": "Case5_same_topic",
            "q1": "如何申請巨資學院的雙聯學制？",
            "q2": "申請需要準備哪些文件？",
            "expected": "same_topic"
        },
        {
            "name": "Case6_topic_shift",
            "q1": "如何申請巨資學院的雙聯學制？",
            "q2": "遠距離教育平台有哪些優點？",
            "expected": "topic_shift"
        },
    ]

    print("\n=== 開始注意力實驗 (僅 forward，不生成) ===\n")
    for case in test_cases:
        name = case["name"]
        q1 = case["q1"]
        q2 = case["q2"]
        expected = case["expected"]

        # Build prompt with two user messages (no assistant)
        messages = [
            {"role": "user", "content": q1},
            {"role": "user", "content": q2},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,   # 只做 prefill，不加 generation prompt
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        attentions = outputs.attentions  # tuple of length num_layers, each shape (batch, num_heads, seq_len, seq_len)
        # Take last layer attention
        last_layer_attn = attentions[-1]  # (batch, num_heads, seq_len, seq_len)
        # Average over heads
        last_layer_attn_avg = last_layer_attn.mean(dim=1)  # (batch, seq_len, seq_len)
        # Also compute average over all layers and heads
        all_layers_attn = torch.stack(attentions)  # (num_layers, batch, num_heads, seq_len, seq_len)
        all_attn_avg = all_layers_attn.mean(dim=(0,1))  # (batch, seq_len, seq_len)

        # Get token spans for q1 (history) and q2 (current query) in the prompt
        # Note: prompt includes special tokens from chat template; we need to map.
        # We'll compute spans on the tokenized prompt (excluding special? include them)
        input_ids = inputs["input_ids"][0].tolist()
        # Tokenize q1 and q2 alone to get their token ids
        q1_ids = tokenizer.encode(q1, add_special_tokens=False)
        q2_ids = tokenizer.encode(q2, add_special_tokens=False)

        # Find spans
        h_start, h_end = get_token_spans(tokenizer, prompt, q1)
        q_start, q_end = get_token_spans(tokenizer, prompt, q2)

        # Safety: ensure spans are within seq_len
        seq_len = input_ids.shape[0]
        h_start = max(0, min(h_start, seq_len-1))
        h_end = max(h_start+1, min(h_end, seq_len))
        q_start = max(0, min(q_start, seq_len-1))
        q_end = max(q_start+1, min(q_end, seq_len))

        history_len = h_end - h_start
        query_len = q_end - q_start

        # Extract attention from query tokens to history tokens
        # shape: (batch, seq_len, seq_len) -> we want query->history
        # For each query token, average attention to all history tokens
        q_to_h_attn = all_attn_avg[0, q_start:q_end, h_start:h_end]  # (query_len, history_len)
        q_to_h_avg = q_to_h_attn.mean().item()
        # Query to self (query tokens attention to themselves)
        q_to_q_attn = all_attn_avg[0, q_start:q_end, q_start:q_end]  # (query_len, query_len)
        q_to_q_avg = q_to_q_attn.mean().item()
        # History self attention (optional)
        h_to_h_attn = all_attn_avg[0, h_start:h_end, h_start:h_end]
        h_to_h_avg = h_to_h_attn.mean().item()

        # History Attention Ratio (as defined in some literature: query->history / (query->history + query->self))
        if (q_to_h_avg + q_to_q_avg) != 0:
            history_attention_ratio = q_to_h_avg / (q_to_h_avg + q_to_q_avg)
        else:
            history_attention_ratio = 0.0

        print(f"--- {name} ---")
        print(f"Q1: {q1}")
        print(f"Q2: {q2}")
        print(f"預期類別: {expected}")
        print(f"History token 數量: {history_len}")
        print(f"Current Query token 數量: {query_len}")
        print(f"Attention tensor shape (batch, seq_len, seq_len): {list(all_attn_avg.shape)}")
        print(f"最後一層 attention shape (batch, seq_len, seq_len): {list(last_layer_attn_avg.shape)}")
        print(f"Query → History 平均 attention (所有層/head 平均): {q_to_h_avg:.6f}")
        print(f"Query → Current Query 平均 attention (所有層/head 平均): {q_to_q_avg:.6f}")
        print(f"History Attention Ratio: {history_attention_ratio:.6f}")
        print(f"(參考) 最後一層 Query→History 平均 attention: {q_to_h_attn.mean().item():.6f}")
        print(f"(參考) 最後一層 Query→Current Query 平均 attention: {q_to_q_attn.mean().item():.6f}")
        print()

if __name__ == "__main__":
    main()