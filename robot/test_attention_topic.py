#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第一階段實驗：觀察 Llama 3.2 3B Instruct 的 Self-Attention
在 prefill 階段，測試 Current Query 對 History 的平均 attention。
不進行任何訓練、不生成回答、不修改既有檔案。
"""
import os
import torch
import statistics
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_token_spans(input_ids_list, question_ids):
    """
    Return (start, end) token indices for `question_ids` inside `input_ids_list`.
    Assumes question_ids appears exactly once.
    """
    q_len = len(question_ids)
    for i in range(len(input_ids_list) - q_len + 1):
        if input_ids_list[i:i+q_len] == question_ids:
            return i, i+q_len
    # Fallback: assume question is at the end
    return len(input_ids_list)-q_len, len(input_ids_list)

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

    # 定義測試案例：10 組 same_topic、10 組 topic_shift
    test_cases = [
        # same_topic 1
        {
            "name": "Case1_same_topic",
            "q1": "巨資學院有哪些特色？",
            "q2": "那它有哪些相關課程？",
            "expected": "same_topic"
        },
        # same_topic 2
        {
            "name": "Case2_same_topic",
            "q1": "巨資學院的核心課程是什麼？",
            "q2": "請問巨資學院有哪些實習機會？",
            "expected": "same_topic"
        },
        # same_topic 3
        {
            "name": "Case3_same_topic",
            "q1": "如何申請巨資學院的雙聯學制？",
            "q2": "申請需要準備哪些文件？",
            "expected": "same_topic"
        },
        # same_topic 4
        {
            "name": "Case4_same_topic",
            "q1": "巨資學院畢業生多半從事什麼工作？",
            "q2": "他們通常會到哪些產業就業？",
            "expected": "same_topic"
        },
        # same_topic 5
        {
            "name": "Case5_same_topic",
            "q1": "巨資學院與其他學系的跨領域合作有哪些？",
            "q2": "有哪些教授在進行資料科學與金融的結合研究？",
            "expected": "same_topic"
        },
        # same_topic 6
        {
            "name": "Case6_same_topic",
            "q1": "巨資學院的學生社團有哪些？",
            "q2": "他們通常組織哪些類型的活動？",
            "expected": "same_topic"
        },
        # same_topic 7
        {
            "name": "Case7_same_topic",
            "q1": "巨資學院的課程設計強調什麼能力？",
            "q2": "因此實作題目佔比是多少？",
            "expected": "same_topic"
        },
        # same_topic 8
        {
            "name": "Case8_same_topic",
            "q1": "巨資學院的師資背景如何？",
            "q2": "他們多半具有產業經驗還是學術背景？",
            "expected": "same_topic"
        },
        # same_topic 9
        {
            "name": "Case9_same_topic",
            "q1": "巨資學院與業界的合作專案有哪些？",
            "q2": "最近完成的產學合作案例是什麼？",
            "expected": "same_topic"
        },
        # same_topic 10
        {
            "name": "Case10_same_topic",
            "q1": "巨資學院的國際交流機會？",
            "q2": "有哪些海外大學可以選修雙學位？",
            "expected": "same_topic"
        },
        # topic_shift 1
        {
            "name": "Case11_topic_shift",
            "q1": "巨資學院有哪些特色？",
            "q2": "今天台北的天氣如何？",
            "expected": "topic_shift"
        },
        # topic_shift 2
        {
            "name": "Case12_topic_shift",
            "q1": "巨資學院的核心課程是什麼？",
            "q2": "資訊管理系的程式設計課程要學什麼語言？",
            "expected": "topic_shift"
        },
        # topic_shift 3
        {
            "name": "Case13_topic_shift",
            "q1": "如何申請巨資學院的雙聯學制？",
            "q2": "校園內的餐廳有哪些素食選擇？",
            "expected": "topic_shift"
        },
        # topic_shift 4
        {
            "name": "Case14_topic_shift",
            "q1": "巨資學院畢業生多半從事什麼工作？",
            "q2": "圖書館的借書規則是什麼？",
            "expected": "topic_shift"
        },
        # topic_shift 5
        {
            "name": "Case15_topic_shift",
            "q1": "巨資學院與其他學系的跨領域合作有哪些？",
            "q2": "運動中心的開放時間是幾點？",
            "expected": "topic_shift"
        },
        # topic_shift 6
        {
            "name": "Case16_topic_shift",
            "q1": "巨資學院的學生社團有哪些？",
            "q2": "學雜費每學期多少錢？",
            "expected": "topic_shift"
        },
        # topic_shift 7
        {
            "name": "Case17_topic_shift",
            "q1": "巨資學院的課程設計強調什麼能力？",
            "q2": "最近的校園演講活動有哪些主題？",
            "expected": "topic_shift"
        },
        # topic_shift 8
        {
            "name": "Case18_topic_shift",
            "q1": "巨資學院的師資背景如何？",
            "q2": "宿舍申請流程需要哪些步驟？",
            "expected": "topic_shift"
        },
        # topic_shift 9
        {
            "name": "Case19_topic_shift",
            "q1": "巨資學院與業界的合作專案有哪些？",
            "q2": "交通車時刻表怎麼查？",
            "expected": "topic_shift"
        },
        # topic_shift 10
        {
            "name": "Case20_topic_shift",
            "q1": "巨資學院的國際交流機會？",
            "q2": "電腦教室的開放時間是什麼？",
            "expected": "topic_shift"
        },
    ]

    same_ratios = []
    topic_ratios = []

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
        input_ids_tensor = inputs["input_ids"]  # shape [1, seq_len]
        input_ids_list = input_ids_tensor[0].tolist()
        seq_len = input_ids_tensor.shape[-1]
        # Tokenize q1 and q2 alone to get their token ids (no special tokens)
        q1_ids = tokenizer.encode(q1, add_special_tokens=False)
        q2_ids = tokenizer.encode(q2, add_special_tokens=False)

        # Find spans in the tokenized prompt (excluding special tokens? the prompt from apply_chat_template includes special tokens,
        # but we encoded without special tokens, so we need to map. We'll search directly in input_ids_list.)
        h_start, h_end = get_token_spans(input_ids_list, q1_ids)
        q_start, q_end = get_token_spans(input_ids_list, q2_ids)

        # Safety: ensure spans are within seq_len
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

        # Store ratio for stats
        if expected == "same_topic":
            same_ratios.append(history_attention_ratio)
        else:
            topic_ratios.append(history_attention_ratio)

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

    # ===== Summary =====
    print("=== Summary ===")
    if same_ratios:
        print("Same Topic:")
        print(f"  count: {len(same_ratios)}")
        print(f"  average: {statistics.mean(same_ratios):.6f}")
        print(f"  median: {statistics.median(same_ratios):.6f}")
        print(f"  min: {min(same_ratios):.6f}")
        print(f"  max: {max(same_ratios):.6f}")
    else:
        print("Same Topic: (no data)")

    if topic_ratios:
        print("Topic Shift:")
        print(f"  count: {len(topic_ratios)}")
        print(f"  average: {statistics.mean(topic_ratios):.6f}")
        print(f"  median: {statistics.median(topic_ratios):.6f}")
        print(f"  min: {min(topic_ratios):.6f}")
        print(f"  max: {max(topic_ratios):.6f}")
    else:
        print("Topic Shift: (no data)")

    if same_ratios and topic_ratios:
        diff = statistics.mean(same_ratios) - statistics.mean(topic_ratios)
        print(f"Difference:")
        print(f"  same_topic_avg - topic_shift_avg = {diff:.6f}")
        # simple ordering comparison: count how many same_topic ratios > each topic_shift? We'll just note.
        # Compute proportion of same_topic ratios greater than each topic_shift ratio? We'll do a simple pairwise:
        # For each same, count how many topic_shift are lower -> not required.
        # We'll just output a note.
        # Let's compute how many same_topic ratios are greater than the median of topic_shift.
        same_above_topic_median = sum(1 for r in same_ratios if r > statistics.median(topic_ratios))
        print(f"  Number of same_topic ratios > topic_shift median: {same_above_topic_median} / {len(same_ratios)}")
    else:
        print("Difference: (cannot compute)")

if __name__ == "__main__":
    main()