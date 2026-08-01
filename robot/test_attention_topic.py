#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
測試演算法二：自注意力機制話題識別
目的：透過取得模型 self-attention 分數，計算當前 query token
對歷史 token 的注意力比例，以判斷是否為話題切換。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """載入可取得 attentions 的模型"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_attentions=True,  # 關鍵：啟用 attentions
    )
    return model, tokenizer

def get_attention_ratio(model, tokenizer, text):
    """
    輸入一段文字，返回：
    - ratio: 當前最後一個 token 對所有先前 token 的注意力平均佔比
    - attentions: 原始注意力張量 (layers, batch, heads, seq, seq)
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.attentions: tuple length = num_layers, each shape (batch, heads, seq, seq)
    attn = torch.stack(outputs.attentions)  # [layers, batch, heads, seq, seq]
    # 取最後一個 token (query) 對所有 token 的注意力
    last_idx = inputs.input_ids.shape[1] - 1
    query_attn = attn[:, :, :, last_idx, :]  # [layers, batch, heads, seq]
    # 平均所有層、所有 head
    avg_query_attn = query_attn.mean(dim=(0, 1, 2))  # [seq]
    # 切開：history (0..last_idx-1) 與 self (last_idx)
    history_attn = avg_query_attn[:-1].sum()
    self_attn = avg_query_attn[-1]
    ratio = (history_attn / (history_attn + self_attn)).item()
    return ratio, attn, inputs

def main():
    model, tokenizer = load_model()
    # 測試句子：前半段為上下文，後半段為明顯話題切換
    test_cases = [
        ("連續上下文", "今天天氣很好，適合去公園玩。"),
        ("話題切換", "今天天氣很好，適合去公園玩。那怎麼申請證補助？"),
        ("多輪切換", "老師，請問微積分的期末考範圍是什麼？另外，校園餐廳今天有什麼菜？"),
    ]
    for label, txt in test_cases:
        ratio, _, _ = get_attention_ratio(model, tokenizer, txt)
        print(f"[{label}] 文本: {txt}")
        print(f"      歷史注意力比例 = {ratio:.4f}")
        # 簡易判斷門檻（可自行調整）
        if ratio < 0.30:
            print("      → 偵測為話題切換 (Topic Shift)")
        else:
            print("      → 判定為上下文相續")
        print("-"*60)

if __name__ == "__main__":
    main()