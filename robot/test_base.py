#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
測試微調前的 Llama 3.2 3B Instruct（Base model）。
不載入任何 LoRA adapter。
"""
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    questions = [
        "為什麼學校要特別成立「巨量資料管理學院（巨資學院）」呢？",
        "巨資學院跟一般的「資工系」有什麼不同？",
        "那巨資學院跟一般的「資管系」差別在哪裡？"
    ]

    print(f"[診斷] 載入 tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    print(f"[診斷] 載入基底模型: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    gen_kwargs = {
        "do_sample": False,
        "max_new_tokens": 256,
        "pad_token_id": tokenizer.eos_token_id,
    }

    times = []

    print("\n=== Base Llama 3.2 ===\n")
    for idx, question in enumerate(questions, start=1):
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        device = base_model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            output_ids = base_model.generate(**inputs, **gen_kwargs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        times.append(elapsed)

        gen_ids = output_ids[:, input_len:]
        answer = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

        print(f"--- Q{idx} ---")
        print(f"問題：{question}")
        print(f"回答：{answer}\n")

    avg_time = sum(times) / len(times) if times else 0
    print("=== 完成 ===")
    print(f"平均生成時間：{avg_time:.4f} 秒")

if __name__ == "__main__":
    main()