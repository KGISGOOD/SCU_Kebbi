#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比較原始 Llama 3.2 3B Instruct 與 LoRA checkpoint-36 的文字生成結果。
僅做推論，不進行訓練或修改任何既有檔案。
"""
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    lora_adapter_path = os.path.join(os.path.dirname(__file__), "lora_output", "checkpoint-36")
    questions = [
        "一句話總結，東吳巨資學院到底在做什麼？",
        "巨資學院跟一般的「資工系」有什麼不同？",
        "那巨資學院跟一般的「資管系」差別在哪裡？"
    ]

    if not os.path.isdir(lora_adapter_path):
        print(f"[錯誤] LoRA adapter 目錄不存在: {lora_adapter_path}")
        return

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

    print(f"[診斷] 載入 LoRA adapter 從: {lora_adapter_path}")
    lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    lora_model.eval()

    gen_kwargs = {
        "do_sample": False,
        "max_new_tokens": 256,
        "pad_token_id": tokenizer.eos_token_id,
    }

    base_times = []
    lora_times = []

    print("\n=== 文字生成比較 ===\n")
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

        # Base model
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            base_output_ids = base_model.generate(**inputs, **gen_kwargs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        base_elapsed = time.time() - start
        base_times.append(base_elapsed)

        base_gen_ids = base_output_ids[:, input_len:]
        base_answer = tokenizer.decode(base_gen_ids[0], skip_special_tokens=True).strip()

        # LoRA model
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            lora_output_ids = lora_model.generate(**inputs, **gen_kwargs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        lora_elapsed = time.time() - start
        lora_times.append(lora_elapsed)

        lora_gen_ids = lora_output_ids[:, input_len:]
        lora_answer = tokenizer.decode(lora_gen_ids[0], skip_special_tokens=True).strip()

        print(f"--- Q{idx} ---")
        print(f"問題：{question}")
        print(f"[Base]：{base_answer}")
        print(f"[LoRA]：{lora_answer}\n")

    avg_base = sum(base_times) / len(base_times) if base_times else 0
    avg_lora = sum(lora_times) / len(lora_times) if lora_times else 0
    print("=== 生成時間統計 ===")
    print(f"Base 平均生成時間：{avg_base:.4f} 秒")
    print(f"LoRA 平均生成時間：{avg_lora:.4f} 秒")

if __name__ == "__main__":
    main()