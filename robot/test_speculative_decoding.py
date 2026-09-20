#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Speculative Decoding experiment using Hugging Face assisted generation.
Target: meta-llama/Llama-3.2-3B-Instruct
Draft: meta-llama/Llama-3.2-1B-Instruct
Baseline: target model only.
Does not modify any existing files.
"""
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    target_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    draft_model_name = "meta-llama/Llama-3.2-1B-Instruct"

    print(f"[診斷] 載入 tokenizer: {target_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(target_model_name, trust_remote_code=True)

    print(f"[診斷] 載入目標模型: {target_model_name}")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    target_model.eval()

    print(f"[診斷] 載入草稿模型: {draft_model_name}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    draft_model.eval()

    generation_kwargs = {
        "do_sample": False,
        "max_new_tokens": 128,
        "pad_token_id": tokenizer.eos_token_id,
    }

    questions = [
        "請介紹資料科學系。",
        "資料科學系有哪些特色？",
        "資訊系學生需要學習哪些程式設計相關課程？",
        "學校的跨領域教育主要如何培養學生？",
        "請介紹學校的人工智慧相關課程。",
        "資料科學系未來有哪些發展方向？"
    ]

    # Warm-up
    print("\n[Warm-up] 執行 baseline warm-up...")
    warm_inputs = tokenizer("測試", return_tensors="pt").to(target_model.device)
    with torch.no_grad():
        target_model.generate(**warm_inputs, **generation_kwargs)
    print("[Warm-up] 執行 speculative warm-up...")
    warm_inputs = tokenizer("測試", return_tensors="pt").to(target_model.device)
    with torch.no_grad():
        target_model.generate(**warm_inputs, assistant_model=draft_model, **generation_kwargs)
    print("[Warm-up] 完成。\n")

    baseline_times = []
    baseline_tokens = []
    speculative_times = []
    speculative_tokens = []

    print("=== 基線 (Target only) ===")
    for idx, q in enumerate(questions, start=1):
        inputs = tokenizer(q, return_tensors="pt")
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]
        inputs = {k: v.to(target_model.device) for k, v in inputs.items()}

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            output_ids = target_model.generate(**inputs, **generation_kwargs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start

        generated_ids = output_ids[:, input_len:]
        gen_tokens = generated_ids.shape[1]
        answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        baseline_times.append(elapsed)
        baseline_tokens.append(gen_tokens)

        print(f"--- Q{idx} ---")
        print(f"問題：{q}")
        print(f"回答：{answer}")
        print(f"生成 token 數：{gen_tokens}")
        print(f"耗時：{elapsed:.4f} 秒")
        print(f"每秒 token 數：{gen_tokens/elapsed if elapsed>0 else 0:.2f}\n")

    print("=== Speculative Decoding (Target + Draft) ===")
    for idx, q in enumerate(questions, start=1):
        inputs = tokenizer(q, return_tensors="pt")
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]
        inputs = {k: v.to(target_model.device) for k, v in inputs.items()}

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            output_ids = target_model.generate(**inputs, assistant_model=draft_model, **generation_kwargs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start

        generated_ids = output_ids[:, input_len:]
        gen_tokens = generated_ids.shape[1]
        answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        speculative_times.append(elapsed)
        speculative_tokens.append(gen_tokens)

        print(f"--- Q{idx} ---")
        print(f"問題：{q}")
        print(f"回答：{answer}")
        print(f"生成 token 數：{gen_tokens}")
        print(f"耗時：{elapsed:.4f} 秒")
        print(f"每秒 token 數：{gen_tokens/elapsed if elapsed>0 else 0:.2f}\n")

    # Summary
    def avg(lst):
        return sum(lst)/len(lst) if lst else 0

    avg_base_time = avg(baseline_times)
    avg_spec_time = avg(speculative_times)
    avg_base_tokpersec = avg([t/ti if ti>0 else 0 for t,ti in zip(baseline_tokens, baseline_times)])
    avg_spec_tokpersec = avg([t/ti if ti>0 else 0 for t,ti in zip(speculative_tokens, speculative_times)])
    speedup = avg_base_time / avg_spec_time if avg_spec_time > 0 else 0

    print("========================================")
    print("Speculative Decoding Experiment")
    print("===============================")
    print("[Baseline]")
    for idx, (q, t, tok, sec) in enumerate(zip(questions, baseline_times, baseline_tokens, [t/ti if ti>0 else 0 for t,ti in zip(baseline_tokens, baseline_times)]), start=1):
        print(f"Q{idx} ...")
        print(f"Time: {t:.4f}")
        print(f"Tokens: {tok}")
        print(f"Tokens/sec: {sec:.2f}")
        print()
    print("[Speculative]")
    for idx, (q, t, tok, sec) in enumerate(zip(questions, speculative_times, speculative_tokens, [t/ti if ti>0 else 0 for t,ti in zip(speculative_tokens, speculative_times)]), start=1):
        print(f"Q{idx} ...")
        print(f"Time: {t:.4f}")
        print(f"Tokens: {tok}")
        print(f"Tokens/sec: {sec:.2f}")
        print()
    print("========================================")
    print("Summary")
    print("=======")
    print(f"Baseline Avg Time: {avg_base_time:.4f} 秒")
    print(f"Speculative Avg Time: {avg_spec_time:.4f} 秒")
    print(f"Baseline Avg Tokens/sec: {avg_base_tokpersec:.2f}")
    print(f"Speculative Avg Tokens/sec: {avg_spec_tokpersec:.2f}")
    print(f"Speedup: {speedup:.2f}")
    print("(註：此實驗僅使用 Hugging Face assisted generation，未額外實作 Medusa、EAGLE 等。)")

if __name__ == "__main__":
    main()