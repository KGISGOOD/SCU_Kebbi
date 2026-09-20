#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Speculative Decoding benchmark (clean version) for Llama 3.2 3B Instruct target
and Llama 3.2 1B Instruct draft using Hugging Face assisted generation.
Does not modify any existing files.
"""
import os
import time
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Print Transformers version
    print(f"[診斷] Transformers 版本: {transformers.__version__}")

    target_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    draft_model_name = "meta-llama/Llama-3.2-1B-Instruct"

    # Load tokenizer (shared)
    print(f"[診斷] 載入 tokenizer: {target_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(target_model_name, trust_remote_code=True)

    # Load target model
    print(f"[診斷] 載入目標模型: {target_model_name}")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16,   # recommended instead of torch_dtype in newer versions
    )
    target_model.eval()

    # Load draft model
    print(f"[診斷] 載入草稿模型: {draft_model_name}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16,
    )
    draft_model.eval()

    # Fixed generation settings
    generation_kwargs = {
        "do_sample": False,
        "max_new_tokens": 512,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # Ensure deterministic behavior (seed)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    questions = [
        "請介紹資料科學系。",
        "資料科學系有哪些特色？",
        "資訊系學生需要學習哪些程式設計相關課程？",
        "學校的跨領域教育主要如何培養學生？",
        "請介紹學校的人工智慧相關課程。",
        "資料科學系統來有哪些發展方向？"
    ]

    # Warm-up (not counted)
    print("\n[Warm-up] 執行 baseline warm-up...")
    warm_inputs = tokenizer("測試", return_tensors="pt").to(target_model.device)
    with torch.no_grad():
        target_model.generate(**warm_inputs, **generation_kwargs)
    print("[Warm-up] 執行 speculative warm-up...")
    warm_inputs = tokenizer("測試", return_tensors="pt").to(target_model.device)
    with torch.no_grad():
        target_model.generate(**warm_inputs, assistant_model=draft_model, **generation_kwargs)
    print("[Warm-up] 完成。\n")

    # Containers for results
    baseline_times = []
    baseline_tokens = []
    baseline_answers = []

    speculative_times = []
    speculative_tokens = []
    speculative_answers = []

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
        baseline_answers.append(answer)

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
        speculative_answers.append(answer)

        print(f"--- Q{idx} ---")
        print(f"問題：{q}")
        print(f"回答：{answer}")
        print(f"生成 token 數：{gen_tokens}")
        print(f"耗時：{elapsed:.4f} 秒")
        print(f"每秒 token 數：{gen_tokens/elapsed if elapsed>0 else 0:.2f}\n")

    # Compute averages
    def avg(lst):
        return sum(lst)/len(lst) if lst else 0

    avg_base_time = avg(baseline_times)
    avg_spec_time = avg(speculative_times)
    avg_base_tokpersec = avg([t/ti if ti>0 else 0 for t,ti in zip(baseline_tokens, baseline_times)])
    avg_spec_tokpersec = avg([t/ti if ti>0 else 0 for t,ti in zip(speculative_tokens, speculative_times)])
    speedup = avg_base_time / avg_spec_time if avg_spec_time > 0 else 0

    # Compare answers
    matches = sum(1 for b, s in zip(baseline_answers, speculative_answers) if b == s)
    mismatches = len(questions) - matches

    # Summary output
    print("========================================")
    print("Speculative Decoding Experiment (clean)")
    print("===============================")
    print("[Baseline]")
    for idx, (q, t, tok, sec) in enumerate(zip(questions, baseline_times, baseline_tokens,
                                                [t/ti if ti>0 else 0 for t,ti in zip(baseline_tokens, baseline_times)]), start=1):
        print(f"Q{idx} ...")
        print(f"Time: {t:.4f}")
        print(f"Tokens: {tok}")
        print(f"Tokens/sec: {sec:.2f}")
        print()
    print("[Speculative]")
    for idx, (q, t, tok, sec) in enumerate(zip(questions, speculative_times, speculative_tokens,
                                                [t/ti if ti>0 else 0 for t,ti in zip(speculative_tokens, speculative_times)]), start=1):
        print(f"Q{idx} ...")
        print(f"Time: {t:.4f}")
        print(f"Tokens: {tok}")
        print(f"Tokens/sec: {sec:.2f}")
        print()
    print("========================================")
    print("Summary")
    print("=======")
    print(f"Baseline Avg Latency: {avg_base_time:.4f} 秒")
    print(f"Speculative Avg Latency: {avg_spec_time:.4f} 秒")
    print(f"Baseline Avg Tokens/sec: {avg_base_tokpersec:.2f}")
    print(f"Speculative Avg Tokens/sec: {avg_spec_tokpersec:.2f}")
    print(f"Speedup (Baseline/Speculative): {speedup:.2f}")
    print(f"答案比較：完全一致題數 {matches} / {len(questions)}")
    print(f"答案比較：不一致題數 {mismatches} / {len(questions)}")
    print("(註：目前 Transformers assisted generation API 無法直接取得 acceptance rate。)")

if __name__ == "__main__":
    main()