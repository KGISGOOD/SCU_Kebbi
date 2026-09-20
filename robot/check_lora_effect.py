#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare logits of base Llama 3.2 3B Instruct and the same model with LoRA adapter
from lora_output/. Uses the native chat template, does not generate text.
Only performs forward pass and compares logits of the last input token.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    lora_adapter_path = os.path.join(os.path.dirname(__file__), "lora_output", "checkpoint-36")
    test_question = "學長，東吳大學是一所什麼樣的大學啊？"

    # ----- Check LoRA adapter path -----
    if not os.path.isdir(lora_adapter_path):
        print(f"[錯誤] LoRA adapter 目錄不存在: {lora_adapter_path}")
        return
    print(f"[診斷] 檢查 LoRA adapter 路徑: {lora_adapter_path}")

    # ----- Load tokenizer -----
    print(f"[診斷] 載入 tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # ----- Load base model -----
    print(f"[診斷] 載入基底模型: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    # ----- Load LoRA adapted model -----
    print(f"[診斷] 載入 LoRA adapter 從: {lora_adapter_path}")
    try:
        lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        lora_model.eval()
        print("[診斷] LoRA adapter 載入成功")
    except Exception as e:
        print(f"[錯誤] 載入 LoRA adapter 失敗: {e}")
        return

    # ----- Verify LoRA config exists -----
    if hasattr(lora_model, "peft_config"):
        print(f"[診斷] LoRA 配置: {lora_model.peft_config}")
    else:
        print("[警告] 無法取得 peft_config")

    # ----- Build input using Llama chat template -----
    messages = [{"role": "user", "content": test_question}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = base_model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ----- Forward pass (no grad) -----
    print("[診斷] 執行 forward pass...")
    with torch.no_grad():
        base_outputs = base_model(**inputs)
        lora_outputs = lora_model(**inputs)

    # Logits shape: (batch_size, seq_len, vocab_size)
    base_logits = base_outputs.logits  # torch.Tensor
    lora_logits = lora_outputs.logits

    # Use the last token position (the position right before generation)
    last_idx = base_logits.shape[1] - 1
    base_last_logits = base_logits[0, last_idx, :]  # (vocab_size,)
    lora_last_logits = lora_logits[0, last_idx, :]

    print("\n=== Logits comparison (last input token) ===")
    print(f"Logits tensor shape: {base_logits.shape}")
    print(f"Base logits – max: {base_last_logits.max().item():.6f}, min: {base_last_logits.min().item():.6f}")
    print(f"LoRA logits – max: {lora_last_logits.max().item():.6f}, min: {lora_last_logits.min().item():.6f}")

    diff = torch.abs(base_last_logits - lora_last_logits)
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()
    num_diff = torch.sum(base_last_logits != lora_last_logits).item()
    total_elem = base_last_logits.numel()

    print(f"最大絕對差異: {max_abs_diff:.6f}")
    print(f"平均絕對差異: {mean_abs_diff:.6f}")
    print(f"不同元素數量: {num_diff} / {total_elem}")

    # ----- Top-10 tokens -----
    def topk(logits, k=10):
        values, indices = torch.topk(logits, k)
        return values.tolist(), indices.tolist()

    base_vals, base_ids = topk(base_last_logits, 10)
    lora_vals, lora_ids = topk(lora_last_logits, 10)

    print("\n=== Top-10 tokens (by logit) ===")
    print(f"{'Rank':>4} {'Token ID':>9} {'Decoded Token':>15} {'Base Logit':>12} {'LoRA Logit':>12}")
    print("-" * 60)
    for rank in range(10):
        b_id = base_ids[rank]
        l_id = lora_ids[rank]
        b_tok = tokenizer.decode([b_id])
        l_tok = tokenizer.decode([l_id])
        print(f"{rank+1:>4} {b_id:>9} {b_tok:>15} {base_vals[rank]:>12.4f} {lora_vals[rank]:>12.4f}")

    # ----- Simple diagnosis -----
    if max_abs_diff > 1e-3 or mean_abs_diff > 1e-4:
        print("\n[診斷結果] LoRA 實際有影響模型輸出")
    else:
        print("\n[診斷結果] LoRA 可能沒有實際作用")

if __name__ == "__main__":
    main()