#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Strict LoRA A/B forward diagnostic.
Compares base model, LoRA-enabled, LoRA-disabled, and merged model logits
for a fixed input using the Llama 3.2 native chat template.
Does not modify any checkpoint or perform training.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def approx_equal(t1, t2, max_thr=1e-3, mean_thr=1e-4):
    """Return True if two tensors are approximately equal."""
    diff = torch.abs(t1 - t2)
    return diff.max().item() < max_thr and diff.mean().item() < mean_thr

def main():
    base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    lora_adapter_path = os.path.join(os.path.dirname(__file__), "lora_output", "checkpoint-36")
    test_question = "學長，東吳大學是一所什麼樣的大學啊？"

    if not os.path.isdir(lora_adapter_path):
        print(f"[錯誤] LoRA adapter 目錄不存在: {lora_adapter_path}")
        return

    # ----- Tokenizer -----
    print(f"[診斷] 載入 tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # ----- Base model -----
    print(f"[診斷] 載入基底模型: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    # ----- Build input once -----
    messages = [{"role": "user", "content": test_question}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = base_model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]

    with torch.inference_mode():
        # ----- Base forward -----
        base_outputs = base_model(**inputs)
        base_logits = base_outputs.logits  # (1, seq_len, vocab_size)
        last_idx = base_logits.shape[1] - 1
        base_last = base_logits[0, last_idx, :]  # (vocab_size,)

        print("\n=== Base model forward ===")
        print(f"Logits shape: {base_logits.shape}")
        print(f"Base last-token logits – max: {base_last.max().item():.6f}, min: {base_last.min().item():.6f}")

        # ----- Load LoRA adapter -----
        print(f"\n[診斷] 載入 LoRA adapter 從: {lora_adapter_path}")
        lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        lora_model.eval()

        # Adapter info
        if hasattr(lora_model, "peft_config"):
            print(f"[診斷] LoRA 配置: {lora_model.peft_config}")
        active = lora_model.active_adapter if hasattr(lora_model, "active_adapter") else "unknown"
        print(f"[診斷] Active adapter: {active}")
        enabled = lora_model.active_adapters  # property returns list
        print(f"[診斷] Enabled adapters: {enabled}")
        # Trainable params
        trainable, all_params = lora_model.get_nb_trainable_parameters()
        print(f"[診斷] 可訓練參數 (LoRA): {trainable:,} / 總參數: {all_params:,}")

        # Show at least one q_proj and v_proj LoRA A/B weight stats
        def print_lora_weight_stats(target_name):
            for name, param in lora_model.named_parameters():
                if target_name in name and ("lora_A" in name or "lora_B" in name):
                    print(f"  {name}: shape={list(param.shape)}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
                    return
            print(f"  未找到 {target_name} 的 LoRA 權重")
        print("[診斷] LoRA 權重樣本:")
        print_lora_weight_stats("q_proj")
        print_lora_weight_stats("v_proj")

        # ----- LoRA enabled forward -----
        lora_outputs_enabled = lora_model(**inputs)
        lora_logits_enabled = lora_outputs_enabled.logits
        lora_last_enabled = lora_logits_enabled[0, last_idx, :]

        print("\n=== LoRA enabled forward ===")
        print(f"LoRA enabled last-token logits – max: {lora_last_enabled.max().item():.6f}, min: {lora_last_enabled.min().item():.6f}")

        # ----- Disable adapter -----
        lora_model.disable_adapter()
        lora_outputs_disabled = lora_model(**inputs)
        lora_logits_disabled = lora_outputs_disabled.logits
        lora_last_disabled = lora_logits_disabled[0, last_idx, :]

        print("\n=== LoRA disabled forward ===")
        print(f"LoRA disabled last-token logits – max: {lora_last_disabled.max().item():.6f}, min: {lora_last_disabled.min().item():.6f}")

        # ----- Merge and unload -----
        merged_model = lora_model.merge_and_unload()
        merged_model.eval()
        merged_outputs = merged_model(**inputs)
        merged_logits = merged_outputs.logits
        merged_last = merged_logits[0, last_idx, :]

        print("\n=== Merged model forward ===")
        print(f"Merged last-token logits – max: {merged_last.max().item():.6f}, min: {merged_last.min().item():.6f}")

        # ----- Comparisons -----
        def compare(tag, t1, t2):
            diff = torch.abs(t1 - t2)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            num_diff = torch.sum(t1 != t2).item()
            total = t1.numel()
            print(f"\n{tag}")
            print(f"  最大絕對差異: {max_diff:.6f}")
            print(f"  平均絕對差異: {mean_diff:.6f}")
            print(f"  不同元素數量: {num_diff} / {total}")
            return max_diff, mean_diff, num_diff, total

        max_base_enabled, mean_base_enabled, _, _ = compare("Base vs LoRA enabled", base_last, lora_last_enabled)
        max_base_disabled, mean_base_disabled, _, _ = compare("Base vs LoRA disabled", base_last, lora_last_disabled)
        max_enabled_disabled, mean_enabled_disabled, _, _ = compare("LoRA enabled vs LoRA disabled", lora_last_enabled, lora_last_disabled)
        max_enabled_merged, mean_enabled_merged, _, _ = compare("LoRA enabled vs Merged", lora_last_enabled, merged_last)

        # ----- Diagnosis -----
        base_eq_disabled = approx_equal(base_last, lora_last_disabled)
        base_neq_enabled = not approx_equal(base_last, lora_last_enabled)
        enabled_eq_merged = approx_equal(lora_last_enabled, merged_last)

        print("\n=== 診斷結論 ===")
        if base_eq_disabled and base_neq_enabled and enabled_eq_merged:
            print("LoRA adapter 確實參與 forward")
        elif base_eq_disabled and approx_equal(base_last, lora_last_enabled):
            print("adapter 雖然載入，但目前 forward 沒有造成可觀察的 logits 差異")
        else:
            print("未符合上述明確情況，請參閱上面的比較結果進行判斷")

if __name__ == "__main__":
    main()