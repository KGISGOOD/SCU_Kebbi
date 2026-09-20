#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
診斷腳本：檢查 LoRA adapter 是否正確儲存、可載入且包含非零權重。
不進行訓練、不生成答案、不修改其他檔案。
"""

import os
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel


def main():
    # ----- 1. Paths -----
    base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    lora_adapter_path = os.path.join(os.path.dirname(__file__), "lora_output", "checkpoint-36")

    print(f"[診斷] 檢查 LoRA adapter 路徑: {lora_adapter_path}")
    if not os.path.isdir(lora_adapter_path):
        print(f"[錯誤] 目錄不存在: {lora_adapter_path}")
        return

    # ----- 2. List files in checkpoint-36 -----
    files = os.listdir(lora_adapter_path)
    print(f"[診斷] {lora_adapter_path} 中的檔案: {files}")

    # ----- 3. Check for adapter_config.json -----
    config_path = os.path.join(lora_adapter_path, "adapter_config.json")
    if os.path.isfile(config_path):
        print("[診斷] 找到 adapter_config.json")
    else:
        print("[錯誤] 未找到 adapter_config.json")

    # ----- 4. Check for LoRA adapter weight file -----
    weight_candidates = [
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_model.bin",
    ]
    weight_found = False
    for candidate in weight_candidates:
        weight_path = os.path.join(lora_adapter_path, candidate)
        if os.path.isfile(weight_path):
            print(f"[診斷] 找到 LoRA 權重檔案: {candidate}")
            weight_found = True
            break
    if not weight_found:
        print("[錯誤] 未找到 LoRA 權重檔案（嘗試了 adapter_model.safetensors、adapter_model.bin、pytorch_model.bin）")

    # ----- 5. Load Base Model -----
    print(f"[診斷] 載入基底模型: {base_model_name}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        base_model.eval()
        print("[診斷] 基底模型載入成功")
    except Exception as e:
        print(f"[錯誤] 載入基底模型失敗: {e}")
        return

    # ----- 6. Load LoRA adapter via PeftModel.from_pretrained -----
    print(f"[診斷] 載入 LoRA adapter 從: {lora_adapter_path}")
    try:
        lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        lora_model.eval()
        print("[診斷] LoRA adapter 載入成功")
    except Exception as e:
        print(f"[錯誤] 載入 LoRA adapter 失敗: {e}")
        return

    # ----- 7. Show LoRA adapter is loaded -----
    # PeftModel has a `peft_config` attribute and `peft_type`
    if hasattr(lora_model, "peft_config"):
        print(f"[診斷] LoRA 配置: {lora_model.peft_config}")
    else:
        print("[警告] 無法取得 peft_config")

    # ----- 8. Show LoRA parameter count -----
    try:
        trainable_params, all_param = lora_model.get_nb_trainable_parameters()
        print(f"[診斷] 可訓練參數 (LoRA): {trainable_params:,}")
        print(f"[診斷] 總參數: {all_param:,}")
        if trainable_params == 0:
            print("[錯誤] 可訓練參數為 0！LoRA 可能未正確掛載。")
    except Exception as e:
        print(f"[錯誤] 取得參數數量失敗: {e}")

    # ----- 9. Check LoRA weights for non-zero values -----
    print("[診斷] 檢查 LoRA 權重是否包含非零值...")
    try:
        # Iterate over named parameters and look for those in the peft modules
        non_zero_found = False
        for name, param in lora_model.named_parameters():
            if "lora" in name.lower():
                # Check if the tensor has any non-zero element
                if torch.any(param != 0):
                    non_zero_found = True
                    print(f"  [診斷] 找到非零權重: {name} (shape={param.shape}, 有非零值)")
                    # Optionally break early if we just want to know if any exist
                    # break
        if not non_zero_found:
            print("[錯誤] 未在任何 LoRA 權重中找到非零值！")
        else:
            print("[診斷] 確認 LoRA 權重包含非零值。")
    except Exception as e:
        print(f"[錯誤] 檢查 LoRA 權重時發生例外: {e}")

    print("[診斷] 完成。")


if __name__ == "__main__":
    main()