#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
標準 LoRA SFT 訓練腳本 (Llama 3.2 3B Instruct)
- 基礎模型 : meta-llama/Llama-3.2-3B-Instruct
- 數據來源 : faq_dataset/ (conversational `messages` 欄位)
- 只做 LoRA + SFT，不加入其他演算法
- 支援 --dry-run，僅初始化 Trainer，不開始實際訓練
"""

import os
import argparse
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def main():
    # ------------------- 參數解析 -------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='僅初始化 Trainer，不開始實際訓練')
    args = parser.parse_args()

    # ------------------- 基本資訊印出 -------------------
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"[Info] Model name          : {model_name}")

    # ------------------- 載入 tokenizer 與模型 -------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"[Info] GPU name            : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ------------------- 載入已製作的 Dataset -------------------
    dataset_path = os.path.join(os.path.dirname(__file__), "faq_dataset")
    dataset_dict = load_from_disk(dataset_path)   # DatasetDict with train & validation
    print(f"[Info] Train samples       : {len(dataset_dict['train'])}")
    print(f"[Info] Validation samples  : {len(dataset_dict['validation'])}")

    # Debug: print dataset structure (optional)
    print("\n=== Dataset 結構偵測 ===")
    for split_name, ds in dataset_dict.items():
        print(f"{split_name}:")
        print(f"  Columns: {ds.column_names}")
        print(f"  Features: {ds.features}")
        # 確認是否為 conversational
        if "messages" in ds.column_names:
            print(f"  Messages field type: {ds.features['messages']}")
            first_msg = ds[0]["messages"]
            print(f"  First example messages: {first_msg}")

    # ------------------- LoRA 設定 -------------------
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    target_modules = ["q_proj", "v_proj"]   # 常見的 Llama 目標模組
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    print(f"[Info] LoRA r              : {lora_r}")
    print(f"[Info] LoRA alpha          : {lora_alpha}")
    print(f"[Info] LoRA dropout        : {lora_dropout}")
    print(f"[Info] LoRA target_modules : {target_modules}")

    # ------------------- SFT 訓練參數 -------------------
    output_dir = os.path.join(os.path.dirname(__file__), "lora_output")
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_length=512,
        fp16=True,
        # evaluation and saving per epoch
        eval_strategy="epoch",
        save_strategy="epoch",
        # 使用原生 chat template (SFTTrainer 會內部處理)
        # 不設定 assistant_only_loss，保持標準 loss
    )
    print(f"[Info] Epochs              : {training_args.num_train_epochs}")
    print(f"[Info] Batch size          : {training_args.per_device_train_batch_size}")
    print(f"[Info] Gradient accum steps: {training_args.gradient_accumulation_steps}")
    print(f"[Info] Learning rate       : {training_args.learning_rate}")
    print(f"[Info] Max length          : {training_args.max_length}")

    # ------------------- 建立 SFTTrainer -------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        processing_class=tokenizer,   # 新 API 使用 processing_class
        peft_config=lora_cfg,         # 讓 Trainer 內部套用 LoRA
    )

    # 訓練前統計可訓練參數
    trainable_params, all_param = trainer.model.get_nb_trainable_parameters()
    print(f"[Info] Trainable params    : {trainable_params:,}")
    print(f"[Info] All params          : {all_param:,}")

    # ------------------- Dry-run 檢查 -------------------
    if args.dry_run:
        print("[Dry Run] Trainer initialization successful.")
        print("[Dry Run] Training was NOT started.")
        return

    # ------------------- 開始訓練 -------------------
    trainer.train()

    # ------------------- 訓練完成後輸出 -------------------
    print(f"[Info] LoRA adapter saved to : {training_args.output_dir}")
    print("[Done] 訓練完成")


if __name__ == "__main__":
    main()