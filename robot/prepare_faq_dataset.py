#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FAQ → SFT Dataset 前處理腳本
將 robot/faq.txt 轉換成適合 TRL SFTTrainer 的 Dataset。
不修改原始 faq.txt，不進行資料清理或增強。
"""

import os
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

def main():
    # ====== 1. 讀取 faq.txt ======
    faq_path = os.path.join(os.path.dirname(__file__), "faq.txt")
    if not os.path.isfile(faq_path):
        raise FileNotFoundError(f"找不到檔案: {faq_path}")

    with open(faq_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip() != ""]
    print(f"讀取到 {len(lines)} 行原始資料")

    # ====== 2. 驗證格式 & 準備訊息 ======
    samples = []
    for idx, line in enumerate(lines, start=1):
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"第 {idx} 行格式錯誤（應為 question<TAB>answer）: {line}")
        question, answer = parts[0].strip(), parts[1].strip()
        # 建立 chat template 訊息（僅 user + assistant）
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        samples.append(messages)

    print(f"成功建立 {len(samples)} 筆訊息樣本")

    # ====== 3. 載入 tokenizer 並套用 chat template ======
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"載入 tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    texts = []
    for msg in samples:
        # apply_chat_template 回傳完整的聊天字串（不加入 generation prompt）
        formatted = tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(formatted)

    print(f"產生 {len(texts)} 筆文字樣本（範例第一筆）:")
    print(texts[0][:200] + ("..." if len(texts[0]) > 200 else ""))

    # ====== 4. 建立 Dataset ======
    dataset = Dataset.from_dict({"text": texts})
    print(f"Dataset 建立完成，欄位: {dataset.column_names}")

    # ====== 5. 劃分 train / validation (90/9) ======
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    validation_dataset = split["test"]
    print(f"切分結果: train={len(train_dataset)} 筆, validation={len(validation_dataset)} 筆")

    # ====== 6. （可選）儲存到磁碟，方便之後直接載入 ======
    output_dir = os.path.join(os.path.dirname(__file__), "faq_dataset")
    dataset_dict = DatasetDict({"train": train_dataset, "validation": validation_dataset})
    dataset_dict.save_to_disk(output_dir)
    print(f"Dataset 已儲存至: {output_dir}")

    # ====== 7. 輸出最終 Dataset 欄位結構 ======
    print("\n=== 最終 Dataset 結構 ===")
    print("DatasetDict 包含兩個 split:")
    for split_name, ds in dataset_dict.items():
        print(f"  {split_name}:")
        print(f"    - 撇筆數: {len(ds)}")
        print(f"    - 欄位: {ds.column_names}")
        print(f"    - 欄位 'text' 類型: {ds.features['text']}")
        print(f"    - 第一筆範例 (前 150 個字元):")
        print(f"      {ds[0]['text'][:150]}{'...' if len(ds[0]['text']) > 150 else ''}")

if __name__ == "__main__":
    main()