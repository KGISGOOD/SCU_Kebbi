#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FAQ → Conversational Dataset 前處理腳本
將 robot/faq.txt 轉換成適合 TRL SFTTrainer (assistant_only_loss) 的 Dataset。
不修改原始 faq.txt，不進行資料清理或增強。
"""

import os
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer  # 僅用來確認 tokenizer 可用，不在此腳本中套用 template

def main():
    # ====== 1. 讀取 faq.txt ======
    faq_path = os.path.join(os.path.dirname(__file__), "faq.txt")
    if not os.path.isfile(faq_path):
        raise FileNotFoundError(f"找不到檔案: {faq_path}")

    with open(faq_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip() != ""]
    print(f"讀取到 {len(lines)} 行原始資料")

    # ====== 2. 建立 conversational messages 列表 ======
    samples = []
    for idx, line in enumerate(lines, start=1):
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"第 {idx} 行格式錯誤（應為 question<TAB>answer）: {line}")
        question, answer = parts[0].strip(), parts[1].strip()
        messages = [
            {"role": "user",   "content": question},
            {"role": "assistant","content": answer}
        ]
        samples.append(messages)

    print(f"成功建立 {len(samples)} 筆訊息樣本")

    # ====== 3. 建立 Dataset ======
    dataset = Dataset.from_dict({"messages": samples})
    print(f"Dataset 建立完成，欄位: {dataset.column_names}")

    # ====== 4. 劃分 train / validation (90/10) ======
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    validation_dataset = split["test"]
    print(f"切分結果: train={len(train_dataset)} 筆, validation={len(validation_dataset)} 筆")

    # ====== 5. （可選）儲存到磁碟，方便之後直接載入 ======
    output_dir = os.path.join(os.path.dirname(__file__), "faq_dataset")
    dataset_dict = DatasetDict({"train": train_dataset, "validation": validation_dataset})
    dataset_dict.save_to_disk(output_dir)
    print(f"Dataset 已儲存至: {output_dir}")

    # ====== 6. 輸出最終 Dataset 欄位結構 ======
    print("\n=== 最終 Dataset 結構 ===")
    print("DatasetDict 包含兩個 split:")
    for split_name, ds in dataset_dict.items():
        print(f"  {split_name}:")
        print(f"    - 撇筆數: {len(ds)}")
        print(f"    - 欄位: {ds.column_names}")
        print(f"    - 欄位 'messages' 類型: {ds.features['messages']}")
        print(f"    - 第一筆範例 (前兩則訊息):")
        print(f"      {ds[0]['messages']}")

    # ====== 7. 額外驗證：重新載入並檢查是否仍為 conversational ======
    reloaded = DatasetDict.load_from_disk(output_dir)
    print("\n=== 重新載入後檢查 ===")
    for split_name, ds in reloaded.items():
        print(f"{split_name}:")
        print(f"  Columns: {ds.column_names}")
        print(f"  Features: {ds.features}")
        print(f"  First example: {ds[0]}")

if __name__ == "__main__":
    main()