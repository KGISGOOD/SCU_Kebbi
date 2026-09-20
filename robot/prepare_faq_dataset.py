#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FAQ dataset preparation.

將 faq.txt 的 99 筆 Q&A
轉換成 Hugging Face conversational Dataset。

本腳本：
- 不修改 faq.txt
- 不套用 tokenizer
- 不套用 chat template
- 不做資料增強
- 不改寫問題或答案
"""

import os

from datasets import Dataset, DatasetDict


def main():
    # 1. 讀取 faq.txt
    faq_path = os.path.join(os.path.dirname(__file__), "faq.txt")

    if not os.path.isfile(faq_path):
        raise FileNotFoundError(f"找不到檔案: {faq_path}")

    with open(faq_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    print(f"讀取到 {len(lines)} 行原始資料")

    # 2. 解析 Q&A
    samples = []

    for idx, line in enumerate(lines, start=1):
        parts = line.split("\t", 1)

        if len(parts) != 2:
            raise ValueError(
                f"第 {idx} 行格式錯誤，應為 question<TAB>answer：{line}"
            )

        question = parts[0]
        answer = parts[1]

        samples.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                ]
            }
        )

    print(f"成功建立 {len(samples)} 筆 conversational samples")

    # 3. 建立 Dataset
    dataset = Dataset.from_list(samples)

    print(f"Dataset 欄位: {dataset.column_names}")

    # 4. 切分 train / validation
    split = dataset.train_test_split(
        test_size=10,
        seed=42,
    )

    train_dataset = split["train"]
    validation_dataset = split["test"]

    print(
        f"切分結果: train={len(train_dataset)}, "
        f"validation={len(validation_dataset)}"
    )

    # 5. 建立 DatasetDict
    dataset_dict = DatasetDict(
        {
            "train": train_dataset,
            "validation": validation_dataset,
        }
    )

    # 6. 儲存
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "faq_dataset",
    )

    dataset_dict.save_to_disk(output_dir)

    print(f"Dataset 已儲存至: {output_dir}")

    # 7. 最終驗證
    print("\n=== 最終 Dataset 結構 ===")

    print(dataset_dict)

    for split_name, ds in dataset_dict.items():
        print(f"\n[{split_name}]")
        print(f"筆數: {len(ds)}")
        print(f"欄位: {ds.column_names}")
        print(f"第一筆: {ds[0]}")

    print("\n=== 驗證完成 ===")
    print("資料仍維持 conversational messages 格式。")


if __name__ == "__main__":
    main()