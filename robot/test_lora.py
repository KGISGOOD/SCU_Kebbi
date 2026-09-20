#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    lora_adapter_path = os.path.join(
    os.path.dirname(__file__),
    "lora_output",
    "checkpoint-36"
)

    test_questions = [
        "學長，東吳大學是一所什麼樣的大學啊？",
        "面對現在的 AI 浪潮，東吳大學有什麼特別的轉變嗎？",
        "為什麼學校要特別成立「巨量資料管理學院（巨資學院）」呢？",
        "巨資學院最核心的學術基礎是什麼？",
        "念巨資學院，是不是只要學會寫程式就夠了？",
        "如果我對大數據很有興趣，進來巨資學院能學到核心技術嗎？",
        "一句話總結，東吳巨資學院到底在做什麼？",
        "學長，和其他資管或資工系相比，東吳巨資學院最大的特色是什麼？",
        "巨資學院跟一般的「資工系」有什麼不同？",
        "那巨資學院跟一般的「資管系」差別在哪裡？"
    ]

    print(f"Loading tokenizer from {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )

    print(f"Loading base model {base_model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    print(f"Loading LoRA adapter from {lora_adapter_path}...")
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path
    )
    lora_model.eval()

    gen_kwargs = {
        "max_new_tokens": 128,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }

    print("\n=== Comparison Results ===\n")

    for idx, question in enumerate(test_questions, start=1):

        # 使用 Llama 3.2 原生 Chat Template
        messages = [
            {"role": "user", "content": question}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(base_model.device)

        input_length = inputs["input_ids"].shape[1]

        # Base model
        with torch.no_grad():
            base_output_ids = base_model.generate(
                **inputs,
                **gen_kwargs
            )

        base_answer = tokenizer.decode(
            base_output_ids[0][input_length:],
            skip_special_tokens=True
        ).strip()

        # LoRA model
        with torch.no_grad():
            lora_output_ids = lora_model.generate(
                **inputs,
                **gen_kwargs
            )

        lora_answer = tokenizer.decode(
            lora_output_ids[0][input_length:],
            skip_special_tokens=True
        ).strip()

        print(f"[{idx:02d}] Question: {question}")
        print(f"    Base model answer: {base_answer}")
        print(f"    LoRA model answer: {lora_answer}")
        print("-" * 80)


if __name__ == "__main__":
    main()