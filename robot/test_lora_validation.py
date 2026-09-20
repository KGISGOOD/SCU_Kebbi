import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    # ----- 1. Paths -----
    base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    lora_adapter_path = os.path.join(os.path.dirname(__file__), "lora_output", "checkpoint-36")
    validation_dataset_path = os.path.join(os.path.dirname(__file__), "faq_dataset")
    # If you stored adapter elsewhere, adjust the path.

    # ----- 2. Load tokenizer -----
    print(f"Loading tokenizer from {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # ----- 3. Load base model -----
    print(f"Loading base model {base_model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    # ----- 4. Load LoRA adapted model -----
    print(f"Loading LoRA adapter from {lora_adapter_path}...")
    lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    lora_model.eval()

    # ----- 5. Load validation dataset -----
    print(f"Loading validation dataset from {validation_dataset_path}...")
    dataset_dict = load_from_disk(validation_dataset_path)
    validation_dataset = dataset_dict["validation"]
    print(f"Validation samples: {len(validation_dataset)}")

    # ----- 6. Generation settings -----
    gen_kwargs = {
        "max_new_tokens": 128,
        "do_sample": False,   # deterministic for comparison
        "temperature": 0.0,
        "top_p": 1.0,
        "pad_token_id": tokenizer.eos_token_id,
    }

    print("\n=== Validation Results ===\n")
    for idx, sample in enumerate(validation_dataset, start=1):
        # Sample structure: {"messages": [{"role":"user","content":question}, {"role":"assistant","content":answer}]}
        messages = sample["messages"]
        # Extract question and reference answer
        question = next((m["content"] for m in messages if m["role"] == "user"), "")
        reference_answer = next((m["content"] for m in messages if m["role"] == "assistant"), "")

        # Build prompt using Llama chat template (user only, with generation prompt)
        user_messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

        # Base model generation
        with torch.no_grad():
            base_output_ids = base_model.generate(**inputs, **gen_kwargs)
        input_length = inputs["input_ids"].shape[1]
        base_answer = tokenizer.decode(
            base_output_ids[0][input_length:],
            skip_special_tokens=True
        ).strip()

        # LoRA model generation
        with torch.no_grad():
            lora_output_ids = lora_model.generate(**inputs, **gen_kwargs)
        lora_answer = tokenizer.decode(
            lora_output_ids[0][input_length:],
            skip_special_tokens=True
        ).strip()

        print(f"[{idx:02d}] Question: {question}")
        print(f"    Reference answer: {reference_answer}")
        print(f"    Base model answer:  {base_answer}")
        print(f"    LoRA model answer:  {lora_answer}")
        print("-" * 80)

if __name__ == "__main__":
    main()