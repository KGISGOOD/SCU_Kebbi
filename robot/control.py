#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-script solution for length-controlled QA generation.
Defines helper functions locally and monkey-patches the original modules.
No original files are modified.
"""

import sys
import os
import re

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# === Length control helper functions (equivalent to control.py) ===
def enhance_prompt_for_length_control(base_prompt: str, target_length: int) -> str:
    """
    Enhance a base prompt with length control instructions.
    """
    length_control_instruction = f"""

**精確長度控制指令**：
請在產出的每個中文字元之後（包括漢字、標點符號和空格），立即輸出一個遞減的數值標記，
從 {target_length} 開始遞減至 0。例如目標長度為 {target_length}：
第一個字元後跟『{target_length}』、第二個字元後跟『{target_length-1}』…最後一個字元後必須跟『0』。
移除所有數值標記後，剩餘純文字內容必須恰好為 {target_length} 個中文字元。

"""
    if base_prompt.endswith("：\n") or base_prompt.endswith(":\n") or base_prompt.endswith("\n\n"):
        return base_prompt + length_control_instruction
    else:
        return base_prompt + "\n" + length_control_instruction


def strip_countdown_markers(text: str) -> str:
    """
    Strip countdown markers from LLM output.
    """
    result = re.sub(r'(.)\d+', r'\1', text)          # half-width digits
    result = re.sub(r'(.)[０-９]+', r'\1', result)   # full-width digits
    return result


def get_length_without_markers(text: str) -> int:
    """Calculate length after removing markers."""
    return len(strip_countdown_markers(text))


# === Import original modules ===
try:
    from prompts import PromptFactory
    from orchestrator import QAOrchestrator
    from service import ChatService
    from config import AppSettings
    from embeddings import HFEmbeddingsProvider
    from vectorstores import VectorStoreLoader
    from retriever import MultiStoreRetriever
    from llm.ollama import ChatOllamaLLM
    from relevance import RelevancePolicy
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the robot directory")
    sys.exit(1)


# === Monkey-patching ===
# Store original methods
original_context_prompt = PromptFactory.context_prompt
original_ask = QAOrchestrator.ask

# Patch PromptFactory.context_prompt to add length control
def patched_context_prompt():
    # Get original template
    base_template = original_context_prompt().template
    # Enhance it with length control (target length configurable)
    enhanced_template = enhance_prompt_for_length_control(base_template, TARGET_LENGTH)
    from langchain.prompts import ChatPromptTemplate
    return ChatPromptTemplate.from_template(enhanced_template)

PromptFactory.context_prompt = patched_context_prompt

# Patch QAOrchestrator.ask to strip countdown markers
def patched_ask(self, question: str, history=None):
    if history is None:
        history = []
    raw_response = original_ask(self, question, history)
    clean_response = strip_countdown_markers(raw_response)
    return clean_response

QAOrchestrator.ask = patched_ask


# === Configuration ===
TARGET_LENGTH = 14  # Target length in Chinese characters (adjust 8-20 as needed)


def main():
    print(f"=== Length-Controlled QA System (target: {TARGET_LENGTH} Chinese characters) ===\n")

    # Initialize components (same as baseline_gradio.py)
    settings = AppSettings()
    emb = HFEmbeddingsProvider(settings.model_name, settings.use_cpu)
    loader = VectorStoreLoader(emb)
    stores = loader.load_all_from_dir(settings.parent_vector_dir)
    retriever = MultiStoreRetriever(stores, top_k=settings.top_k, fetch_k=settings.fetch_k)
    llm = ChatOllamaLLM(
        model_name=settings.ollama_model,
        url=settings.ollama_url,
        stream=settings.ollama_stream,
        timeout_sec=settings.ollama_timeout_sec,
    )
    prompts = PromptFactory()  # Uses patched version
    orch = QAOrchestrator(retriever, llm, prompts)
    relevance = RelevancePolicy()
    service = ChatService(orch, relevance, retrieve_only=False)

    print("System initialized!\n")

    # Demo with predefined questions
    demo_questions = [
        "什麼是人工智慧？",
        "今天天氣如何？",
        "系上有哪些研究所？",
        "機器人學課程什麼時候開？"
    ]

    print("--- Demo Questions ---")
    for i, q in enumerate(demo_questions, 1):
        print(f"\nQ{i}: {q}")
        try:
            response = service.handle(q, [])
            clean_len = get_length_without_markers(response)
            print(f"Raw response length: {len(response)}")
            print(f"Clean response: {response}")
            print(f"Clean length: {clean_len} characters (target: {TARGET_LENGTH})")
            if abs(clean_len - TARGET_LENGTH) <= 2:
                print("✓ Length within target range")
            else:
                print("✗ Length outside target range")
        except Exception as e:
            print(f"Error: {e}")

    # Interactive mode
    print("\n" + "="*50)
    print("Enter interactive mode (type 'quit' to exit)")
    while True:
        try:
            user_input = input("\n請輸入問題: ").strip()
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("再見！")
                break
            if not user_input:
                continue

            response = service.handle(user_input, [])
            clean_len = get_length_without_markers(response)
            print(f"\n回答: {response}")
            print(f"長度: {clean_len} 個字元")

        except KeyboardInterrupt:
            print("\n再見！")
            break
        except Exception as e:
            print(f"錯誤: {e}")


if __name__ == "__main__":
    main()