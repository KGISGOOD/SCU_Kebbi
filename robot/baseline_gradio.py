#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline QA system with Gradio UI.
Mirrors the logic of run_speed_test.py but provides an interactive chat interface.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from service import ChatService
from config import AppSettings
from embeddings import HFEmbeddingsProvider
from vectorstores import VectorStoreLoader
from retriever import MultiStoreRetriever
from llm.ollama import ChatOllamaLLM
from prompts import PromptFactory
from orchestrator import QAOrchestrator
from relevance import RelevancePolicy


def initialize_components():
    """Set up the same pipeline as run_speed_test.py."""
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
    prompts = PromptFactory()
    orch = QAOrchestrator(retriever, llm, prompts)
    relevance = RelevancePolicy()
    service = ChatService(orch, relevance, retrieve_only=False)
    return service, orch


def respond(message, chat_history):
    """
    Gradio callback: receives user message and current chat history,
    returns updated chat history.
    """
    # Get shared components (initialized once via outer scope)
    service, orch = respond._globals  # type: ignore
    try:
        # Retrieve context (for source display)
        ctx, uniq_sources, vec_time = orch.retrieve_only(message)
        # Generate answer
        ans, llm_time = orch.ask(message, [])
        total_time = vec_time + llm_time

        # Build source string
        src_lines = []
        for i, (ep, pod) in enumerate(uniq_sources, 1):
            src_lines.append(f"Result {i}: {ep}, {pod}")
        src_block = "\n可參考下方節目集數：\n" + "\n".join(src_lines) if src_lines else ""

        # Compose final answer with timing
        answer_text = (
            f"{ans.strip()}\n\n"
            f"{src_block}\n"
            f"---\n向量檢索時間: {vec_time:.2f} 秒\nLLM 生成時間: {llm_time:.2f} 秒\n"
            f"總延遲: {total_time:.2f} 秒"
        )
        # Append to chat history
        chat_history.append((message, answer_text))
        return "", chat_history
    except Exception as e:
        err_msg = f"發生錯誤: {e}"
        chat_history.append((message, err_msg))
        return "", chat_history


def main():
    # Initialise shared components and store them in function attribute for reuse
    service, orch = initialize_components()
    respond._globals = (service, orch)  # type: ignore

    with gr.Blocks(title="Podcast LLM Baseline (Gradio)") as demo:
        gr.Markdown("# Podcast LLM Baseline – Interactive Query")
        chatbot = gr.Chatbot(label="對話紀錄", height=400)
        with gr.Row():
            txt = gr.Textbox(label="請輸入問題", placeholder="輸入您的問題後按 Enter", lines=1)
            submit_btn = gr.Button("送出")
        # Bind events
        txt.submit(respond, [txt, chatbot], [txt, chatbot])
        submit_btn.click(respond, [txt, chatbot], [txt, chatbot])

    demo.queue()  # enable queuing for better handling
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()