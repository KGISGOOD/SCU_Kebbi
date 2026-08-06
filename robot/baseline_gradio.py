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

def main():
    # Initialize components as in main.py
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

    def respond(message: str, chat_history):
        """
        Gradio callback: receives user message and current chat history (list of dicts with role/content),
        returns updated chat history.
        """
        try:
            # Convert chat_history (messages format) to list of (user, assistant) tuples
            history_tuples = []
            i = 0
            n = len(chat_history)
            while i + 1 < n:
                user_msg = chat_history[i]
                assoc_msg = chat_history[i + 1]
                if user_msg.get("role") == "user" and assoc_msg.get("role") == "assistant":
                    history_tuples.append((user_msg["content"], assoc_msg["content"]))
                    i += 2
                else:
                    # Unexpected format, skip
                    i += 1
            # Call service
            response = service.handle(message, history_tuples)
            # Append to chat_history as messages
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": response})
            return "", chat_history
        except Exception as e:
            err_msg = f"發生�錯�誤: {e}"
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": err_msg})
            return "", chat_history

    with gr.Blocks(title="Podcast LLM Baseline (Gradio)") as demo:
        gr.Markdown("# Podcast LLM Baseline – Interactive Query")
        chatbot = gr.Chatbot(label="對話紀錄", height=400)
        with gr.Row():
            txt = gr.Textbox(label="�請�輸入問題", placeholder="�輸入您的問題後按 Enter", lines=1)
            submit_btn = gr.Button("送出")
        # Bind events
        txt.submit(respond, [txt, chatbot], [txt, chatbot])
        submit_btn.click(respond, [txt, chatbot], [txt, chatbot])

    demo.queue()  # enable queuing for better handling
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=False,
        prevent_thread_lock=False
    )

if __name__ == "__main__":
    main()