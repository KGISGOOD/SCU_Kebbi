"""Interactive chat interface for the RAG baseline.

Provides a simple command-line interface for asking questions
and receiving answers from the RAG pipeline.
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import time
from pathlib import Path


def configured_chat(config_path):
    """Single-turn Baseline 1, without importing an HTTP server."""
    from experiments.shared.a0_adapter import A0Adapter, measured
    root = Path(__file__).resolve().parents[3]
    path = Path(config_path).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if config.get("prompt_version") != "baseline_senior_v2" or "faq" in config:
        raise ValueError("this CLI requires Baseline 1 with baseline_senior_v2, without FAQ")
    print("載入 Baseline 1 模型與既有索引，請稍候……")
    adapter = A0Adapter(root, config)
    if adapter.identity["prompt_version"] != "baseline_senior_v2" or adapter.pipeline.prompt_version != "baseline_senior_v2":
        raise ValueError("loaded Prompt does not match Baseline 1")
    print("Prompt: baseline_senior_v2；單輪問答，輸入 exit 結束。")
    while True:
        try:
            question = input("您的問題: ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                break
            if not question:
                continue
            try:
                (answer, detail), elapsed = measured(adapter.answer, question)
                print("回答:", answer)
                print("字數（含標點及空白）: %d；超過100字: %s；耗時: %.3f 秒" %
                      (len(answer), "是" if len(answer) > 100 else "否", elapsed))
                print("Prompt:", adapter.pipeline.prompt_version)
                for doc in detail["documents"]:
                    print("來源:", doc["metadata"].get("source_file", doc["metadata"]),
                          "相似度:", doc["score"])
            except Exception as exc:
                print("問答失敗:", exc)
        except (KeyboardInterrupt, EOFError):
            break
    print("再見！")


def main():
    """Main function to run the interactive chat."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", help="Baseline 1 JSON; use a0_baseline_v2.local.json on GPU")
    args = parser.parse_args()
    if args.config:
        return configured_chat(args.config)
    from .config import settings
    from .pipeline import create_pipeline_from_config
    print("=" * 60)
    print("RAG Baseline Interactive Chat")
    print("=" * 60)
    print("Type your questions in Traditional Chinese.")
    print("Commands:")
    print("  exit, quit, q - Exit the chat")
    print("  help, h - Show this help message")
    print("  stats - Show pipeline statistics")
    print("=" * 60)

    # Initialize pipeline
    print("\nInitializing pipeline...")
    try:
        pipeline = create_pipeline_from_config()
        doc_count = pipeline.vector_store.get_document_count()
        print(f"Pipeline ready! Vector store contains {doc_count} documents.")
        if doc_count == 0:
            print(
                "WARNING: Vector store is empty. Please run build_index.py first."
            )
            print("The chat will still work but will not retrieve any documents.")
    except Exception as e:
        print(f"ERROR: Failed to initialize pipeline: {e}")
        print("Please check that:")
        print("  1. The vector index has been built (run build_index.py)")
        print("  2. The LLM model is accessible")
        print("  3. All dependencies are installed")
        sys.exit(1)

    print("\nChat started! Ask your questions below.\n")

    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("您的問題: ").strip()

            # Check for exit commands
            if user_input.lower() in ("exit", "quit", "q"):
                print("\n再見！")
                break

            # Check for help
            if user_input.lower() in ("help", "h"):
                print("\nCommands:")
                print("  exit, quit, q - Exit the chat")
                print("  help, h - Show this help message")
                print("  stats - Show pipeline statistics")
                print("  (Ask any question in Traditional Chinese)\n")
                continue

            # Check for stats
            if user_input.lower() == "stats":
                print(f"\nPipeline Statistics:")
                print(f"  Vector store documents: {pipeline.vector_store.get_document_count()}")
                print(f"  Embedding model: {settings.embedding_model}")
                print(f"  LLM model: {settings.llm_model_id}")
                print(f"  Top-K retrieval: {settings.top_k}")
                print(f"  Chunk size: {settings.chunk_size}")
                print(f"  Chunk overlap: {settings.chunk_overlap}\n")
                continue

            # Skip empty input
            if not user_input:
                continue

            # Process the question
            print("\n思考中...")
            try:
                answer, retrieved_docs, scores = pipeline(
                    user_input, return_context=settings.show_sources
                )

                # Print answer
                print(f"\n回答: {answer}")

                # Show sources if enabled and available
                if settings.show_sources and retrieved_docs:
                    print(f"\n參考來源 ({len(retrieved_docs)} 筆):")
                    for i, (doc, score) in enumerate(
                        zip(retrieved_docs, scores), start=1
                    ):
                        source = doc.metadata.get("source_file", "未知來源")
                        content_type = doc.metadata.get("content_type", "未知類型")
                        print(
                            f"  {i}. [{source}] {content_type} (相關度: {score:.3f})"
                        )
                print()  # Empty line for readability

            except Exception as e:
                print(f"\n錯誤: {e}\n")

        except KeyboardInterrupt:
            print("\n\n收到中斷 signal，正在退出...")
            break
        except EOFError:
            print("\n\n再見！")
            break

    print("Chat ended.")


if __name__ == "__main__":
    main()
