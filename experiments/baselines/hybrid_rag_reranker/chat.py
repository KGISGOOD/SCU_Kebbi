"""Interactive chat for Hybrid RAG + Reranker (A4).

Provides a command-line interface for chatting with the Hybrid RAG + Reranker system.
"""

from __future__ import annotations

import sys
from typing import List, Dict, Any

from .config import settings
from .pipeline import create_pipeline_from_config
from experiments.shared.data_loader import load_documents_from_type1_type2
from experiments.shared.build_bm25_index import chunk_documents
from experiments.shared.evaluation import timer


def main():
    """Main function for interactive chat."""
    print("=" * 60)
    print("資科萬事通 - Hybrid RAG + Reranker (A4) Interactive Chat")
    print("=" * 60)
    print("選擇模式:")
    print("  1. A4-1: BM25 + Dense + RRF → LLM (無 reranker)")
    print("  2. A4-2: BM25 + Dense + RRF + Reranker → LLM (有 reranker)")
    print("  輸入 'quit' 或 'exit' 結束對話")
    print("=" * 60)

    # Load and chunk documents (same as A0 for fair comparison)
    print("\n正在載入和處理文件...")
    with timer("文件載入和處理"):
        raw_documents, stats = load_documents_from_type1_type2(
            settings.type1_dir, settings.type2_dir
        )
        documents = chunk_documents(
            raw_documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    print(f"   載入了 {stats['total_documents']} 份原始文件")
    print(f"   建立了 {len(documents)} 個文件區塊")

    # Get user choice for mode
    while True:
        choice = input("\n請選擇模式 (1 或 2): ").strip()
        if choice in ["1", "2"]:
            use_reranker = (choice == "2")
            break
        elif choice.lower() in ["quit", "exit"]:
            print("再見！")
            return 0
        else:
            print("無效選擇，請輸入 1 或 2")

    mode_name = "A4-2 (BM25 + Dense + RRF + Reranker → LLM)" if use_reranker else "A4-1 (BM25 + Dense + RRF → LLM)"
    print(f"\n已選擇: {mode_name}")

    # Initialize pipeline
    print("\n正在初始化系統...")
    try:
        with timer("系統初始化"):
            pipeline = create_pipeline_from_config(
                use_reranker=use_reranker,
                documents=documents
            )
        doc_count = pipeline.vector_store.get_document_count()
        if doc_count == 0:
            print("   錯誤：向量存儲為空。請先運行 build_index.py 建立索引。")
            return 1
        print(f"   向量存儲中有 {doc_count} 個文件區塊")
    except Exception as e:
        print(f"   錯誤：初始化系統失敗: {e}")
        return 1

    print("\n系統已就緒！開始對話吧！（輸入 'quit' 或 'exit' 結束）")
    print("-" * 60)

    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("\n您的問題: ").strip()

            # Check for exit commands
            if user_input.lower() in ["quit", "exit"]:
                print("再見！")
                break

            # Skip empty input
            if not user_input:
                continue

            # Process query
            print("\n正在思考中...")
            try:
                with timer("查詢處理"):
                    answer, _, _, info = pipeline(
                        user_input,
                        return_context=True
                    )

                # Print answer
                print(f"\n回答: {answer}")

                # Show sources if available
                if info["final_docs"]:
                    print(f"\n參考來源 ({len(info['final_docs'])} 個文件):")
                    for j, (doc, score) in enumerate(zip(info["final_docs"], info["final_scores"]), start=1):
                        source = doc.metadata.get("source_file", "未知來源")
                        content_type = doc.metadata.get("content_type", "未知類型")
                        print(f"  {j}. [{source} | {content_type}] (相關度分數: {score:.3f})")

            except Exception as e:
                print(f"\n處理查詢時發生錯誤: {e}")
                import traceback
                traceback.print_exc()

        except KeyboardInterrupt:
            print("\n\n再見！")
            break
        except EOFError:
            print("\n\n再見！")
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())