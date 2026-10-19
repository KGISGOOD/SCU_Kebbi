"""Prompt templates for the RAG baseline.

Provides functions to build prompts from retrieved context and questions.
"""

from __future__ import annotations

from typing import List
from langchain_core.documents import Document

from .config import settings


# Prompt version tracking
PROMPT_VERSION = "baseline_basic_v1"


def build_prompt(context: str, question: str) -> str:
    """Build a prompt from context and question using the fixed baseline prompt.

    Args:
        context: Retrieved context string.
        question: User question.

    Returns:
        Formatted prompt string following the baseline basic template.
    """
    system_prompt = """你是「資科萬事通」，專門回答東吳大學巨量資料管理學院資料科學系的相關問題。
請主要根據提供的參考資料回答，使用繁體中文，內容清楚、簡潔。
如果參考資料不足以確認答案，請明確說明目前資料不足，不要自行猜測或編造。
涉及日期、招生、學分或規定時，不要把不同年度的資訊混為一談；無法確認目前最新資訊時，應說明無法確認。
如果問題與東吳資料科學系完全無關，請簡短說明本系統主要提供系所相關資訊。
測試參考日期為 2026 年 10 月 19 日；這是固定的離線測試基準，不代表資料一定是最新的。"""

    user_prompt = f"""參考資料：

{context}

問題：

{question}

請回答："""

    # Combine system and user prompts in a format suitable for Llama 3.2 Instruct
    # Using the standard chat format: <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    # {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    # {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return prompt


def format_documents(documents: List[Document]) -> str:
    """Format a list of documents into a context string.

    Args:
        documents: List of retrieved documents.

    Returns:
        Formatted context string.
    """
    if not documents:
        return "無相關資訊"

    # Sort by metadata if available (e.g., by date or source)
    # For simplicity, we'll just concatenate with clear separators
    formatted_parts = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source_file", "未知來源")
        content_type = doc.metadata.get("content_type", "未知類型")
        # Include helpful metadata
        header = f"[資料 {i}] 來源：{source} | 類型：{content_type}"
        formatted_parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted_parts)


def create_context_from_documents(documents: List[Document]) -> str:
    """Create context string from retrieved documents.

    Args:
        documents: List of retrieved documents.

    Returns:
        Context string for the prompt.
    """
    return format_documents(documents)


if __name__ == "__main__":
    # Simple test
    from langchain_core.documents import Document

    test_docs = [
        Document(
            page_content="這是測試內容1",
            metadata={"source_file": "test1.txt", "content_type": "測試"},
        ),
        Document(
            page_content="這是測試內容2",
            metadata={"source_file": "test2.txt", "content_type": "測試"},
        ),
    ]
    context = create_context_from_documents(test_docs)
    print("Formatted context:")
    print(context)
    prompt = build_prompt(context, "這是測試問題？")
    print("\nFull prompt:")
    print(prompt)