"""Prompt templates for the RAG baseline.

Provides functions to build prompts from retrieved context and questions.
"""

from __future__ import annotations

from typing import List
from langchain_core.documents import Document

from .config import settings


def build_prompt(context: str, question: str) -> str:
    """Build a prompt from context and question.

    Args:
        context: Retrieved context string.
        question: User question.

    Returns:
        Formatted prompt string.
    """
    return settings.prompt_template.format(context=context, question=question)


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