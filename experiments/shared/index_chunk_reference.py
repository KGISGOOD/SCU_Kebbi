"""Preserved historical chunk reference used by Dense Index integrity checks.

Extracted unchanged from the former A4 builder; no BM25 runtime dependency.
"""
from typing import List
from langchain_core.documents import Document


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 128,
) -> List[Document]:
    """Split documents into chunks while preserving metadata.

    Same function as used in A0 for fair comparison.

    Args:
        documents: List of Document objects.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between chunks in characters.

    Returns:
        List of chunked Document objects.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )

    chunked_docs = []
    for doc in documents:
        # Split the document
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            # Create metadata for the chunk
            chunk_metadata = doc.metadata.copy()
            chunk_metadata["chunk_id"] = i
            chunk_metadata["parent_document_id"] = doc.metadata.get(
                "document_id", f"doc_{len(chunked_docs)}"
            )
            # Update document_id to be unique for this chunk
            chunk_metadata["document_id"] = (
                f"{chunk_metadata.get('document_id', 'doc')}_chunk_{i}"
            )

            # Create chunk document
            chunked_doc = Document(page_content=chunk, metadata=chunk_metadata)
            chunked_docs.append(chunked_doc)

    return chunked_docs
