"""Build the BM25 index preparation for Hybrid RAG + Reranker (A4).

This script loads data from type1 and type2 directories (same as A0),
prepares the documents for BM25 retrieval, and saves them with version information.
Note: BM25 doesn't require building an index like dense vectors - we just need
to prepare and save the documents with metadata for consistent retrieval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .config import settings
from experiments.shared.data_loader import load_documents_from_type1_type2
from experiments.shared.evaluation import timer
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


def save_bm25_index(
    documents: List[Document],
    index_dir: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    source_directories: List[str],
) -> None:
    """Save BM25 index information (documents and metadata).

    Args:
        documents: List of Document objects to save
        index_dir: Directory to save the BM25 index
        embedding_model: Embedding model used (for version tracking)
        chunk_size: Chunk size used
        chunk_overlap: Chunk overlap used
        source_directories: Source directories used
    """
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    # Save documents as JSONL (for BM25 retrieval)
    documents_file = index_path / "documents.jsonl"
    with open(documents_file, "w", encoding="utf-8") as f:
        for doc in documents:
            # Convert Document to dict for JSON serialization
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            f.write(json.dumps(doc_dict, ensure_ascii=False) + "\n")

    # Save version information
    version_file = index_path / "version.json"
    version_data = {
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "source_directories": source_directories,
        "document_count": len(set(d.metadata.get("document_id", "") for d in documents)),
        "chunk_count": len(documents),
        "data_fingerprint": _compute_data_fingerprint(documents),
        "index_type": "BM25",
    }
    with open(version_file, "w", encoding="utf-8") as f:
        json.dump(version_data, f, indent=2, ensure_ascii=False)


def _compute_data_fingerprint(documents: List[Document]) -> str:
    """Compute a fingerprint of the documents for version validation."""
    import hashlib

    # Sort by document_id for consistent fingerprinting
    sorted_docs = sorted(documents, key=lambda d: d.metadata.get("document_id", ""))
    # Concatenate document_id and page_content
    content_to_hash = ""
    for doc in sorted_docs:
        doc_id = doc.metadata.get("document_id", "")
        content_to_hash += f"{doc_id}:{doc.page_content}\n"
    return hashlib.md5(content_to_hash.encode("utf-8")).hexdigest()


def load_bm25_index(index_dir: str) -> List[Document]:
    """Load BM25 index (documents) from disk.

    Args:
        index_dir: Directory containing the BM25 index

    Returns:
        List of Document objects
    """
    index_path = Path(index_dir)
    documents_file = index_path / "documents.jsonl"

    if not documents_file.exists():
        return []

    documents = []
    with open(documents_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc_dict = json.loads(line)
                doc = Document(
                    page_content=doc_dict["page_content"],
                    metadata=doc_dict["metadata"]
                )
                documents.append(doc)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to parse document line: {e}")
                continue

    return documents


def is_bm25_index_valid(
    index_dir: str,
    documents: List[Document],
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    source_directories: List[str],
) -> bool:
    """Check if the existing BM25 index is valid for the given parameters and data.

    Args:
        index_dir: Directory containing the BM25 index
        documents: List of Document objects to validate against
        embedding_model: Embedding model used
        chunk_size: Chunk size used
        chunk_overlap: Chunk overlap used
        source_directories: Source directories used

    Returns:
        True if index is valid, False otherwise
    """
    index_path = Path(index_dir)
    version_file = index_path / "version.json"

    if not version_file.exists():
        return False

    try:
        with open(version_file, "r", encoding="utf-8") as f:
            version_info = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False

    # Check basic parameters
    if version_info.get("embedding_model") != embedding_model:
        return False
    if version_info.get("chunk_size") != chunk_size:
        return False
    if version_info.get("chunk_overlap") != chunk_overlap:
        return False
    # Check source directories (order-independent)
    if set(version_info.get("source_directories", [])) != set(source_directories):
        return False

    # Check data fingerprint
    current_fingerprint = _compute_data_fingerprint(documents)
    if version_info.get("data_fingerprint") != current_fingerprint:
        return False

    return True


def main():
    """Main function to prepare BM25 index for Hybrid RAG + Reranker."""
    parser = argparse.ArgumentParser(
        description="Prepare BM25 index for Hybrid RAG + Reranker (A4)"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild even if valid index exists",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help=f"Chunk size (default: {settings.chunk_size})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help=f"Chunk overlap (default: {settings.chunk_overlap})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Preparing BM25 Index for Hybrid RAG + Reranker (A4)")
    print("=" * 60)

    # Load raw documents (same as A0)
    print("\n1. Loading documents from type1 and type2 directories...")
    with timer("Document loading"):
        raw_documents, stats = load_documents_from_type1_type2(
            settings.type1_dir, settings.type2_dir
        )
    print(f"   Loaded {stats['total_documents']} raw documents")
    print(f"   Type1 files: {stats['type1_files']} (success: {stats['type1_success']}, failed: {stats['type1_failed']})")
    print(f"   Type2 files: {stats['type2_files']} (success: {stats['type2_success']}, failed: {stats['type2_failed']})")
    print(f"   Duplicates removed: {stats['duplicates_removed']}")

    if stats["total_documents"] == 0:
        print("ERROR: No documents loaded. Please check the data directories.")
        sys.exit(1)

    # Chunk documents (same as A0)
    print(f"\n2. Chunking documents (chunk_size={args.chunk_size}, overlap={args.chunk_overlap})...")
    with timer("Document chunking"):
        chunked_documents = chunk_documents(
            raw_documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    print(f"   Created {len(chunked_documents)} chunks from {len(raw_documents)} documents")

    # Check if we can load existing BM25 index
    bm25_index_dir = settings.bm25_index_dir
    if not args.force_rebuild and is_bm25_index_valid(
        bm25_index_dir,
        chunked_documents,
        settings.embedding_model,
        args.chunk_size,
        args.chunk_overlap,
        [settings.type1_dir, settings.type2_dir],
    ):
        print(f"\n3. Valid existing BM25 index found at {bm25_index_dir}")
        print("   Use --force-rebuild to rebuild the index.")
        return
    else:
        if os.path.exists(bm25_index_dir):
            print(f"\n3. Existing BM25 index found at {bm25_index_dir} but validation failed. Rebuilding...")
        else:
            print(f"\n3. No existing BM25 index found at {bm25_index_dir}. Preparing new index...")

    # Save BM25 index
    print(f"\n4. Saving BM25 index to {bm25_index_dir}...")
    with timer("BM25 index saving"):
        save_bm25_index(
            documents=chunked_documents,
            index_dir=bm25_index_dir,
            embedding_model=settings.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            source_directories=[settings.type1_dir, settings.type2_dir],
        )
    print(f"   BM25 index saved to: {bm25_index_dir}")

    # Final stats
    print("\n5. BM25 index preparation completed successfully!")
    print(f"   Embedding model: {settings.embedding_model}")
    print(f"   Chunk size: {args.chunk_size}")
    print(f"   Chunk overlap: {args.chunk_overlap}")
    print(f"   Source directories: {[settings.type1_dir, settings.type2_dir]}")
    print(f"   Document count: {len(set(d.metadata.get('document_id', '') for d in chunked_documents))}")
    print(f"   Chunk count: {len(chunked_documents)}")

    print("\n" + "=" * 60)
    print("BM25 index preparation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()