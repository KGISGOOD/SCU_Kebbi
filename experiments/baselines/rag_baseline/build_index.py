"""Build the FAISS vector index for the RAG baseline.

This script loads data from type1 and type2 directories, creates embeddings,
builds a FAISS index, and saves it with version information.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .config import settings
from experiments.shared.data_loader import load_documents_from_type1_type2
from experiments.shared.embeddings import BGEM3Embedder
from experiments.shared.vector_store import FAISSVectorStore
from experiments.shared.evaluation import timer


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 128,
) -> List[Document]:
    """Split documents into chunks while preserving metadata.

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


def main():
    """Main function to build the vector index."""
    parser = argparse.ArgumentParser(
        description="Build FAISS vector index for RAG baseline"
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
    print("Building FAISS Vector Index for RAG Baseline")
    print("=" * 60)

    # Load raw documents
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

    # Chunk documents
    print(f"\n2. Chunking documents (chunk_size={args.chunk_size}, overlap={args.chunk_overlap})...")
    with timer("Document chunking"):
        chunked_documents = chunk_documents(
            raw_documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    print(f"   Created {len(chunked_documents)} chunks from {len(raw_documents)} documents")

    # Initialize embedder
    print("\n3. Initializing BGE-M3 embedder...")
    with timer("Embedder initialization"):
        embedder = BGEM3Embedder(
            model_name=settings.embedding_model,
            device=settings.llm_device,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=settings.embedding_normalize,
        )
    print(f"   Model: {settings.embedding_model}")
    print(f"   Device: {embedder.device}")
    print(f"   Embedding dimension: {embedder.embedding_dim}")

    # Initialize vector store
    print("\n4. Initializing vector store...")
    vector_store = FAISSVectorStore(
        embedding_dim=embedder.embedding_dim,
        index_path=settings.vector_store_dir,
        use_hnsw=settings.use_hnsw,
        hnsw_M=settings.hnsw_M,
        hnsw_efConstruction=settings.hnsw_efConstruction,
    )

    # Check if we can load existing index
    if not args.force_rebuild and vector_store.load():
        # Validate the existing index
        if vector_store.is_valid(
            documents=chunked_documents,
            embedding_model=settings.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            source_directories=[settings.type1_dir, settings.type2_dir],
        ):
            print("   Valid existing index found and loaded.")
            print(f"   Index contains {vector_store.get_document_count()} documents.")
            return
        else:
            print("   Existing index found but validation failed. Rebuilding...")

    # Generate embeddings
    print(f"\n5. Generating embeddings for {len(chunked_documents)} chunks...")
    texts = [doc.page_content for doc in chunked_documents]
    with timer("Embedding generation"):
        embeddings = embedder.encode(texts, batch_size=settings.embedding_batch_size)
    print(f"   Generated embeddings shape: {embeddings.shape}")

    # Build index
    print("\n6. Building FAISS index...")
    with timer("Index building"):
        vector_store.build(embeddings, chunked_documents)
    print(f"   Index type: {'IndexHNSWFlat' if settings.use_hnsw else 'IndexFlatIP'}")
    print(f"   Index contains {vector_store.get_document_count()} vectors")

    # Save index and version info
    print("\n7. Saving index and version information...")
    with timer("Index saving"):
        vector_store.save()
        vector_store._save_version_info(
            documents=chunked_documents,
            embedding_model=settings.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            source_directories=[settings.type1_dir, settings.type2_dir],
        )
    print(f"   Index saved to: {settings.vector_store_dir}")
    print("   Version information saved.")

    # Final stats
    version_info = vector_store._load_version_info()
    if version_info:
        print("\n8. Index build completed successfully!")
        print(f"   Embedding model: {version_info.get('embedding_model')}")
        print(f"   Embedding dimension: {version_info.get('embedding_dimension')}")
        print(f"   Chunk size: {version_info.get('chunk_size')}")
        print(f"   Chunk overlap: {version_info.get('chunk_overlap')}")
        print(f"   Source directories: {version_info.get('source_directories')}")
        print(f"   Document count: {version_info.get('document_count')}")
        print(f"   Chunk count: {version_info.get('chunk_count')}")
        print(f"   Index type: {version_info.get('index_type')}")
        print(f"   Data fingerprint: {version_info.get('data_fingerprint')[:16]}...")

    print("\n" + "=" * 60)
    print("Index building completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Import Document here to avoid circular imports
    from langchain_core.documents import Document
    main()