"""Test the retrieval component of the RAG baseline.

This script tests the retrieval functionality by loading the built index
and running some example queries.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

from .config import settings
from experiments.shared.data_loader import load_documents_from_type1_type2
from experiments.shared.embeddings import BGEM3Embedder
from experiments.shared.vector_store import FAISSVectorStore
from experiments.shared.retrieval import Retriever


def main():
    """Main function to test retrieval."""
    parser = argparse.ArgumentParser(
        description="Test retrieval functionality for RAG baseline"
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=[
            "資料科學系的課程有哪些？",
            "畢業生的就業情況如何？",
            "國際合作計畫有哪些？",
            "系學會活動資訊",
            "教授研究領域",
        ],
        help="Queries to test retrieval with",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.top_k,
        help=f"Number of results to return (default: {settings.top_k})",
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Show detailed metadata for each result",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Testing Retrieval Component")
    print("=" * 60)

    # Load embedder
    print("\n1. Loading BGE-M3 embedder...")
    embedder = BGEM3Embedder(
        model_name=settings.embedding_model,
        device=settings.llm_device,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=settings.embedding_normalize,
    )
    print(f"   Model: {settings.embedding_model}")
    print(f"   Device: {embedder.device}")

    # Load vector store
    print("\n2. Loading FAISS vector store...")
    vector_store = FAISSVectorStore(
        embedding_dim=embedder.embedding_dim,
        index_path=settings.vector_store_dir,
        use_hnsw=settings.use_hnsw,
        hnsw_M=settings.hnsw_M,
        hnsw_efConstruction=settings.hnsw_efConstruction,
    )
    if not vector_store.load():
        print(f"ERROR: Failed to load index from {settings.vector_store_dir}")
        print("Please run build_index.py first to build the index.")
        sys.exit(1)
    print(f"   Loaded index with {vector_store.get_document_count()} documents")

    # Initialize retriever
    print("\n3. Initializing retriever...")
    retriever = Retriever(
        vector_store=vector_store,
        embedder=embedder,
        top_k=args.top_k,
    )
    print(f"   Top-K: {args.top_k}")

    # Test queries
    print(f"\n4. Testing retrieval with {len(args.queries)} queries...")
    print("-" * 60)

    all_results = []
    for i, query in enumerate(args.queries, start=1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)

        try:
            results = retriever.retrieve(query)
            all_results.append((query, results))

            if not results:
                print("   No results found.")
                continue

            for j, (doc, score) in enumerate(results, start=1):
                print(f"   Rank {j}:")
                print(f"     Score: {score:.4f}")
                print(f"     Source: {doc.metadata.get('source_file', 'Unknown')}")
                print(f"     Type: {doc.metadata.get('content_type', 'Unknown')}")
                if args.show_details:
                    print(f"     Path: {doc.metadata.get('source_path', 'Unknown')}")
                    print(f"     Row: {doc.metadata.get('original_row', 'N/A')}")
                    print(f"     Date: {doc.metadata.get('published_date', 'N/A')}")
                # Show content preview
                content_preview = doc.page_content[:100].replace("\n", " ")
                print(f"     Content: {content_preview}...")
        except Exception as e:
            print(f"   ERROR: {e}")
            all_results.append((query, []))

    # Summary
    print("\n" + "=" * 60)
    print("Retrieval Test Summary")
    print("=" * 60)
    total_results = sum(len(results) for _, results in all_results)
    avg_results = total_results / len(args.queries) if args.queries else 0
    print(f"Total queries: {len(args.queries)}")
    print(f"Total results returned: {total_results}")
    print(f"Average results per query: {avg_results:.1f}")

    # Check if we got reasonable results
    if total_results == 0:
        print("\nWARNING: No results returned for any query!")
        print("This could mean:")
        print("  1. The index is empty or not built correctly")
        print("  2. The embedding model is not working")
        print("  3. The queries don't match any content in the index")
        return 1
    elif avg_results < 1:
        print("\nWARNING: Very few results returned on average")
    else:
        print("\nRetrieval test completed successfully!")

    return 0


if __name__ == "__main__":
    # Import Document here to avoid circular imports
    from langchain_core.documents import Document
    sys.exit(main())