"""Test the retrieval components of Hybrid RAG + Reranker (A4).

This script tests the BM25, dense, and hybrid retrieval functionality
by loading the built indices and running some example queries.
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
from .bm25_retrieval import BM25Retriever, HybridRetriever
from .fusion import reciprocal_rank_fusion


def main():
    """Main function to test retrieval."""
    parser = argparse.ArgumentParser(
        description="Test retrieval functionality for Hybrid RAG + Reranker (A4)"
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
    print("Testing Retrieval Components for Hybrid RAG + Reranker (A4)")
    print("=" * 60)

    # Load documents (same as A0)
    print("\n1. Loading documents from type1 and type2 directories...")
    raw_documents, stats = load_documents_from_type1_type2(
        settings.type1_dir, settings.type2_dir
    )
    print(f"   Loaded {stats['total_documents']} raw documents")

    # Chunk documents (same as A0)
    print(f"\n2. Chunking documents (chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap})...")
    from .build_bm25_index import chunk_documents
    chunked_documents = chunk_documents(
        raw_documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    print(f"   Created {len(chunked_documents)} chunks")

    # Load embedder
    print("\n3. Loading BGE-M3 embedder...")
    embedder = BGEM3Embedder(
        model_name=settings.embedding_model,
        device=settings.llm_device,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=settings.embedding_normalize,
    )
    print(f"   Model: {settings.embedding_model}")
    print(f"   Device: {embedder.device}")

    # Load dense vector store
    print("\n4. Loading FAISS vector store (dense retrieval)...")
    vector_store = FAISSVectorStore(
        embedding_dim=embedder.embedding_dim,
        index_path=settings.dense_vector_store_dir,
        use_hnsw=settings.use_hnsw,
        hnsw_M=settings.hnsw_M,
        hnsw_efConstruction=settings.hnsw_efConstruction,
    )
    if not vector_store.load():
        print(f"ERROR: Failed to load index from {settings.dense_vector_store_dir}")
        print("Please run build_index.py first to build the index.")
        sys.exit(1)
    print(f"   Loaded index with {vector_store.get_document_count()} documents")

    # Initialize dense retriever
    print("\n5. Initializing dense retriever...")
    dense_retriever = Retriever(
        vector_store=vector_store,
        embedder=embedder,
        top_k=args.top_k,
    )
    print(f"   Top-K: {args.top_k}")

    # Initialize BM25 retriever
    print("\n6. Initializing BM25 retriever...")
    bm25_retriever = BM25Retriever(
        documents=chunked_documents,
        k1=1.5,
        b=0.75,
        epsilon=0.25,
    )
    print(f"   BM25 retriever initialized with {len(chunked_documents)} documents")

    # Initialize hybrid retriever
    print("\n7. Initializing hybrid retriever...")
    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        rrf_k=settings.rrf_k,
    )
    print(f"   Hybrid retriever initialized with RRF k={settings.rrf_k}")

    # Test queries
    print(f"\n8. Testing retrieval with {len(args.queries)} queries...")
    print("-" * 60)

    all_results = []
    for i, query in enumerate(args.queries, start=1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)

        try:
            # Get hybrid retrieval results
            bm25_docs, bm25_scores, dense_docs, dense_scores, hybrid_docs, hybrid_scores = hybrid_retriever.retrieve(
                query, top_k=args.top_k
            )

            # Store results for summary
            all_results.append((query, bm25_docs, bm25_scores, dense_docs, dense_scores, hybrid_docs, hybrid_scores))

            # === DIAGNOSTIC INFORMATION ===
            # 1. Original query
            print(f"Original Query: {query}")

            # 2. Query tokenization
            from .bm25_retrieval import _chinese_tokenize
            query_tokens = _chinese_tokenize(query)
            print(f"Query Tokenization ({len(query_tokens)} tokens): {query_tokens}")

            # 3. Tokenizer info
            try:
                import jieba
                tokenizer_info = f"jieba version: {jieba.__version__}"
            except ImportError:
                tokenizer_info = "jieba not available (using fallback)"
            print(f"Tokenizer: {tokenizer_info}")

            # 4. BM25 raw scores (top-10) and positive count
            raw_scores = bm25_retriever.get_scores(query)
            # Get top-10 indices by raw score
            raw_top_indices = np.argsort(raw_scores)[::-1][:args.top_k]
            raw_top_scores = [raw_scores[i] for i in raw_top_indices]
            positive_count = sum(1 for s in raw_scores if s > 0)
            print(f"BM25 Raw Top-{args.top_k} Scores: {' '.join(f'{s:.8f}' for s in raw_top_scores)}")
            print(f"BM25 Positive Score Documents: {positive_count}/{len(raw_scores)}")

            # 5. BM25 filtered results count (after removing non-positive)
            print(f"BM25 Filtered Results (score > 0): {len(bm25_docs)}")

            # 6. Dense Top-10 scores
            print(f"Dense Top-{args.top_k} Scores: {' '.join(f'{s:.8f}' for s in dense_scores)}")

            # 7. Hybrid (RRF) Top-10 scores
            print(f"Hybrid (RRF) Top-{args.top_k} Scores: {' '.join(f'{s:.8f}' for s in hybrid_scores)}")

            # 8. Overlap between BM25 raw top-10 and dense top-10
            # Helper function to get document ID (consistent with hybrid retriever)
            def get_doc_id(doc: Document) -> str:
                doc_id = doc.metadata.get("document_id")
                if doc_id:
                    return doc_id
                return str(hash(doc.page_content))

            bm25_raw_top_docs = [bm25_retriever.documents[i] for i in raw_top_indices]
            bm25_raw_top_ids = set(get_doc_id(doc) for doc in bm25_raw_top_docs)
            dense_top_docs = dense_docs  # from the hybrid retriever result
            dense_top_ids = set(get_doc_id(doc) for doc in dense_top_docs)
            overlap = len(bm25_raw_top_ids & dense_top_ids)
            print(f"BM25-Dense Top-{args.top_k} Overlap: {overlap} documents")

            # Print BM25 results
            print(f"BM25 Results ({len(bm25_docs)} documents):")
            if not bm25_docs:
                print("   No results found.")
            else:
                for j, (doc, score) in enumerate(zip(bm25_docs, bm25_scores), start=1):
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

            # Print Dense results
            print(f"\nDense Results ({len(dense_docs)} documents):")
            if not dense_docs:
                print("   No results found.")
            else:
                for j, (doc, score) in enumerate(zip(dense_docs, dense_scores), start=1):
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

            # Print Hybrid (RRF) results
            print(f"\nHybrid (RRF) Results ({len(hybrid_docs)} documents):")
            if not hybrid_docs:
                print("   No results found.")
            else:
                for j, (doc, score) in enumerate(zip(hybrid_docs, hybrid_scores), start=1):
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
            import traceback
            traceback.print_exc()
            all_results.append((query, [], [], [], [], [], []))

    # Summary
    print("\n" + "=" * 60)
    print("Retrieval Test Summary")
    print("=" * 60)

    total_bm25 = sum(len(results[1]) for results in all_results)
    total_dense = sum(len(results[3]) for results in all_results)
    total_hybrid = sum(len(results[5]) for results in all_results)
    avg_bm25 = total_bm25 / len(args.queries) if args.queries else 0
    avg_dense = total_dense / len(args.queries) if args.queries else 0
    avg_hybrid = total_hybrid / len(args.queries) if args.queries else 0

    print(f"Total queries: {len(args.queries)}")
    print(f"Total BM25 results returned: {total_bm25}")
    print(f"Total Dense results returned: {total_dense}")
    print(f"Total Hybrid results returned: {total_hybrid}")
    print(f"Average BM25 results per query: {avg_bm25:.1f}")
    print(f"Average Dense results per query: {avg_dense:.1f}")
    print(f"Average Hybrid results per query: {avg_hybrid:.1f}")

    # Check if we got reasonable results
    if total_hybrid == 0:
        print("\nWARNING: No hybrid results returned for any query!")
        print("This could mean:")
        print("  1. The indices are empty or not built correctly")
        print("  2. The embedding model is not working")
        print("  3. The queries don't match any content in the index")
        return 1
    elif avg_hybrid < 1:
        print("\nWARNING: Very few hybrid results returned on average")
    else:
        print("\nRetrieval test completed successfully!")

    return 0


if __name__ == "__main__":
    # Import Document here to avoid circular imports
    from langchain_core.documents import Document
    # Import numpy here to avoid issues if not installed
    import numpy as np
    sys.exit(main())