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