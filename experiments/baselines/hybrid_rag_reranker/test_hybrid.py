"""Test the complete Hybrid RAG + Reranker pipeline (A4).

This script tests the end-to-llm functionality by running the full pipeline
on a set of test questions and saving the results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .config import settings
from .pipeline import create_pipeline_from_config, HybridRAGRerankerPipeline
from experiments.shared.data_loader import load_documents_from_type1_type2
from experiments.shared.build_bm25_index import chunk_documents
from experiments.shared.evaluation import timer, save_jsonl


def load_test_questions(filepath: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load test questions from a JSONL file or use default ones.

    Args:
        filepath: Path to a JSONL file with test questions.

    Returns:
        List of question dictionaries.
    """
    # If no filepath provided, use the fixed question dataset
    if filepath is None:
        filepath = os.path.join(
            settings.datasets_dir, "scu_fixed_questions_v1_1.jsonl"
        )

    # If the fixed dataset doesn't exist, fall back to the original behavior
    if not os.path.exists(filepath):
        # Fall back to original test_questions.txt behavior
        fallback_path = os.path.join(settings.datasets_dir, "test_questions.txt")
        if os.path.exists(fallback_path):
            filepath = fallback_path
        else:
            # Ultimate fallback to hardcoded questions
            default_questions = [
                "資料科學系有哪些核心課程？",
                "畢業生主要去哪些產業工作？",
                "系統有什麼國際合作機會？",
                "如何申請系辦的助教職缺？",
                "系上有哪些社團或學會？",
                "研究所的申請截止時間是什麼時候？",
                "有什麼產學合作的專案可以參與？",
                "系學會每週的會議時間是什麼時候？",
                "畢業展通常在什麼時候舉辦？",
                "有哪些教授專長於人工智慧領域？",
            ]
            return [{"question_id": f"A{i+1}", "question": q, "test_type": "direct"}
                   for i, q in enumerate(default_questions)]

    try:
        questions = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    question_data = json.loads(line)
                    # Ensure required fields exist
                    if "question" not in question_data:
                        print(f"Warning: Line {line_num} missing 'question' field")
                        continue
                    # Set defaults for optional fields
                    question_data.setdefault("question_id", f"Q{line_num}")
                    question_data.setdefault("test_type", "unknown")
                    question_data.setdefault("as_of", "2026-10-19")
                    question_data.setdefault("dataset_version", "1.1")
                    questions.append(question_data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON on line {line_num}: {e}")
                    continue

        if not questions:
            print(f"Warning: No valid questions loaded from {filepath}")
            # Fall back to hardcoded questions
            default_questions = [
                "資料科學系有哪些核心課程？",
                "畢業生主要去哪些產業工作？",
                "系統有什麼國際合作機會？",
                "如何申請系辦的助教職缺？",
                "系上有哪些社團或學會？",
                "研究所的申請截止時間是什麼時候？",
                "有什麼產學合作的專案可以參與？",
                "系學會每週的會議時間是什麼時候？",
                "畢業展通常在什麼時候舉辦？",
                "有哪些教授專長於人工智慧領域？",
            ]
            return [{"question_id": f"A{i+1}", "question": q, "test_type": "direct"}
                   for i, q in enumerate(default_questions)]

        return questions
    except Exception as e:
        print(f"Warning: Failed to load questions from {filepath}: {e}")
        # Fall back to hardcoded questions
        default_questions = [
            "資料科學系有哪些核心課程？",
            "畢業生主要去哪些產業工作？",
            "系統有什麼國際合作機會？",
            "如何申請系辦的助教職缺？",
            "系上有哪些社團或學會？",
            "研究所的申請截止時間是什麼時候？",
            "有什麼產學合作的專案可以參與？",
            "系學會每週的會議時間是什麼時候？",
            "畢業展通常在什麼時候舉辦？",
            "有哪些教授專長於人工智慧領域？",
        ]
        return [{"question_id": f"A{i+1}", "question": q, "test_type": "direct"}
               for i, q in enumerate(default_questions)]


def save_run_metadata(output_dir: str, questions: List[Dict[str, Any]], use_reranker: bool) -> None:
    """Save run metadata to a JSON file.

    Args:
        output_dir: Directory to save the metadata file.
        questions: List of question dictionaries.
        use_reranker: Whether reranker was used in this run.
    """
    try:
        # Get actual settings from config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_file = os.path.join(output_dir, f"run_metadata_{timestamp}.json")

        # Try to get actual values from settings, with fallbacks
        try:
            embedding_model = settings.embedding_model
        except:
            embedding_model = "BAAI/bge-m3"

        try:
            llm_model_id = settings.llm_model_id
        except:
            llm_model_id = "meta-llama/Llama-3.2-3B-Instruct"

        try:
            vector_store_dir = settings.dense_vector_store_dir
        except:
            vector_store_dir = str(Path(__file__).resolve().parent.parent.parent /
                                 "experiments" / "vector_stores" / "type1_type2_bge_m3_v1")

        try:
            bm25_index_dir = settings.bm25_index_dir
        except:
            bm25_index_dir = str(Path(__file__).resolve().parent.parent.parent /
                               "experiments" / "vector_stores" / "hybrid_bm25_v1")

        try:
            top_k = settings.top_k
        except:
            top_k = 5

        try:
            chunk_size = settings.chunk_size
        except:
            chunk_size = 512

        try:
            chunk_overlap = settings.chunk_overlap
        except:
            chunk_overlap = 128

        metadata = {
            "run_timestamp": timestamp,
            "run_datetime": datetime.now().isoformat(),
            "dataset_version": "1.1",
            "dataset_as_of": "2026-10-19",
            "prompt_version": "a4_hybrid_reranker_v1",
            "question_dataset_path": os.path.join(
                settings.datasets_dir, "scu_fixed_questions_v1_1.jsonl"
            ),
            "question_count": len(questions),
            "llm_model_id": llm_model_id,
            "embedding_model": embedding_model,
            "vector_store_path": vector_store_dir,
            "vector_store_type": "FAISS IndexFlatIP",
            "bm25_index_path": bm25_index_dir,
            "bm25_index_type": "BM25",
            "top_k": top_k,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "use_reranker": use_reranker,
            "reranker_model": settings.reranker_model if use_reranker else None,
            "rrf_k": settings.rrf_k,
            "llm_parameters": {
                "temperature": getattr(settings, 'llm_temperature', 0.7),
                "top_p": getattr(settings, 'llm_top_p', 0.9),
                "max_new_tokens": getattr(settings, 'llm_max_new_tokens', 512),
                "do_sample": getattr(settings, 'llm_do_sample', True),
                "repetition_penalty": getattr(settings, 'llm_repetition_penalty', 1.0),
            }
        }

        # Try to add data fingerprint if available
        try:
            if hasattr(settings, 'data_fingerprint') and settings.data_fingerprint:
                metadata["data_fingerprint"] = settings.data_fingerprint
        except:
            pass

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"   Run metadata saved to: {metadata_file}")
    except Exception as e:
        print(f"Warning: Failed to save run metadata: {e}")


def main():
    """Main function to test the complete Hybrid RAG + Reranker pipeline."""
    parser = argparse.ArgumentParser(
        description="Test the complete Hybrid RAG + Reranker pipeline (A4)"
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default=None,
        help="Path to file containing test questions (JSONL format)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: from config)"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions to test (default: all)"
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Use reranker after fusion (A4-2)"
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Show retrieved sources in output"
    )
    args = parser.parse_args()

    # Determine test mode
    if args.use_reranker:
        test_mode = "A4-2 (with reranker)"
        mode_suffix = "with_reranker"
    else:
        test_mode = "A4-1 (without reranker)"
        mode_suffix = "without_reranker"

    print("=" * 60)
    print(f"Testing Complete Hybrid RAG + Reranker Pipeline ({test_mode})")
    print("=" * 60)

    # Load test questions
    questions_file = args.questions_file or os.path.join(
        settings.datasets_dir, "scu_fixed_questions_v1_1.jsonl"
    )
    questions = load_test_questions(questions_file)
    if args.num_questions:
        questions = questions[: args.num_questions]
    print(f"\n1. Loaded {len(questions)} test questions")

    # Create output directory
    output_dir = args.output_dir or settings.results_dir
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"results_{timestamp}_{mode_suffix}.jsonl")
    print(f"2. Results will be saved to: {output_file}")

    # Load and chunk documents (same as A0 for fair comparison)
    print("\n3. Loading and chunking documents...")
    with timer("Document loading and chunking"):
        raw_documents, stats = load_documents_from_type1_type2(
            settings.type1_dir, settings.type2_dir
        )
        documents = chunk_documents(
            raw_documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    print(f"   Loaded {stats['total_documents']} raw documents")
    print(f"   Created {len(documents)} chunks")

    # Initialize pipeline
    print(f"\n4. Initializing Hybrid RAG + Reranker pipeline...")
    try:
        with timer("Pipeline initialization"):
            pipeline = create_pipeline_from_config(
                use_reranker=args.use_reranker,
                documents=documents
            )
        doc_count = pipeline.vector_store.get_document_count()
        print(f"   Vector store documents: {doc_count}")
        if doc_count == 0:
            print("   ERROR: Vector store is empty. Please run build_index.py first.")
            sys.exit(1)
    except Exception as e:
        print(f"   ERROR: Failed to initialize pipeline: {e}")
        sys.exit(1)

    # Process each question
    print(f"\n5. Processing {len(questions)} questions...")
    print("-" * 60)

    results = []
    total_start_time = time.perf_counter()  # Use perf_counter for more precise timing

    for i, question_data in enumerate(questions, start=1):
        question_id = question_data.get("question_id", f"Q{i}")
        question_text = question_data.get("question", "")
        test_type = question_data.get("test_type", "unknown")

        print(f"\nQuestion {i}/{len(questions)} [{question_id}]:")
        print(f"   Q: {question_text}")
        print(f"   Type: {test_type}")

        try:
            # Run pipeline with timing
            pipeline_start = time.perf_counter()
            answer, retrieved_docs, retrieval_scores, retrieval_info = pipeline(
                question_text,
                return_context=True
            )
            pipeline_end = time.perf_counter()
            pipeline_total_time = pipeline_end - pipeline_start

            # Prepare result entry with all requested fields
            result_entry = {
                "question_id": question_id,
                "test_type": test_type,
                "question": question_text,
                "answer": answer,
                "success": not answer.startswith("Error:"),
                "error": None if not answer.startswith("Error:") else answer[6:],  # Remove "Error: " prefix
                "bm25_top_10": [
                    {
                        "rank": j + 1,
                        "source_file": doc.metadata.get("source_file", "Unknown"),
                        "content_type": doc.metadata.get("content_type", "Unknown"),
                        "score": float(score),
                        "content": doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""),
                        "metadata": doc.metadata
                    }
                    for j, (doc, score) in enumerate(
                        zip(retrieval_info["bm25_docs"], retrieval_info["bm25_scores"])
                    )
                ] if retrieval_info["bm25_docs"] else [],
                "dense_top_10": [
                    {
                        "rank": j + 1,
                        "source_file": doc.metadata.get("source_file", "Unknown"),
                        "content_type": doc.metadata.get("content_type", "Unknown"),
                        "score": float(score),
                        "content": doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""),
                        "metadata": doc.metadata
                    }
                    for j, (doc, score) in enumerate(
                        zip(retrieval_info["dense_docs"], retrieval_info["dense_scores"])
                    )
                ] if retrieval_info["dense_docs"] else [],
                "rrf_fusion_results": [
                    {
                        "rank": j + 1,
                        "source_file": doc.metadata.get("source_file", "Unknown"),
                        "content_type": doc.metadata.get("content_type", "Unknown"),
                        "score": float(score),
                        "content": doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""),
                        "metadata": doc.metadata
                    }
                    for j, (doc, score) in enumerate(
                        zip(retrieval_info["hybrid_docs"], retrieval_info["hybrid_scores"])
                    )
                ] if retrieval_info["hybrid_docs"] else [],
                "reranker_scores_and_sorting": [
                    {
                        "rank": j + 1,
                        "source_file": doc.metadata.get("source_file", "Unknown"),
                        "content_type": doc.metadata.get("content_type", "Unknown"),
                        "score": float(score),
                        "content": doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""),
                        "metadata": doc.metadata
                    }
                    for j, (doc, score) in enumerate(
                        zip(retrieval_info["final_docs"], retrieval_info["final_scores"])
                    )
                ] if args.use_reranker and retrieval_info["final_docs"] else [],
                "final_top_5_for_llm": [
                    {
                        "rank": j + 1,
                        "source_file": doc.metadata.get("source_file", "Unknown"),
                        "content_type": doc.metadata.get("content_type", "Unknown"),
                        "score": float(score),
                        "content": doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""),
                        "metadata": doc.metadata
                    }
                    for j, (doc, score) in enumerate(
                        zip(retrieval_info["final_docs"][:5], retrieval_info["final_scores"][:5])
                    )
                ] if retrieval_info["final_docs"] else [],
                "embedding_time": None,  # Embedding time is included in retrieval time
                "dense_retrieval_time": retrieval_info["retrieval_time"],
                "bm25_time": retrieval_info["retrieval_time"],  # Approximate - both retrieved together
                "fusion_time": None,  # Fusion time is included in retrieval time
                "reranking_time": retrieval_info["reranking_time"] if args.use_reranker else None,
                "generation_time": retrieval_info["generation_time"],
                "total_time": retrieval_info["total_time"],
            }

            # Add original question metadata if present
            for key in ["as_of", "dataset_version", "pair_group", "repeat_of"]:
                if key in question_data:
                    result_entry[key] = question_data[key]

            results.append(result_entry)

            # Print answer preview and timing info
            answer_preview = answer[:200].replace("\n", " ")
            print(f"   A: {answer_preview}...")
            print(f"   Timing - Retrieval: {retrieval_info['retrieval_time']:.3f}s")
            if args.use_reranker:
                print(f"   Timing - Reranking: {retrieval_info['reranking_time']:.3f}s")
            print(f"   Timing - Generation: {retrieval_info['generation_time']:.3f}s")
            print(f"   Timing - Total: {retrieval_info['total_time']:.3f}s")
            if args.show_sources and retrieval_info["final_docs"]:
                print(f"   Sources: {len(retrieval_info['final_docs'])} documents retrieved")
                for j, (doc, score) in enumerate(zip(retrieval_info["final_docs"], retrieval_info["final_scores"]), start=1):
                    source = doc.metadata.get("source_file", "Unknown")
                    print(f"     {j}. {source} (score: {score:.3f})")

        except Exception as e:
            print(f"   ERROR: {e}")
            traceback.print_exc()
            # Still save an error result with timing
            question_end_time = time.perf_counter()
            question_total_time = question_end_time - total_start_time  # This is approximate for errors

            result_entry = {
                "question_id": question_id,
                "test_type": test_type,
                "question": question_text,
                "answer": f"Error: {str(e)}",
                "success": False,
                "error": str(e),
                "bm25_top_10": [],
                "dense_top_10": [],
                "rrf_fusion_results": [],
                "reranker_scores_and_sorting": [],
                "final_top_5_for_llm": [],
                "embedding_time": None,
                "dense_retrieval_time": None,
                "bm25_time": None,
                "fusion_time": None,
                "reranking_time": None,
                "generation_time": None,
                "total_time": question_total_time,  # Approximate
            }

            # Add original question metadata if present
            for key in ["as_of", "dataset_version", "pair_group", "repeat_of"]:
                if key in question_data:
                    result_entry[key] = question_data[key]

            results.append(result_entry)

    total_end_time = time.perf_counter()
    total_elapsed = total_end_time - total_start_time

    # Save results
    print(f"\n6. Saving results to {output_file}...")
    save_jsonl(results, output_file)
    print(f"   Saved {len(results)} results")

    # Save run metadata
    save_run_metadata(output_dir, questions, args.use_reranker)

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Total questions: {len(questions)}")
    print(f"Total time: {total_elapsed:.2f} seconds")
    print(f"Average time per question: {total_elapsed/len(questions):.2f} seconds")
    print(f"Results saved to: {output_file}")

    # Count successful vs failed
    successful = sum(1 for r in results if r.get("success", False))
    failed = len(results) - successful
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed questions:")
        for result in results:
            if not result.get("success", True):
                print(f"  Q{result['question_id']}: {result['question'][:50]}...")

    print("\nTest completed!")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    # Import Document here to avoid circular imports
    from langchain_core.documents import Document
    sys.exit(main())