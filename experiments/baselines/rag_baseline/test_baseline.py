"""Test the complete RAG + LLM baseline.

This script tests the end-to-end functionality by running the full pipeline
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
from typing import List, Dict, Any, Optional

from .config import settings
from .pipeline import create_pipeline_from_config
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


def save_run_metadata(output_dir: str, questions: List[Dict[str, Any]]) -> None:
    """Save run metadata to a JSON file.

    Args:
        output_dir: Directory to save the metadata file.
        questions: List of question dictionaries.
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
            vector_store_dir = settings.vector_store_dir
        except:
            vector_store_dir = str(Path(__file__).resolve().parent.parent.parent /
                                 "experiments" / "vector_stores" / "type1_type2_bge_m3_v1")

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
            "prompt_version": "baseline_basic_v1",
            "question_dataset_path": os.path.join(
                settings.datasets_dir, "scu_fixed_questions_v1_1.jsonl"
            ),
            "question_count": len(questions),
            "llm_model_id": llm_model_id,
            "embedding_model": embedding_model,
            "vector_store_path": vector_store_dir,
            "vector_store_type": "FAISS IndexFlatIP",
            "top_k": top_k,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
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
    """Main function to test the complete baseline."""
    parser = argparse.ArgumentParser(
        description="Test the complete RAG + LLM baseline"
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
        "--show-sources",
        action="store_true",
        help="Show retrieved sources in output"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Testing Complete RAG + LLM Baseline")
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
    output_file = os.path.join(output_dir, f"results_{timestamp}.jsonl")
    print(f"2. Results will be saved to: {output_file}")

    # Initialize pipeline
    print("\n3. Initializing RAG pipeline...")
    try:
        with timer("Pipeline initialization"):
            pipeline = create_pipeline_from_config()
        doc_count = pipeline.vector_store.get_document_count()
        print(f"   Vector store documents: {doc_count}")
        if doc_count == 0:
            print("   ERROR: Vector store is empty. Please run build_index.py first.")
            sys.exit(1)
    except Exception as e:
        print(f"   ERROR: Failed to initialize pipeline: {e}")
        sys.exit(1)

    # Process each question
    print(f"\n4. Processing {len(questions)} questions...")
    print("-" * 60)

    results = []
    total_start_time = time.time()

    for i, question_data in enumerate(questions, start=1):
        question_id = question_data.get("question_id", f"Q{i}")
        question_text = question_data.get("question", "")
        test_type = question_data.get("test_type", "unknown")

        print(f"\nQuestion {i}/{len(questions)} [{question_id}]:")
        print(f"   Q: {question_text}")
        print(f"   Type: {test_type}")

        try:
            # Run pipeline with timing
            with timer(f"Question {i}"):
                answer, retrieved_docs, scores = pipeline(
                    question_text, return_context=True
                )

            # Prepare result entry with all requested fields
            result_entry = {
                "question_id": question_id,
                "test_type": test_type,
                "question": question_text,
                "answer": answer,
                "success": not answer.startswith("Error:"),
                "error": None if not answer.startswith("Error:") else answer[6:],  # Remove "Error: " prefix
                "retrieved_sources": [
                    {
                        "rank": j + 1,
                        "source_file": doc.metadata.get("source_file", "Unknown"),
                        "content_type": doc.metadata.get("content_type", "Unknown"),
                        "score": float(score),
                        "content": doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""),
                        "metadata": doc.metadata
                    }
                    for j, (doc, score) in enumerate(zip(retrieved_docs, scores))
                ]
                if retrieved_docs
                else [],
                "timestamp": time.time(),
                "total_time": None,  # Will be filled in after timing
                "retrieval_time": None,
                "generation_time": None,
            }

            # Add original question metadata if present
            for key in ["as_of", "dataset_version", "pair_group", "repeat_of"]:
                if key in question_data:
                    result_entry[key] = question_data[key]

            results.append(result_entry)

            # Print answer preview
            answer_preview = answer[:200].replace("\n", " ")
            print(f"   A: {answer_preview}...")
            if args.show_sources and retrieved_docs:
                print(f"   Sources: {len(retrieved_docs)} documents retrieved")
                for j, (doc, score) in enumerate(zip(retrieved_docs, scores), start=1):
                    source = doc.metadata.get("source_file", "Unknown")
                    print(f"     {j}. {source} (score: {score:.3f})")

        except Exception as e:
            print(f"   ERROR: {e}")
            traceback.print_exc()
            # Still save an error result
            result_entry = {
                "question_id": question_id,
                "test_type": test_type,
                "question": question_text,
                "answer": f"Error: {str(e)}",
                "success": False,
                "error": str(e),
                "retrieved_sources": [],
                "timestamp": time.time(),
                "total_time": None,
            }

            # Add original question metadata if present
            for key in ["as_of", "dataset_version", "pair_group", "repeat_of"]:
                if key in question_data:
                    result_entry[key] = question_data[key]

            results.append(result_entry)

    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time

    # Calculate individual question times (approximate)
    # In a more sophisticated implementation, we would track these per question
    for i, result in enumerate(results):
        result["total_time"] = total_elapsed / len(results) if results else 0
        # Note: For more accurate timing, we would need to modify the pipeline
        # to return timing information, but the instructions say not to redesign
        # the pipeline just for timing metrics

    # Save results
    print(f"\n5. Saving results to {output_file}...")
    save_jsonl(results, output_file)
    print(f"   Saved {len(results)} results")

    # Save run metadata
    save_run_metadata(output_dir, questions)

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