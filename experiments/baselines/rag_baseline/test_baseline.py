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
from pathlib import Path
from typing import List, Dict, Any

from .config import settings
from .pipeline import create_pipeline_from_config
from experiments.shared.evaluation import timer, save_jsonl


def load_test_questions(filepath: Optional[str] = None) -> List[str]:
    """Load test questions from a file or use default ones.

    Args:
        filepath: Path to a text file with one question per line.

    Returns:
        List of question strings.
    """
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

    if filepath is None or not os.path.exists(filepath):
        return default_questions

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
        return questions if questions else default_questions
    except Exception as e:
        print(f"Warning: Failed to load questions from {filepath}: {e}")
        return default_questions


def main():
    """Main function to test the complete baseline."""
    parser = argparse.ArgumentParser(
        description="Test the complete RAG + LLM baseline"
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default=None,
        help="Path to file containing test questions (one per line)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: from config)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions to test (default: all)",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Show retrieved sources in output",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Testing Complete RAG + LLM Baseline")
    print("=" * 60)

    # Load test questions
    questions_file = args.questions_file or os.path.join(
        settings.datasets_dir, "test_questions.txt"
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
        print(f"   Vector store documents: {pipeline.vector_store.get_document_count()}")
        if pipeline.vector_store.get_document_count() == 0:
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

    for i, question in enumerate(questions, start=1):
        print(f"\nQuestion {i}/{len(questions)}:")
        print(f"   Q: {question}")

        try:
            # Run pipeline with timing
            with timer(f"Question {i}"):
                answer, retrieved_docs, scores = pipeline(
                    question, return_context=True
                )

            # Prepare result entry
            result_entry = {
                "question_id": i,
                "question": question,
                "answer": answer,
                "retrieved_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score),
                    }
                    for doc, score in zip(retrieved_docs, scores)
                ]
                if retrieved_docs
                else [],
                "timestamp": time.time(),
            }
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
            # Still save an error result
            results.append(
                {
                    "question_id": i,
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "retrieved_documents": [],
                    "timestamp": time.time(),
                    "error": str(e),
                }
            )

    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time

    # Save results
    print(f"\n5. Saving results to {output_file}...")
    save_jsonl(results, output_file)
    print(f"   Saved {len(results)} results")

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Total questions: {len(questions)}")
    print(f"Total time: {total_elapsed:.2f} seconds")
    print(f"Average time per question: {total_elapsed/len(questions):.2f} seconds")
    print(f"Results saved to: {output_file}")

    # Count successful vs failed
    successful = sum(1 for r in results if not r.get("answer", "").startswith("Error:"))
    failed = len(results) - successful
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed questions:")
        for result in results:
            if result.get("answer", "").startswith("Error:"):
                print(f"  Q{result['question_id']}: {result['question'][:50]}...")

    print("\nTest completed!")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    # Import Document here to avoid circular imports
    from langchain_core.documents import Document
    sys.exit(main())