"""Test the FAQ semantic matching baseline on the fixed 60 questions.

This script tests the FAQ matcher on the fixed 60 question dataset and saves the results.
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
from .matcher import FAQMatcher
from experiments.shared.evaluation import save_jsonl


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
                "系학會每週的會議時間是什麼時候？",
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
            "系학會每週的會議時間是什麼時候？",
            "畢業展通常在什麼時候舉辦？",
            "有哪些教授專長於人工智慧領域？",
        ]
        return [{"question_id": f"A{i+1}", "question": q, "test_type": "direct"}
               for i, q in enumerate(default_questions)]


def main():
    """Main function to test the FAQ matcher."""
    parser = argparse.ArgumentParser(
        description="Test the FAQ semantic matching baseline"
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
        "--show-details",
        action="store_true",
        help="Show detailed matching information in output"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Testing FAQ Semantic Matching Baseline")
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

    # Initialize FAQ matcher
    print("\n3. Initializing FAQ matcher...")
    try:
        matcher = FAQMatcher()
        # Try to load the index first
        if not matcher.load_index():
            print("   WARNING: Failed to load FAQ index. Attempting to build...")
            matcher.build_index()
        if not matcher.index_built:
            print("   ERROR: FAQ index is not built or loaded. Cannot proceed.")
            sys.exit(1)
        print(f"   FAQ matcher initialized with {len(matcher.faqs)} FAQs.")
    except Exception as e:
        print(f"   ERROR: Failed to initialize FAQ matcher: {e}")
        sys.exit(1)

    # Process each question
    print(f"\n4. Processing {len(questions)} questions...")
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
            # Time the matching phase
            match_start = time.perf_counter()
            matched_answer, similarity_score, matched_faq, top2_faqs, matching_time = matcher.match(
                question_text, return_faq=True
            )
            match_end = time.perf_counter()
            question_total_time = match_end - match_start

            # Determine match status
            threshold = settings.similarity_threshold
            is_matched = threshold is not None and similarity_score is not None and similarity_score >= threshold
            match_status = "matched" if is_matched else "no_match"

            # Prepare result entry with all requested fields
            result_entry = {
                "question_id": question_id,
                "test_type": test_type,
                "question": question_text,
                "answer": matched_answer if matched_answer else "No matching FAQ found.",
                "matched_faq_id": matched_faq.get("faq_id") if matched_faq else None,
                "matched_faq_question": matched_faq.get("question") if matched_faq else None,
                "similarity_score": similarity_score,
                "second_faq_id": top2_faqs[1].get("faq_id") if len(top2_faqs) > 1 else None,
                "second_similarity_score": top2_faqs[1].get("similarity_score") if len(top2_faqs) > 1 else None,
                "score_margin": (top2_faqs[0].get("similarity_score", 0.0) -
                                top2_faqs[1].get("similarity_score", 0.0)) if len(top2_faqs) > 1 else None,
                "match_status": match_status,
                "threshold": threshold,
                "embedding_time": None,  # Not measured separately in this simple matcher
                "matching_time": matching_time,
                "total_time": question_total_time,
                "success": matched_answer is not None,  # Success if we got an answer (even if no_match, we still have a default answer)
                "error": None if matched_answer is not None else "No answer returned",
            }

            # Add original question metadata if present
            for key in ["as_of", "dataset_version", "pair_group", "repeat_of"]:
                if key in question_data:
                    result_entry[key] = question_data[key]

            results.append(result_entry)

            # Print answer and timing info
            answer_preview = (matched_answer or "No matching FAQ found.")[:200].replace("\n", " ")
            print(f"   A: {answer_preview}...")
            print(f"   Similarity Score: {similarity_score:.4f}" if similarity_score is not None else "   Similarity Score: N/A")
            print(f"   Match Status: {match_status}")
            if args.show_details and top2_faqs:
                print(f"   Top-2 FAQs:")
                for j, faq in enumerate(top2_faqs, start=1):
                    print(f"     {j}. FAQ ID: {faq.get('faq_id')}, Score: {faq.get('similarity_score'):.4f}")
                    print(f"        Question: {faq.get('question')[:100]}...")
            print(f"   Timing - Matching: {matching_time:.3f}s, Total: {question_total_time:.3f}s")

        except Exception as e:
            print(f"   ERROR: {e}")
            traceback.print_exc()
            # Still save an error result
            question_end_time = time.perf_counter()
            question_total_time = question_end_time - total_start_time  # Approximate

            result_entry = {
                "question_id": question_id,
                "test_type": test_type,
                "question": question_text,
                "answer": f"Error: {str(e)}",
                "matched_faq_id": None,
                "matched_faq_question": None,
                "similarity_score": None,
                "second_faq_id": None,
                "second_similarity_score": None,
                "score_margin": None,
                "match_status": "error",
                "threshold": settings.similarity_threshold,
                "embedding_time": None,
                "matching_time": None,
                "total_time": question_total_time,  # Approximate
                "success": False,
                "error": str(e),
            }

            # Add original question metadata if present
            for key in ["as_of", "dataset_version", "pair_group", "repeat_of"]:
                if key in question_data:
                    result_entry[key] = question_data[key]

            results.append(result_entry)

    total_end_time = time.perf_counter()
    total_elapsed = total_end_time - total_start_time

    # Save results
    print(f"\n5. Saving results to {output_file}...")
    save_jsonl(results, output_file)
    print(f"   Saved {len(results)} results")

    # Save run metadata
    save_run_metadata(output_dir, questions, matcher)

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Total questions: {len(questions)}")
    print(f"Total time: {total_elapsed:.2f} seconds")
    print(f"Average time per question: {total_elapsed/len(questions):.2f} seconds")
    print(f"Results saved to: {output_file}")

    # Count by match status
    matched_count = sum(1 for r in results if r.get("match_status") == "matched")
    no_match_count = sum(1 for r in results if r.get("match_status") == "no_match")
    error_count = sum(1 for r in results if r.get("match_status") == "error")
    print(f"Matched: {matched_count}")
    print(f"No Match: {no_match_count}")
    print(f"Error: {error_count}")

    if error_count > 0:
        print("\nErrors encountered:")
        for result in results:
            if result.get("match_status") == "error":
                print(f"  Q{result['question_id']}: {result['question'][:50]}... - {result['error']}")

    print("\nTest completed!")
    return 0 if error_count == 0 else 1


def save_run_metadata(output_dir: str, questions: List[Dict[str, Any]], matcher: FAQMatcher) -> None:
    """Save run metadata to a JSON file.

    Args:
        output_dir: Directory to save the metadata file.
        questions: List of question dictionaries.
        matcher: The FAQMatcher instance used.
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
            llm_model_id = "N/A (FAQ matching does not use LLM)"
        except:
            llm_model_id = "N/A"

        try:
            faq_vector_store_dir = settings.faq_vector_store_dir
        except:
            faq_vector_store_dir = str(Path(__file__).resolve().parent.parent.parent /
                                     "experiments" / "vector_stores" / "faq_bge_m3_v1")

        try:
            faq_dataset_path = settings.faq_dataset_path
        except:
            faq_dataset_path = str(Path(__file__).resolve().parent.parent.parent /
                                 "experiments" / "datasets" / "faq" / "faq_canonical_v1.jsonl")

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
            "prompt_version": "N/A (FAQ matching)",
            "question_dataset_path": os.path.join(
                settings.datasets_dir, "scu_fixed_questions_v1_1.jsonl"
            ),
            "question_count": len(questions),
            "faq_dataset_path": faq_dataset_path,
            "faq_count": len(matcher.faqs) if matcher.faqs else 0,
            "embedding_model": embedding_model,
            "vector_store_path": faq_vector_store_dir,
            "vector_store_type": "FAISS IndexFlatIP",
            "top_k": top_k,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "similarity_threshold": settings.similarity_threshold,
            "faq_matching_parameters": {
                "similarity_threshold": settings.similarity_threshold,
                "top_k_for_analysis": max(2, settings.top_k),
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


if __name__ == "__main__":
    # Import Document here to avoid circular imports
    from langchain_core.documents import Document
    sys.exit(main())