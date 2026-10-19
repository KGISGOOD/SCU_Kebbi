"""FAQ loader for the FAQ semantic matching baseline.

Loads FAQ data from CSV files and converts to JSONL format.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import List, Dict, Any

from .config import settings


def load_parents_faq(csv_path: str) -> List[Dict[str, Any]]:
    """Load parents FAQ from CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of FAQ dictionaries.
    """
    faqs = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                faq = {
                    "faq_id": f"P{row['編號'].strip()}",  # Prefix with P for parents
                    "question": row['問題內容'].strip(),
                    "answer": row['ai答覆'].strip(),
                    "source_file": os.path.basename(csv_path),
                    "source_type": "parents_faq",
                    "review_status": "approved"  # Assume approved for now
                }
                faqs.append(faq)
    except Exception as e:
        print(f"Error loading parents FAQ from {csv_path}: {e}")
    return faqs


def load_highschool_faq(csv_path: str) -> List[Dict[str, Any]]:
    """Load high school FAQ from CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of FAQ dictionaries.
    """
    faqs = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                faq = {
                    "faq_id": f"S{row['編號'].strip()}",  # Prefix with S for high school
                    "question": row['問題內容'].strip(),
                    "answer": row['ai答覆'].strip(),
                    "source_file": os.path.basename(csv_path),
                    "source_type": "highschool_faq",
                    "review_status": "approved"  # Assume approved for now
                }
                faqs.append(faq)
    except Exception as e:
        print(f"Error loading high school FAQ from {csv_path}: {e}")
    return faqs


def load_all_faq() -> List[Dict[str, Any]]:
    """Load all FAQ data from the known CSV files.

    Returns:
        List of all FAQ dictionaries.
    """
    type2_dir = Path(settings.type2_dir) if hasattr(settings, 'type2_dir') else Path("../畢業專題_爬蟲/type2")
    parents_csv = type2_dir / "資料擴充 - 家長 FAQ 資料集（30 題）.csv"
    highschool_csv = type2_dir / "資料擴充 - 高中生 FAQ 資料集（30 題）.csv"

    faqs = []
    faqs.extend(load_parents_faq(str(parents_csv)))
    faqs.extend(load_highschool_faq(str(highschool_csv)))

    return faqs


def save_faq_as_jsonl(faqs: List[Dict[str, Any]], output_path: str) -> None:
    """Save FAQ list as JSONL file.

    Args:
        faqs: List of FAQ dictionaries.
        output_path: Path to save the JSONL file.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for faq in faqs:
                f.write(json.dumps(faq, ensure_ascii=False) + '\n')
        print(f"Saved {len(faqs)} FAQs to {output_path}")
    except Exception as e:
        print(f"Error saving FAQ JSONL to {output_path}: {e}")


if __name__ == "__main__":
    # For testing the loader
    faqs = load_all_faq()
    print(f"Loaded {len(faqs)} FAQs")
    # Save to the canonical path
    save_faq_as_jsonl(faqs, settings.faq_dataset_path)