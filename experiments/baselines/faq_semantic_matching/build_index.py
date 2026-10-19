"""Build the FAQ vector store index for the FAQ semantic matching baseline.

This script loads the FAQ dataset, computes embeddings using BGE-M3,
and builds a FAISS IndexFlatIP index.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .matcher import FAQMatcher
from .config import settings


def main():
    """Main function to build the FAQ index."""
    parser = argparse.ArgumentParser(
        description="Build the FAQ vector store index for FAQ semantic matching"
    )
    parser.add_argument(
        "--faq-dataset",
        type=str,
        default=None,
        help="Path to the FAQ JSONL dataset (default: from config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the vector store (default: from config)"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild of the index even if it exists"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Building FAQ Vector Store Index")
    print("=" * 60)

    # Determine paths
    faq_dataset_path = args.faq_dataset or settings.faq_dataset_path
    faq_vector_store_dir = args.output_dir or settings.faq_vector_store_dir

    print(f"FAQ dataset path: {faq_dataset_path}")
    print(f"Vector store directory: {faq_vector_store_dir}")

    # Check if the dataset exists
    if not os.path.exists(faq_dataset_path):
        print(f"ERROR: FAQ dataset not found at {faq_dataset_path}")
        print("Please run the faq_loader.py script to generate the dataset first.")
        return 1

    # Initialize the matcher
    matcher = FAQMatcher(
        faq_dataset_path=faq_dataset_path,
        faq_vector_store_dir=faq_vector_store_dir
    )

    # Check if index exists and whether to rebuild
    index_exists = os.path.exists(os.path.join(faq_vector_store_dir, "index.faiss"))
    if index_exists and not args.force_rebuild:
        print(f"Existing index found at {faq_vector_store_dir}")
        print("Use --force-rebuild to rebuild the index.")
        # Try to load the index to verify
        if matcher.load_index():
            print("Index loaded successfully.")
            return 0
        else:
            print("Failed to load existing index. Rebuilding...")
            # Fall through to rebuild
    else:
        if index_exists and args.force_rebuild:
            print(f"Removing existing index at {faq_vector_store_dir} and rebuilding...")
        else:
            print("No existing index found. Building new index...")

    # Build the index
    start_time = time.time()
    matcher.build_index()
    end_time = time.time()

    print(f"\nIndex building completed in {end_time - start_time:.2f} seconds.")
    print(f"Index saved to: {faq_vector_store_dir}")

    return 0


if __name__ == "__main__":
    # Import json here to avoid circular imports
    import json
    sys.exit(main())