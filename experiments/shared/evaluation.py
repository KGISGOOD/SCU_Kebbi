"""Evaluation utilities for timing and result saving.

Provides timing context managers and JSONL saving functions.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import numpy as np


@contextmanager
def timer(name: str = "") -> Generator[None, None, None]:
    """Context manager to time a block of code.

    Args:
        name: Name of the operation for logging.

    Yields:
        None
    """
    start = time.time()
    yield
    end = time.time()
    elapsed = end - start
    if name:
        print(f"[{name}] Elapsed time: {elapsed:.3f} seconds")
    else:
        print(f"Elapsed time: {elapsed:.3f} seconds")


def save_jsonl(
    data: List[Dict[str, Any]],
    filepath: str,
    append: bool = False,
) -> None:
    """Save a list of dictionaries to a JSONL file.

    Args:
        data: List of dictionaries to save.
        filepath: Path to the output JSONL file.
        append: Whether to append to existing file.
    """
    mode = "a" if append else "w"
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file.

    Args:
        filepath: Path to the input JSONL file.

    Returns:
        List of dictionaries.
    """
    data = []
    if not os.path.exists(filepath):
        return data
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse line: {line}")
    return data


def compute_metrics(
    retrieved_docs: List[List[Document]],
    ground_truth: Optional[List[List[str]]] = None,
) -> Dict[str, float]:
    """Compute basic retrieval metrics.

    Args:
        retrieved_docs: List of lists of retrieved documents per query.
        ground_truth: Optional list of lists of relevant document IDs per query.

    Returns:
        Dictionary of metrics.
    """
    metrics = {}
    if ground_truth is not None:
        # Compute precision@k, recall@k, MRR, etc.
        # This is a simplified version - in practice you'd use proper IR metrics
        pass
    # Add basic stats
    avg_docs_per_query = np.mean([len(docs) for docs in retrieved_docs]) if retrieved_docs else 0
    metrics["avg_docs_per_query"] = avg_docs_per_query
    return metrics


if __name__ == "__main__":
    # Simple test
    with timer("test"):
        time.sleep(0.1)
    test_data = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
    save_jsonl(test_data, "test.jsonl")
    loaded = load_jsonl("test.jsonl")
    print(f"Loaded {len(loaded)} items")