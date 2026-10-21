"""Separate entry point for the provisional Train-intent diagnostic only.

Run as python -m experiments.a2_faq_finetuning.evaluate_hard_diagnostic.
Shares embedding/ranking/metrics implementation, never formal split results.
"""
from .evaluate_faq_retrieval import main


if __name__ == "__main__":
    raise SystemExit(main(diagnostic=True))
