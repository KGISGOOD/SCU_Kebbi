"""Shift is the positive class. Thresholds are fitted on calibration only."""
from __future__ import annotations

import math


def classification(gold, predictions):
    if not gold or len(gold) != len(predictions):
        raise ValueError("empty/misaligned labels")
    tp = sum(g == 1 and p == 1 for g, p in zip(gold, predictions))
    tn = sum(g == 0 and p == 0 for g, p in zip(gold, predictions))
    fp = sum(g == 0 and p == 1 for g, p in zip(gold, predictions))
    fn = sum(g == 1 and p == 0 for g, p in zip(gold, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"count": len(gold), "accuracy": (tp + tn) / len(gold), "precision": precision,
            "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "confusion_matrix": {"tp_shift": tp, "tn_same": tn, "fp_shift": fp, "fn_shift": fn},
            "positive_label": "topic_shift", "zero_division_policy": 0.0}


def fit_threshold(scores, labels):
    if set(labels) != {0, 1} or not all(math.isfinite(s) for s in scores):
        raise ValueError("calibration requires finite scores and both classes")
    values = sorted(set(scores))
    candidates = [math.nextafter(values[0], -math.inf), math.nextafter(values[-1], math.inf)]
    candidates += [(a + b) / 2 for a, b in zip(values, values[1:])]
    ranked = [(classification(labels, [int(s < t) for s in scores]), t) for t in sorted(candidates)]
    result, threshold = max(ranked, key=lambda pair: (pair[0]["f1"], pair[0]["accuracy"], -pair[1]))
    return {"threshold": threshold, "rule": "score < threshold predicts topic_shift",
            "selection": "calibration F1, then accuracy, then smaller threshold", "calibration_metrics": result}
