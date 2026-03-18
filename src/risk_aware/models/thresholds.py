from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def capacity_based_threshold(p1_scores: np.ndarray, capacity: float) -> float:
    if not 0 < capacity <= 1:
        raise ValueError("capacity must be in (0, 1].")
    if p1_scores.size == 0:
        return 1.0
    quantile = max(0.0, min(1.0, 1.0 - capacity))
    return float(np.quantile(p1_scores, quantile))


def risk_based_threshold(
    p1_scores: np.ndarray,
    y_true_p1: np.ndarray,
    min_recall: float,
    grid_size: int = 500,
) -> float:
    if not 0 < min_recall <= 1:
        raise ValueError("min_recall must be in (0, 1].")
    if p1_scores.size == 0:
        return 1.0
    best_threshold = 0.0
    best_precision = -1.0
    thresholds = np.linspace(0.0, 1.0, num=grid_size)
    for thr in thresholds:
        predicted = p1_scores >= thr
        tp = int(np.sum(predicted & (y_true_p1 == 1)))
        fn = int(np.sum((~predicted) & (y_true_p1 == 1)))
        fp = int(np.sum(predicted & (y_true_p1 == 0)))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        if recall >= min_recall and precision >= best_precision:
            best_precision = precision
            best_threshold = float(thr)
    return best_threshold


@dataclass(slots=True)
class ThresholdPolicy:
    mode: str
    threshold_p1: float

    def resolve_priority(self, p1: float, p2: float, p3: float) -> str:
        if p1 >= self.threshold_p1:
            return "P1"
        return "P2" if p2 >= p3 else "P3"

