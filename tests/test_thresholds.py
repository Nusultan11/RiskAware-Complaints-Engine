import numpy as np

from risk_aware.models.thresholds import capacity_based_threshold, risk_based_threshold


def test_capacity_threshold_respects_top_share() -> None:
    scores = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1])
    threshold = capacity_based_threshold(scores, capacity=0.25)
    predicted_p1 = scores >= threshold
    share = float(np.mean(predicted_p1))
    assert share <= 0.25 + 1e-9


def test_risk_threshold_hits_target_recall_when_possible() -> None:
    scores = np.array([0.95, 0.85, 0.8, 0.75, 0.5, 0.4, 0.2, 0.1])
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    threshold = risk_based_threshold(scores, y_true, min_recall=0.75)
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (y_true == 1)))
    fn = int(np.sum((~predicted) & (y_true == 1)))
    recall = tp / (tp + fn)
    assert recall >= 0.75

