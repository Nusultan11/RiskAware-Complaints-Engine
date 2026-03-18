from __future__ import annotations

from typing import Any, Sequence

from sklearn.metrics import accuracy_score, classification_report, f1_score


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(f1_score(y_true=y_true, y_pred=y_pred, average="macro"))


def compute_category_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    return {
        "macro_f1": macro_f1(y_true=y_true, y_pred=y_pred),
        "accuracy": float(accuracy_score(y_true=y_true, y_pred=y_pred)),
        "classification_report": classification_report(
            y_true=y_true,
            y_pred=y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }
