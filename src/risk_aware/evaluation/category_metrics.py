from __future__ import annotations

from typing import Sequence

from sklearn.metrics import f1_score


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(f1_score(y_true=y_true, y_pred=y_pred, average="macro"))
