from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from risk_aware.models.priority import RuleBasedPriorityModel, RuleWeights
from risk_aware.models.registry import build_text_stack
from risk_aware.models.thresholds import capacity_based_threshold, risk_based_threshold


@dataclass(slots=True)
class CategoryTrainer:
    stack_name: str
    stack_cfg: dict[str, Any]
    labels: list[str]

    def train(self, train_df: pd.DataFrame, text_col: str, label_col: str):
        model = build_text_stack(self.stack_name, self.labels, self.stack_cfg)
        model.fit(train_df[text_col].fillna("").tolist(), train_df[label_col].tolist())
        return model


@dataclass(slots=True)
class LegalTrainer:
    stack_name: str
    stack_cfg: dict[str, Any]

    def train(self, train_df: pd.DataFrame, text_col: str, label_col: str):
        model = build_text_stack(self.stack_name, labels=["0", "1"], stack_cfg=self.stack_cfg)
        model.fit(train_df[text_col].fillna("").tolist(), train_df[label_col].tolist())
        return model


@dataclass(slots=True)
class RuleBasedPriorityTrainer:
    weights: RuleWeights

    def train(self) -> RuleBasedPriorityModel:
        return RuleBasedPriorityModel(weights=self.weights)


@dataclass(slots=True)
class ThresholdTrainer:
    mode: str
    capacity_p1: float
    min_recall_p1: float

    def train(self, p1_scores: np.ndarray, y_true_p1: np.ndarray) -> float:
        if self.mode == "capacity_based":
            return capacity_based_threshold(p1_scores=p1_scores, capacity=self.capacity_p1)
        if self.mode == "risk_based":
            return risk_based_threshold(
                p1_scores=p1_scores,
                y_true_p1=y_true_p1,
                min_recall=self.min_recall_p1,
            )
        raise ValueError(f"Unsupported threshold mode: {self.mode}")

