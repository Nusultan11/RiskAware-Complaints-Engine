from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

from risk_aware.models.base import TabularClassifier


def normalize_column(values: np.ndarray) -> np.ndarray:
    vmax = values.max(initial=0.0)
    if vmax <= 0:
        return np.zeros_like(values)
    return values / vmax


@dataclass(slots=True)
class RuleWeights:
    w_amount: float = 0.35
    w_repeat: float = 0.25
    w_legal: float = 0.40


class RuleBasedPriorityModel:
    def __init__(self, weights: RuleWeights | None = None) -> None:
        self.weights = weights or RuleWeights()

    def score(self, amount: np.ndarray, repeat_count: np.ndarray, legal_proba: np.ndarray) -> np.ndarray:
        normalized_amount = normalize_column(amount)
        normalized_repeat = normalize_column(repeat_count)
        return (
            self.weights.w_amount * normalized_amount
            + self.weights.w_repeat * normalized_repeat
            + self.weights.w_legal * legal_proba
        )


class HeuristicPriorityClassifier(TabularClassifier):
    def __init__(self, weights: RuleWeights | None = None) -> None:
        self.weights = weights or RuleWeights()
        self.scorer = RuleBasedPriorityModel(self.weights)
        self._labels = ["P1", "P2", "P3"]

    @property
    def labels(self) -> list[str]:
        return self._labels

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        _ = (x, y)
        return None

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] < 6:
            raise ValueError("Priority feature matrix must include category probabilities and CRM features.")

        legal = x[:, -6]
        amount = x[:, -5]
        repeat = x[:, -4]

        p1 = np.clip(self.scorer.score(amount=amount, repeat_count=repeat, legal_proba=legal), 0.0, 1.0)
        residual = 1.0 - p1
        repeat_factor = normalize_column(repeat)
        p2 = residual * (0.45 + 0.45 * repeat_factor)
        p3 = np.maximum(0.0, residual - p2)
        probs = np.column_stack([p1, p2, p3])
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return probs


class LightGBMPriorityModel(TabularClassifier):
    def __init__(self, **kwargs: object) -> None:
        params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "objective": "multiclass",
            "random_state": 42,
        }
        params.update(kwargs)
        self.model = LGBMClassifier(**params)
        self._labels = ["P1", "P2", "P3"]

    @property
    def labels(self) -> list[str]:
        return self._labels

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(x))


class MLPPriorityModel(TabularClassifier):
    def __init__(self, **kwargs: object) -> None:
        params = {"hidden_layer_sizes": (256, 128), "alpha": 0.0001, "max_iter": 400, "random_state": 42}
        params.update(kwargs)
        self.model = MLPClassifier(**params)
        self._labels = ["P1", "P2", "P3"]

    @property
    def labels(self) -> list[str]:
        return self._labels

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(x))


def build_priority_model(model_name: str, model_cfg: dict[str, object]) -> TabularClassifier:
    if model_name == "rule_based":
        weights = RuleWeights(
            w_amount=float(model_cfg.get("w_amount", 0.35)),
            w_repeat=float(model_cfg.get("w_repeat", 0.25)),
            w_legal=float(model_cfg.get("w_legal", 0.4)),
        )
        return HeuristicPriorityClassifier(weights=weights)
    if model_name == "lightgbm":
        return LightGBMPriorityModel(**model_cfg)
    if model_name == "mlp":
        return MLPPriorityModel(**model_cfg)
    raise ValueError(f"Unsupported priority model: {model_name}")
