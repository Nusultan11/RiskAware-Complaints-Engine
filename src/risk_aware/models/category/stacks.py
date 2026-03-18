from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from risk_aware.models.base import TextClassifier


class TfidfLogRegTextStack(TextClassifier):
    def __init__(
        self,
        labels: list[str],
        max_features: int = 50000,
        ngram_range: tuple[int, int] = (1, 2),
        c: float = 2.0,
        class_weight: str | None = "balanced",
    ) -> None:
        self._labels = labels
        self.pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
                (
                    "clf",
                    LogisticRegression(
                        C=c,
                        class_weight=class_weight,
                        max_iter=1000,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    @property
    def labels(self) -> list[str]:
        return self._labels

    def fit(self, texts: Sequence[str], labels: Sequence[str | int]) -> None:
        self.pipeline.fit(list(texts), list(labels))

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))


class BiLSTMTextStack(TextClassifier):
    def __init__(self, labels: list[str]) -> None:
        self._labels = labels

    @property
    def labels(self) -> list[str]:
        return self._labels

    def fit(self, texts: Sequence[str], labels: Sequence[str | int]) -> None:
        raise NotImplementedError("BiLSTM stack scaffold is declared but not implemented yet.")

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("BiLSTM stack scaffold is declared but not implemented yet.")


class BertTextStack(TextClassifier):
    def __init__(self, labels: list[str]) -> None:
        self._labels = labels

    @property
    def labels(self) -> list[str]:
        return self._labels

    def fit(self, texts: Sequence[str], labels: Sequence[str | int]) -> None:
        raise NotImplementedError("Transformer stack scaffold is declared but not implemented yet.")

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("Transformer stack scaffold is declared but not implemented yet.")
