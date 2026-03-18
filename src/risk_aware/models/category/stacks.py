from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from risk_aware.models.base import TextClassifier
from risk_aware.preprocessing.tfidf import tfidf_clean


class TfidfLogRegTextStack(TextClassifier):
    def __init__(
        self,
        labels: list[str],
        max_features: int = 50_000,
        ngram_range: tuple[int, int] = (1, 2),
        c: float = 2.0,
        class_weight: str | None = "balanced",
    ) -> None:
        self._labels = labels
        self.pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range,
                        lowercase=False,
                        preprocessor=tfidf_clean,
                    ),
                ),
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

    def predict(self, texts: Sequence[str]) -> list[str]:
        return self.pipeline.predict(list(texts)).tolist()


class BiLSTMTextStack(TextClassifier):
    def __init__(self, labels: list[str]) -> None:
        self._labels = labels

    @property
    def labels(self) -> list[str]:
        return self._labels

    def fit(self, texts: Sequence[str], labels: Sequence[str | int]) -> None:
        raise NotImplementedError("BiLSTM stack is not implemented yet.")

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("BiLSTM stack is not implemented yet.")


class BertTextStack(TextClassifier):
    def __init__(self, labels: list[str]) -> None:
        self._labels = labels

    @property
    def labels(self) -> list[str]:
        return self._labels

    def fit(self, texts: Sequence[str], labels: Sequence[str | int]) -> None:
        raise NotImplementedError("BERT stack is not implemented yet.")

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("BERT stack is not implemented yet.")
