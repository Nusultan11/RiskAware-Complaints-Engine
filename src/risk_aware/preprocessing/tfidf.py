from __future__ import annotations

from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer

from risk_aware.preprocessing.base import TextPreprocessor


class TfidfTextPreprocessor(TextPreprocessor):
    def __init__(self, max_features: int = 50000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    def fit(self, texts: Sequence[str]) -> None:
        self.vectorizer.fit(list(texts))

    def transform(self, texts: Sequence[str]):
        return self.vectorizer.transform(list(texts))
