from __future__ import annotations

import re
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer

from risk_aware.preprocessing.base import TextPreprocessor


def tfidf_clean(text: str) -> str:
    cleaned = str(text or "").strip().lower()
    # Based on EDA: "xxxx" is dominant anonymization token and acts as TF-IDF noise.
    cleaned = re.sub(r"\b(x{2,})\b", " ", cleaned)
    cleaned = re.sub(r"\d+", " num ", cleaned)
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned


class TfidfTextPreprocessor(TextPreprocessor):
    def __init__(self, max_features: int = 50000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=False,
            preprocessor=tfidf_clean,
        )

    def fit(self, texts: Sequence[str]) -> None:
        self.vectorizer.fit(list(texts))

    def transform(self, texts: Sequence[str]):
        return self.vectorizer.transform(list(texts))
