from __future__ import annotations

from typing import Sequence

from risk_aware.preprocessing.base import TextPreprocessor


class NeuralTextPreprocessor(TextPreprocessor):
    def fit(self, texts: Sequence[str]) -> None:
        # Reserved for tokenizer/vocab fitting in LSTM and Transformer stacks.
        _ = texts

    def transform(self, texts: Sequence[str]):
        # Reserved for tokenization/encoding logic in next iteration.
        return list(texts)
