from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from risk_aware.preprocessing.base import TextPreprocessor


def neural_clean(text: str) -> str:
    """
    LSTM/Transformer-friendly normalization aligned with EDA findings:
    - keep anonymization signal ("xxxx") as <anon>
    - keep digits as <num>
    - remove punctuation noise
    """
    cleaned = str(text or "").strip().lower()
    cleaned = re.sub(r"\bx{2,}\b", " <anon> ", cleaned)
    cleaned = re.sub(r"\d+", " <num> ", cleaned)
    cleaned = re.sub(r"[^a-z<>'\s]", " ", cleaned)
    return " ".join(cleaned.split())


def simple_tokenize(text: str) -> list[str]:
    normalized = neural_clean(text)
    return normalized.split() if normalized else []


@dataclass(slots=True)
class Vocabulary:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]


class NeuralTextPreprocessor(TextPreprocessor):
    """
    Token-id preprocessing for sequence models (LSTM baseline).
    Vocabulary is fitted only on training texts to avoid leakage.
    """

    def __init__(
        self,
        max_vocab_size: int = 50_000,
        min_token_freq: int = 2,
        max_length: int = 256,
    ) -> None:
        if max_vocab_size < 10:
            raise ValueError("max_vocab_size must be >= 10")
        if min_token_freq < 1:
            raise ValueError("min_token_freq must be >= 1")
        if max_length < 8:
            raise ValueError("max_length must be >= 8")

        self.max_vocab_size = int(max_vocab_size)
        self.min_token_freq = int(min_token_freq)
        self.max_length = int(max_length)
        self.vocab: Vocabulary | None = None

    def _require_vocab(self) -> Vocabulary:
        if self.vocab is None:
            raise ValueError("Vocabulary is not fitted. Call fit() first.")
        return self.vocab

    def fit(self, texts: Sequence[str]) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(simple_tokenize(str(text)))

        specials = ["<pad>", "<unk>"]
        max_base_vocab = max(self.max_vocab_size - len(specials), 0)

        ranked_tokens = [
            token
            for token, freq in counter.most_common()
            if freq >= self.min_token_freq
        ][:max_base_vocab]

        id_to_token = specials + ranked_tokens
        token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
        self.vocab = Vocabulary(token_to_id=token_to_id, id_to_token=id_to_token)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        vocab = self._require_vocab()
        encoded = np.full(
            shape=(len(texts), self.max_length),
            fill_value=vocab.pad_id,
            dtype=np.int32,
        )

        for i, text in enumerate(texts):
            tokens = simple_tokenize(str(text))[: self.max_length]
            token_ids = [vocab.token_to_id.get(tok, vocab.unk_id) for tok in tokens]
            if token_ids:
                encoded[i, : len(token_ids)] = np.asarray(token_ids, dtype=np.int32)
        return encoded

    def build_attention_mask(self, token_ids: np.ndarray) -> np.ndarray:
        vocab = self._require_vocab()
        return (token_ids != vocab.pad_id).astype(np.uint8)
