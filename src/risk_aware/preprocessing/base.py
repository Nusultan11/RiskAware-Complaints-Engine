from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

class TextPreprocessor(ABC):
    @abstractmethod
    def fit(self, texts: Sequence[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def transform(self, texts: Sequence[str]):
        raise NotImplementedError
