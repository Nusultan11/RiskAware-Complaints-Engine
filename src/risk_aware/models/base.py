from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class TextClassifier(ABC):
    @property
    @abstractmethod
    def labels(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def fit(self, texts: Sequence[str], labels: Sequence[str | int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class TabularClassifier(ABC):
    @property
    @abstractmethod
    def labels(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

