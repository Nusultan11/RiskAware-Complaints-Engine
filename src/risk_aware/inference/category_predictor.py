from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path


class CategoryPredictor:
    def __init__(self, model_path: str | Path) -> None:
        self.model = joblib.load(Path(model_path))

    def predict(self, texts: list[str]) -> np.ndarray:
        proba = self.model.predict_proba(texts)
        pred_idx = np.argmax(proba, axis=1)
        return np.array([self.model.labels[i] for i in pred_idx], dtype=object)
