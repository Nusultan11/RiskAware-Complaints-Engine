from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from risk_aware.models.category.registry import build_text_stack


@dataclass(slots=True)
class CategoryTrainer:
    stack_name: str
    stack_cfg: dict[str, Any]
    labels: list[str]

    def train(self, train_df: pd.DataFrame, text_col: str, label_col: str):
        model = build_text_stack(self.stack_name, self.labels, self.stack_cfg)
        model.fit(train_df[text_col].fillna("").tolist(), train_df[label_col].tolist())
        return model
