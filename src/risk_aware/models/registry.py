from __future__ import annotations

from typing import Any

from risk_aware.models.base import TextClassifier
from risk_aware.models.category.stacks import TfidfLogRegTextStack


def build_text_stack(stack_name: str, labels: list[str], stack_cfg: dict[str, Any]) -> TextClassifier:
    if stack_name == "tfidf_lr":
        ngram = stack_cfg.get("ngram_range", [1, 2])
        return TfidfLogRegTextStack(
            labels=labels,
            max_features=int(stack_cfg.get("max_features", 50000)),
            ngram_range=(int(ngram[0]), int(ngram[1])),
            c=float(stack_cfg.get("c", 2.0)),
            class_weight=stack_cfg.get("class_weight", "balanced"),
        )
    if stack_name == "bilstm":
        raise ValueError(
            "bilstm stack is trained via dedicated pipeline. "
            "Use scripts/train_lstm_category.py."
        )
    if stack_name == "bert":
        raise ValueError(
            "bert/distilbert stack is trained via dedicated pipeline. "
            "Use scripts/train_transformer_category.py."
        )
    raise ValueError(f"Unsupported NLP stack: {stack_name}")
