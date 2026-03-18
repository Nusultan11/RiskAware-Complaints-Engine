from __future__ import annotations

from typing import Any

from risk_aware.models.base import TextClassifier
from risk_aware.models.category.stacks import (
    BertTextStack,
    BiLSTMTextStack,
    TfidfLogRegTextStack,
)


def build_text_stack(
    stack_name: str,
    labels: list[str],
    stack_cfg: dict[str, Any],
) -> TextClassifier:
    """
    Factory for building text classification stacks.
    """

    if stack_name == "tfidf_lr":
        ngram = stack_cfg.get("ngram_range", [1, 2])

        return TfidfLogRegTextStack(
            labels=labels,
            max_features=int(stack_cfg.get("max_features", 50_000)),
            ngram_range=(int(ngram[0]), int(ngram[1])),
            c=float(stack_cfg.get("c", 2.0)),
            class_weight=stack_cfg.get("class_weight", "balanced"),
        )

    if stack_name == "bilstm":
        return BiLSTMTextStack(labels=labels)

    if stack_name == "bert":
        return BertTextStack(labels=labels)

    raise ValueError(f"Unsupported NLP stack: {stack_name}")