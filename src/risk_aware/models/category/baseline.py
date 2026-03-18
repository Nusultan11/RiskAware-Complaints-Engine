from __future__ import annotations

from risk_aware.models.category.stacks import TfidfLogRegTextStack


def build_tfidf_baseline(labels: list[str]) -> TfidfLogRegTextStack:
    return TfidfLogRegTextStack(labels=labels)
