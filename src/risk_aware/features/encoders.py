from __future__ import annotations

from typing import Sequence


def encode_labels(labels: Sequence[str]) -> list[str]:
    return sorted({str(label) for label in labels})
