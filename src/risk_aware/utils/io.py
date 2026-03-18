from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    payload = Path(path)
    with payload.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {payload}")
    return data


def write_json(data: dict[str, Any], path: str | Path) -> None:
    payload = Path(path)
    payload.parent.mkdir(parents=True, exist_ok=True)
    with payload.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
