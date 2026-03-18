from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config at {cfg_path} must be a mapping.")
    return payload


def load_project_configs(config_dir: str | Path = "configs") -> dict[str, dict[str, Any]]:
    base_dir = Path(config_dir)
    files = {
        "base": base_dir / "base.yaml",
        "category": base_dir / "category.yaml",
    }
    return {name: load_yaml(path) for name, path in files.items()}
