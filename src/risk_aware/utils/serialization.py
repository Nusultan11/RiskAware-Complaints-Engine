from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def save_artifact(obj: Any, path: str | Path) -> None:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, artifact_path)


def load_artifact(path: str | Path) -> Any:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    return joblib.load(artifact_path)

