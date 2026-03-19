from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from risk_aware.evaluation.category_metrics import macro_f1

TEST_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("artifacts/category/model.joblib")
REPORT_PATH = Path("reports/metrics/category_metrics_test.json")
TEXT_COL = "consumer_complaint_narrative"
TARGET_COL = "category"


def _resolve_pred_labels(model, proba: np.ndarray) -> np.ndarray:
    pred_idx = np.argmax(proba, axis=1)

    # Preferred source of class order: fitted sklearn classifier classes_.
    clf = getattr(getattr(model, "pipeline", None), "named_steps", {}).get("clf")
    if clf is not None and hasattr(clf, "classes_"):
        classes = np.asarray(clf.classes_, dtype=object)
        return classes[pred_idx]

    # Fallback to model labels only if dimensions are compatible.
    labels = np.asarray(getattr(model, "labels", []), dtype=object)
    if labels.size == proba.shape[1]:
        return labels[pred_idx]

    raise ValueError("Cannot resolve class mapping for predicted probabilities.")


def main() -> None:
    test_df = pd.read_csv(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    texts = test_df[TEXT_COL].fillna("").tolist()
    proba = model.predict_proba(texts)
    y_pred = _resolve_pred_labels(model, proba)
    y_true = test_df[TARGET_COL].astype(str).to_numpy()

    metrics = {"category_macro_f1": macro_f1(y_true=y_true, y_pred=y_pred)}

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"category_macro_f1: {metrics['category_macro_f1']:.6f}")
    print(f"Saved metrics to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
