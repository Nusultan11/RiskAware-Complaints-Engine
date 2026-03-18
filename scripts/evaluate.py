from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

TEST_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("artifacts/models/category_model.joblib")
REPORT_PATH = Path("reports/metrics/category_metrics.json")

TEXT_COL = "consumer_complaint_narrative"
CATEGORY_COL = "category"


def evaluate() -> dict[str, float]:
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test split not found: {TEST_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Category model not found: {MODEL_PATH}")

    test_df = pd.read_csv(TEST_PATH)
    category_model = joblib.load(MODEL_PATH)

    texts = test_df[TEXT_COL].fillna("").tolist()
    category_proba = category_model.predict_proba(texts)
    pred_idx = np.argmax(category_proba, axis=1)
    pred_labels = np.array([category_model.labels[i] for i in pred_idx], dtype=object)

    category_macro_f1 = float(
        f1_score(
            y_true=test_df[CATEGORY_COL].astype(str),
            y_pred=pred_labels,
            average="macro",
        )
    )
    return {"category_macro_f1": category_macro_f1}


def save_metrics(metrics: dict[str, float]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def main() -> None:
    metrics = evaluate()
    save_metrics(metrics)
    print(f"category_macro_f1: {metrics['category_macro_f1']:.6f}")
    print(f"Saved metrics to: {REPORT_PATH}")


if __name__ == "__main__":
    main()

