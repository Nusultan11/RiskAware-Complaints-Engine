from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from risk_aware.evaluation.category_metrics import macro_f1

TEST_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("artifacts/category/model.joblib")
REPORT_PATH = Path("reports/metrics/category_metrics.json")
TEXT_COL = "consumer_complaint_narrative"
TARGET_COL = "category"


def main() -> None:
    test_df = pd.read_csv(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    texts = test_df[TEXT_COL].fillna("").tolist()
    proba = model.predict_proba(texts)
    pred_idx = np.argmax(proba, axis=1)
    y_pred = np.array([model.labels[i] for i in pred_idx], dtype=object)
    y_true = test_df[TARGET_COL].astype(str).to_numpy()

    metrics = {"category_macro_f1": macro_f1(y_true=y_true, y_pred=y_pred)}

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"category_macro_f1: {metrics['category_macro_f1']:.6f}")
    print(f"Saved metrics to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
