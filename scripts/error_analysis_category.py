from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import typer
from sklearn.metrics import precision_recall_fscore_support

app = typer.Typer(add_completion=False, help="Tail-class error analysis for category model.")

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("artifacts/category/model.joblib")
OUT_DIR = Path("reports/analysis")

ID_COL = "complaint_id"
TEXT_COL = "consumer_complaint_narrative"
TARGET_COL = "category"


def _resolve_classes(model) -> np.ndarray:
    clf = getattr(getattr(model, "pipeline", None), "named_steps", {}).get("clf")
    if clf is not None and hasattr(clf, "classes_"):
        return np.asarray(clf.classes_, dtype=object)

    labels = np.asarray(getattr(model, "labels", []), dtype=object)
    if labels.size > 0:
        return labels

    raise ValueError("Unable to resolve model classes.")


def _predict_with_proba(model, texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proba = model.predict_proba(texts)
    classes = _resolve_classes(model)
    if proba.shape[1] != classes.size:
        raise ValueError("Probability shape does not match number of classes.")
    pred_idx = np.argmax(proba, axis=1)
    y_pred = classes[pred_idx]
    return y_pred, proba, classes


@app.command()
def run(
    tail_threshold: int = 100,
    worst_k: int = 20,
    samples_per_class: int = 5,
) -> None:
    if not TRAIN_PATH.exists():
        raise typer.BadParameter(f"Train split not found: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise typer.BadParameter(f"Test split not found: {TEST_PATH}")
    if not MODEL_PATH.exists():
        raise typer.BadParameter(f"Model artifact not found: {MODEL_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    train_counts = train_df[TARGET_COL].astype(str).value_counts()
    tail_classes = train_counts[train_counts < tail_threshold].index.astype(str).tolist()
    if not tail_classes:
        raise ValueError(f"No tail classes found with threshold < {tail_threshold}.")

    texts = test_df[TEXT_COL].fillna("").astype(str).tolist()
    y_true = test_df[TARGET_COL].astype(str).to_numpy()
    y_pred, proba, classes = _predict_with_proba(model, texts)
    class_to_idx = {str(c): i for i, c in enumerate(classes.tolist())}

    p, r, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=tail_classes,
        zero_division=0,
    )
    metrics_df = pd.DataFrame(
        {
            "category": tail_classes,
            "support_test": support.astype(int),
            "precision": p,
            "recall": r,
            "f1": f1,
        }
    )

    fn_counts = []
    fp_counts = []
    for cls in tail_classes:
        fn_counts.append(int(np.sum((y_true == cls) & (y_pred != cls))))
        fp_counts.append(int(np.sum((y_true != cls) & (y_pred == cls))))
    metrics_df["fn_count"] = fn_counts
    metrics_df["fp_count"] = fp_counts

    worst_df = (
        metrics_df[metrics_df["support_test"] > 0]
        .sort_values(["f1", "support_test"], ascending=[True, False])
        .head(worst_k)
        .reset_index(drop=True)
    )
    worst_classes = worst_df["category"].tolist()

    fn_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []

    test_frame = test_df[[ID_COL, TEXT_COL]].copy()
    test_frame["y_true"] = y_true
    test_frame["y_pred"] = y_pred
    test_frame["pred_confidence"] = np.max(proba, axis=1)

    for cls in worst_classes:
        cls_idx = class_to_idx.get(cls)
        if cls_idx is None:
            continue

        # False negatives for this class.
        fn_mask = (test_frame["y_true"] == cls) & (test_frame["y_pred"] != cls)
        fn_cls = test_frame.loc[fn_mask].copy()
        if not fn_cls.empty:
            idx = fn_cls.index.to_numpy()
            fn_cls["true_class_proba"] = proba[idx, cls_idx]
            fn_cls = fn_cls.sort_values("true_class_proba", ascending=True).head(samples_per_class)
            for _, row in fn_cls.iterrows():
                fn_rows.append(
                    {
                        "focus_class": cls,
                        "complaint_id": row[ID_COL],
                        "y_true": row["y_true"],
                        "y_pred": row["y_pred"],
                        "pred_confidence": float(row["pred_confidence"]),
                        "true_class_proba": float(row["true_class_proba"]),
                        "text": row[TEXT_COL],
                    }
                )

        # False positives for this class.
        fp_mask = (test_frame["y_true"] != cls) & (test_frame["y_pred"] == cls)
        fp_cls = test_frame.loc[fp_mask].copy()
        if not fp_cls.empty:
            idx = fp_cls.index.to_numpy()
            fp_cls["predicted_class_proba"] = proba[idx, cls_idx]
            fp_cls = fp_cls.sort_values("predicted_class_proba", ascending=False).head(samples_per_class)
            for _, row in fp_cls.iterrows():
                fp_rows.append(
                    {
                        "focus_class": cls,
                        "complaint_id": row[ID_COL],
                        "y_true": row["y_true"],
                        "y_pred": row["y_pred"],
                        "pred_confidence": float(row["pred_confidence"]),
                        "predicted_class_proba": float(row["predicted_class_proba"]),
                        "text": row[TEXT_COL],
                    }
                )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.sort_values("f1", ascending=True).to_csv(OUT_DIR / "tail_class_metrics.csv", index=False)
    worst_df.to_csv(OUT_DIR / "tail_worst20_metrics.csv", index=False)
    pd.DataFrame(fn_rows).to_csv(OUT_DIR / "tail_worst20_fn_samples.csv", index=False)
    pd.DataFrame(fp_rows).to_csv(OUT_DIR / "tail_worst20_fp_samples.csv", index=False)

    summary = {
        "tail_threshold": tail_threshold,
        "worst_k": worst_k,
        "samples_per_class": samples_per_class,
        "n_tail_classes_train": int(len(tail_classes)),
        "n_tail_classes_with_test_support": int((metrics_df["support_test"] > 0).sum()),
        "mean_tail_f1": float(metrics_df["f1"].mean()),
        "worst_classes": worst_classes,
    }
    with (OUT_DIR / "tail_error_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    typer.echo("Tail error analysis completed.")
    typer.echo(f"Saved metrics to: {OUT_DIR / 'tail_worst20_metrics.csv'}")
    typer.echo(f"Saved FN samples to: {OUT_DIR / 'tail_worst20_fn_samples.csv'}")
    typer.echo(f"Saved FP samples to: {OUT_DIR / 'tail_worst20_fp_samples.csv'}")


if __name__ == "__main__":
    app()
