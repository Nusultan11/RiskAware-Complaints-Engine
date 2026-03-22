from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_row_from_metrics(
    model_name: str,
    source: str,
    metrics: dict[str, Any],
    config_summary: str,
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "split": "test",
        "source": source,
        "accuracy": float(metrics.get("accuracy", float("nan"))),
        "macro_f1": float(metrics.get("macro_f1", float("nan"))),
        "weighted_f1": float(metrics.get("weighted_f1", float("nan"))),
        "config_summary": config_summary,
    }


def main() -> None:
    metrics_dir = Path("reports/metrics")
    final_dir = Path("reports/final")
    final_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    # Canonical metrics registry for final comparison table.
    metric_specs = [
        {
            "model_name": "tfidf_lr",
            "source": "category_metrics_test.json",
            "file": "category_metrics_test.json",
            "config_summary": "analyzer=word, ngram=(1,2), min_df=3, max_df=0.9, sublinear_tf=True",
        },
        {
            "model_name": "bilstm",
            "source": "category_lstm_summary.json:test",
            "file": "category_lstm_summary.json",
            "json_path": ["test"],
            "config_summary": "architecture=bilstm_only, max_length=384, neural_clean keeps contractions",
        },
        {
            "model_name": "distilbert_base_uncased",
            "source": "category_transformer_metrics_test.json",
            "file": "category_transformer_metrics_test.json",
            "config_summary": "max_length=256, lr=2e-5, epochs=3",
        },
        {
            "model_name": "distilbert_base_uncased_len256",
            "source": "category_transformer_len256_test.json",
            "file": "category_transformer_len256_test.json",
            "config_summary": "max_length=256, lr=2e-5, epochs=3",
        },
        {
            "model_name": "distilbert_base_uncased_len384",
            "source": "category_transformer_len384_test.json",
            "file": "category_transformer_len384_test.json",
            "config_summary": "max_length=384, lr=2e-5, epochs=3",
        },
    ]

    for spec in metric_specs:
        payload = _read_json(metrics_dir / spec["file"])
        if not payload:
            continue

        metrics = payload
        for key in spec.get("json_path", []):
            metrics = metrics.get(key, {}) if isinstance(metrics, dict) else {}
        if not isinstance(metrics, dict) or not metrics:
            continue

        rows.append(
            _build_row_from_metrics(
                model_name=str(spec["model_name"]),
                source=str(spec["source"]),
                metrics=metrics,
                config_summary=str(spec["config_summary"]),
            )
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="macro_f1", ascending=False).reset_index(drop=True)

    csv_path = final_dir / "model_comparison.csv"
    json_path = final_dir / "model_comparison.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
