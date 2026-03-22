from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    metrics_dir = Path("reports/metrics")
    final_dir = Path("reports/final")
    final_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    tfidf = _read_json(metrics_dir / "category_metrics_test.json")
    if tfidf:
        rows.append(
            {
                "model_name": "tfidf_lr",
                "split": "test",
                "accuracy": float(tfidf.get("accuracy", float("nan"))),
                "macro_f1": float(tfidf.get("macro_f1", float("nan"))),
                "weighted_f1": float(tfidf.get("weighted_f1", float("nan"))),
                "config_summary": "analyzer=word, ngram=(1,2), min_df=5, sublinear_tf=True",
            }
        )

    lstm = _read_json(metrics_dir / "category_lstm_metrics_test.json")
    if lstm:
        lstm_summary = _read_json(metrics_dir / "category_lstm_summary.json")
        rows.append(
            {
                "model_name": "bilstm",
                "split": "test",
                "accuracy": float(lstm.get("accuracy", float("nan"))),
                "macro_f1": float(lstm.get("macro_f1", float("nan"))),
                "weighted_f1": float(lstm_summary.get("test", {}).get("weighted_f1", float("nan"))),
                "config_summary": "max_length=384, neural_clean with contractions",
            }
        )

    distilbert_256 = _read_json(metrics_dir / "category_transformer_len256_test.json")
    if distilbert_256:
        rows.append(
            {
                "model_name": "distilbert_base_uncased_len256",
                "split": "test",
                "accuracy": float(distilbert_256.get("accuracy", float("nan"))),
                "macro_f1": float(distilbert_256.get("macro_f1", float("nan"))),
                "weighted_f1": float(distilbert_256.get("weighted_f1", float("nan"))),
                "config_summary": "max_length=256, lr=2e-5, epochs=3",
            }
        )

    distilbert_384 = _read_json(metrics_dir / "category_transformer_len384_test.json")
    if distilbert_384:
        rows.append(
            {
                "model_name": "distilbert_base_uncased_len384",
                "split": "test",
                "accuracy": float(distilbert_384.get("accuracy", float("nan"))),
                "macro_f1": float(distilbert_384.get("macro_f1", float("nan"))),
                "weighted_f1": float(distilbert_384.get("weighted_f1", float("nan"))),
                "config_summary": "max_length=384, lr=2e-5, epochs=3",
            }
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
