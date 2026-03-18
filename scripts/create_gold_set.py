from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(add_completion=False, help="Create a human-labeling gold set from processed splits.")

INPUT_PATH = Path("data/processed/train.csv")
OUTPUT_DIR = Path("reports/labeling")
OUTPUT_CSV = OUTPUT_DIR / "gold_candidates.csv"
MANIFEST_JSON = OUTPUT_DIR / "gold_manifest.json"

LABEL_COLS = [
    "gold_legal_threat",
    "gold_priority",
    "annotator_id",
    "confidence",
    "review_notes",
]


def _sample_frame(df: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    # Keep representation of all weak-priority classes for a balanced first gold set.
    if "priority" not in df.columns:
        return df.sample(n=min(n_samples, len(df)), random_state=seed)

    per_class = max(1, n_samples // max(1, df["priority"].nunique()))
    parts: list[pd.DataFrame] = []
    for _, group in df.groupby("priority", dropna=False):
        take = min(per_class, len(group))
        parts.append(group.sample(n=take, random_state=seed))
    sampled = pd.concat(parts, axis=0).drop_duplicates(subset=["complaint_id"])
    if len(sampled) < n_samples:
        remaining = df[~df["complaint_id"].isin(sampled["complaint_id"])]
        extra_n = min(n_samples - len(sampled), len(remaining))
        if extra_n > 0:
            sampled = pd.concat(
                [sampled, remaining.sample(n=extra_n, random_state=seed)],
                axis=0,
            )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@app.command()
def run(n_samples: int = 1500, seed: int = 42) -> None:
    if not INPUT_PATH.exists():
        raise typer.BadParameter(f"Input split not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    required = [
        "complaint_id",
        "consumer_complaint_narrative",
        "product",
        "issue",
        "category",
        "legal_threat",
        "priority",
        "amount",
        "repeat_count",
        "client_type",
        "account_age",
        "channel",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise typer.BadParameter(f"Missing required columns in input split: {missing}")

    sample_df = _sample_frame(df, n_samples=n_samples, seed=seed).copy()
    for col in LABEL_COLS:
        sample_df[col] = ""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(OUTPUT_CSV, index=False)

    manifest = {
        "input_split": str(INPUT_PATH),
        "output_csv": str(OUTPUT_CSV),
        "n_rows": int(len(sample_df)),
        "seed": seed,
        "label_schema": {
            "gold_legal_threat": "int: 0/1",
            "gold_priority": "str: P1/P2/P3",
            "annotator_id": "str",
            "confidence": "int: 1..5",
            "review_notes": "str",
        },
        "class_balance_in_sample": sample_df["priority"].value_counts(dropna=False).to_dict(),
    }
    with MANIFEST_JSON.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    typer.echo(f"Gold candidates created: {OUTPUT_CSV}")
    typer.echo(f"Manifest saved: {MANIFEST_JSON}")


if __name__ == "__main__":
    app()

