from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
REQUIRED_COLUMNS = ["consumer_complaint_narrative", "product", "issue"]
FINAL_COLUMNS = ["complaint_id", "consumer_complaint_narrative", "product", "issue", "category"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cfpb_complaints.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return df


def build_category_slice(df: pd.DataFrame) -> pd.DataFrame:
    frame = df[REQUIRED_COLUMNS].copy()
    frame = frame.dropna(subset=["consumer_complaint_narrative", "product", "issue"])
    text = frame["consumer_complaint_narrative"].astype(str).str.strip()
    frame = frame[text.str.len() > 0].copy()
    frame["category"] = frame["product"].astype(str) + "|" + frame["issue"].astype(str)
    frame = frame.reset_index(drop=True)
    frame["complaint_id"] = "cfpb_" + frame.index.astype(str)
    return frame[FINAL_COLUMNS].copy()


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        train_df, temp_df = train_test_split(
            df,
            train_size=0.7,
            random_state=SEED,
            stratify=df["category"],
        )
        val_df, test_df = train_test_split(
            temp_df,
            train_size=0.5,
            random_state=SEED,
            stratify=temp_df["category"],
        )
    except ValueError:
        train_df, temp_df = train_test_split(df, train_size=0.7, random_state=SEED)
        val_df, test_df = train_test_split(temp_df, train_size=0.5, random_state=SEED)
    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)


def save_metadata(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    metadata = {
        "seed": SEED,
        "task": "category_only",
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "columns": FINAL_COLUMNS,
        "n_categories_train": int(train_df["category"].nunique()),
    }
    with (PROCESSED_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    df = load_raw_data()
    df = build_category_slice(df)
    train_df, val_df, test_df = split_dataset(df)
    save_splits(train_df, val_df, test_df)
    save_metadata(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
