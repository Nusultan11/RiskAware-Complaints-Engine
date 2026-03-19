from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
REQUIRED_COLUMNS = ["consumer_complaint_narrative", "product", "issue"]
FINAL_COLUMNS = ["complaint_id", "consumer_complaint_narrative", "product", "issue", "category"]
MIN_TEXT_LEN = 20

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cfpb_complaints.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _text_key(text: str) -> str:
    normalized = str(text).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return df


def build_category_slice(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = df[REQUIRED_COLUMNS].copy()
    rows_raw = len(frame)
    frame = frame.dropna(subset=["consumer_complaint_narrative", "product", "issue"])
    rows_after_na = len(frame)
    text = frame["consumer_complaint_narrative"].astype(str).str.strip()
    frame = frame[text.str.len() > MIN_TEXT_LEN].copy()
    rows_after_len = len(frame)
    frame["category"] = frame["product"].astype(str) + "|" + frame["issue"].astype(str)
    frame["text_key"] = frame["consumer_complaint_narrative"].map(_text_key)
    frame = frame.drop_duplicates(subset=["text_key"], keep="first").copy()
    rows_after_dedup = len(frame)
    frame = frame.reset_index(drop=True)
    frame["complaint_id"] = "cfpb_" + frame.index.astype(str)
    stats = {
        "rows_raw": rows_raw,
        "rows_after_na": rows_after_na,
        "rows_after_min_len": rows_after_len,
        "rows_after_text_dedup": rows_after_dedup,
        "dropped_by_text_dedup": rows_after_len - rows_after_dedup,
    }
    return frame[FINAL_COLUMNS].copy(), stats


def _assert_no_text_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_text = set(train_df["consumer_complaint_narrative"].map(_text_key))
    val_text = set(val_df["consumer_complaint_narrative"].map(_text_key))
    test_text = set(test_df["consumer_complaint_narrative"].map(_text_key))

    overlap_tv = len(train_text.intersection(val_text))
    overlap_tt = len(train_text.intersection(test_text))
    overlap_vt = len(val_text.intersection(test_text))

    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            "Text leakage detected between splits: "
            f"train-val={overlap_tv}, train-test={overlap_tt}, val-test={overlap_vt}"
        )


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = df["category"].value_counts()

    # Classes with fewer than 3 samples cannot be safely stratified in a 70/15/15 split.
    rare_classes = counts[counts < 3].index
    rare_df = df[df["category"].isin(rare_classes)].copy()
    common_df = df[~df["category"].isin(rare_classes)].copy()

    if common_df.empty:
        raise ValueError("No classes eligible for stratified split after rare-class filtering.")

    train_common, temp_common = train_test_split(
        common_df,
        train_size=0.7,
        random_state=SEED,
        stratify=common_df["category"],
    )

    # For second split, only keep categories that still have at least 2 rows in temp.
    temp_counts = temp_common["category"].value_counts()
    temp_rare_classes = temp_counts[temp_counts < 2].index
    temp_rare_df = temp_common[temp_common["category"].isin(temp_rare_classes)].copy()
    temp_strat_df = temp_common[~temp_common["category"].isin(temp_rare_classes)].copy()

    if temp_strat_df.empty:
        raise ValueError("Validation/test stratified split has no eligible rows.")

    val_df, test_df = train_test_split(
        temp_strat_df,
        train_size=0.5,
        random_state=SEED,
        stratify=temp_strat_df["category"],
    )

    # Keep unsplittable tails in train/val deterministically (never in test).
    train_df = pd.concat([train_common, rare_df], ignore_index=True)
    val_df = pd.concat([val_df, temp_rare_df], ignore_index=True)

    # Deterministic shuffling for stable row order.
    train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    _assert_no_text_overlap(train_df, val_df, test_df)
    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)


def save_metadata(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    prepare_stats: dict[str, int],
) -> None:
    split_classes = {
        "train": int(train_df["category"].nunique()),
        "val": int(val_df["category"].nunique()),
        "test": int(test_df["category"].nunique()),
    }
    metadata = {
        "seed": SEED,
        "task": "category_only",
        "min_text_len": MIN_TEXT_LEN,
        "split_strategy": "stratified_with_tail_handling",
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "columns": FINAL_COLUMNS,
        "n_categories_by_split": split_classes,
        "prepare_stats": prepare_stats,
    }
    with (PROCESSED_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    df = load_raw_data()
    df, prepare_stats = build_category_slice(df)
    train_df, val_df, test_df = split_dataset(df)
    save_splits(train_df, val_df, test_df)
    save_metadata(train_df, val_df, test_df, prepare_stats=prepare_stats)


if __name__ == "__main__":
    main()
