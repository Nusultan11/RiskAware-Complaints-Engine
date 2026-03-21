from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
REQUIRED_COLUMNS = ["consumer_complaint_narrative", "product", "issue"]
FINAL_COLUMNS = ["complaint_id", "consumer_complaint_narrative", "text_key", "product", "issue", "category"]
MIN_TEXT_LEN = 20
MIN_CLASS_COUNT = 5

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
    n_classes_before_min_count = int(frame["category"].nunique())

    category_counts = frame["category"].value_counts()
    valid_categories = category_counts[category_counts >= MIN_CLASS_COUNT].index
    frame = frame[frame["category"].isin(valid_categories)].copy()

    n_classes_after_min_count = int(frame["category"].nunique())
    min_class_size_after_filter = int(frame["category"].value_counts().min()) if not frame.empty else 0
    rows_after_class_filter = len(frame)

    # Remove ambiguous groups where one normalized text maps to multiple categories.
    rows_before_conflict_filter = len(frame)
    category_per_text_key = frame.groupby("text_key")["category"].nunique()
    conflicting_text_keys = category_per_text_key[category_per_text_key > 1].index
    n_conflicting_text_keys_before = int(len(conflicting_text_keys))
    if n_conflicting_text_keys_before > 0:
        frame = frame[~frame["text_key"].isin(conflicting_text_keys)].copy()
    rows_after_conflict_filter = len(frame)
    dropped_conflict_rows = rows_before_conflict_filter - rows_after_conflict_filter

    # Re-apply class threshold after conflict cleanup.
    category_counts_post_conflict = frame["category"].value_counts()
    valid_categories_post_conflict = category_counts_post_conflict[
        category_counts_post_conflict >= MIN_CLASS_COUNT
    ].index
    frame = frame[frame["category"].isin(valid_categories_post_conflict)].copy()
    rows_after_class_refilter = len(frame)
    n_classes_after_refilter = int(frame["category"].nunique())
    min_class_size_after_refilter = int(frame["category"].value_counts().min()) if not frame.empty else 0

    remaining_conflicts = frame.groupby("text_key")["category"].nunique()
    n_conflicting_text_keys_after = int((remaining_conflicts > 1).sum())

    rows_after_text_key = len(frame)
    frame = frame.reset_index(drop=True)
    frame["complaint_id"] = "cfpb_" + frame.index.astype(str)
    stats = {
        "rows_raw": rows_raw,
        "rows_after_na": rows_after_na,
        "rows_after_min_len": rows_after_len,
        "rows_after_class_filter": rows_after_class_filter,
        "rows_before_conflict_filter": rows_before_conflict_filter,
        "rows_after_conflict_filter": rows_after_conflict_filter,
        "dropped_conflict_rows": dropped_conflict_rows,
        "rows_after_class_refilter": rows_after_class_refilter,
        "rows_after_text_key": rows_after_text_key,
        "dropped_by_text_dedup": 0,
        "n_classes_before_min_count": n_classes_before_min_count,
        "n_classes_after_min_count": n_classes_after_min_count,
        "min_class_size_after_filter": min_class_size_after_filter,
        "n_classes_after_refilter": n_classes_after_refilter,
        "min_class_size_after_refilter": min_class_size_after_refilter,
        "n_conflicting_text_keys_before": n_conflicting_text_keys_before,
        "n_conflicting_text_keys_after": n_conflicting_text_keys_after,
        "min_class_count_threshold": MIN_CLASS_COUNT,
    }
    return frame[FINAL_COLUMNS].copy(), stats


def _assert_no_text_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_keys = set(train_df["text_key"].astype(str))
    val_keys = set(val_df["text_key"].astype(str))
    test_keys = set(test_df["text_key"].astype(str))

    overlap_tv = len(train_keys.intersection(val_keys))
    overlap_tt = len(train_keys.intersection(test_keys))
    overlap_vt = len(val_keys.intersection(test_keys))

    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            "Group leakage detected between splits by text_key: "
            f"train-val={overlap_tv}, train-test={overlap_tt}, val-test={overlap_vt}"
        )


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_cols = {"category", "text_key"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"split_dataset requires columns: {sorted(required_cols)}")

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    y = df["category"].astype(str)
    groups = df["text_key"].astype(str)
    train_val_idx, test_idx = next(outer.split(df, y, groups))

    train_val_df = df.iloc[train_val_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    y_inner = train_val_df["category"].astype(str)
    groups_inner = train_val_df["text_key"].astype(str)
    train_idx, val_idx = next(inner.split(train_val_df, y_inner, groups_inner))

    train_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].copy().reset_index(drop=True)

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
