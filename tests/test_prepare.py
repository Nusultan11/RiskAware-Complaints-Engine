import json
from pathlib import Path

import pandas as pd

import risk_aware.data.prepare as prepare


def _row(text: str, product: str, issue: str) -> dict[str, str]:
    return {
        "consumer_complaint_narrative": text,
        "product": product,
        "issue": issue,
    }


def _long_text(prefix: str, i: int) -> str:
    return f"{prefix} complaint narrative sample {i} with enough text length for filtering checks"


def test_build_category_slice_applies_min_count_and_conflict_filters() -> None:
    rows: list[dict[str, str]] = []

    # Class A: 7 rows (includes one row with text duplicated inside the same class).
    for i in range(6):
        rows.append(_row(_long_text("A", i), "Mortgage", "Servicing"))
    rows.append(_row("duplicate same-class text key long enough for category A", "Mortgage", "Servicing"))

    # Class B: 6 rows before conflict removal.
    for i in range(5):
        rows.append(_row(_long_text("B", i), "Debt collection", "Disclosure verification of debt"))
    rows.append(
        _row(
            "shared conflict key complaint narrative appears in two categories and must be dropped",
            "Debt collection",
            "Disclosure verification of debt",
        )
    )

    # Add the conflicting twin in class A.
    rows.append(
        _row(
            "shared conflict key complaint narrative appears in two categories and must be dropped",
            "Mortgage",
            "Servicing",
        )
    )

    # Class C: below threshold and must be removed by MIN_CLASS_COUNT=5.
    for i in range(4):
        rows.append(_row(_long_text("C", i), "Credit card", "APR or interest rate"))

    raw = pd.DataFrame(rows)
    prepared_df, stats = prepare.build_category_slice(raw)

    assert "category" in prepared_df.columns
    assert "text_key" in prepared_df.columns
    assert set(prepared_df["category"].unique()) == {
        "Mortgage|Servicing",
        "Debt collection|Disclosure verification of debt",
    }
    assert prepared_df["category"].value_counts().min() >= 5
    assert stats["min_class_count_threshold"] == 5
    assert stats["n_conflicting_text_keys_before"] == 1
    assert stats["n_conflicting_text_keys_after"] == 0
    assert stats["dropped_conflict_rows"] == 2
    assert stats["dropped_by_text_dedup"] == 0


def test_split_dataset_has_no_text_key_leakage() -> None:
    rows: list[dict[str, str]] = []

    # Build 3 classes with enough unique text_key groups for StratifiedGroupKFold(n_splits=5).
    for i in range(10):
        rows.append(_row(_long_text("M", i), "Mortgage", "Servicing"))
    for i in range(10):
        rows.append(_row(_long_text("D", i), "Debt collection", "Disclosure verification of debt"))
    for i in range(10):
        rows.append(_row(_long_text("P", i), "Payday loan", "Can't contact lender"))

    raw = pd.DataFrame(rows)
    prepared_df, _ = prepare.build_category_slice(raw)
    train_df, val_df, test_df = prepare.split_dataset(prepared_df)

    train_keys = set(train_df["text_key"])
    val_keys = set(val_df["text_key"])
    test_keys = set(test_df["text_key"])

    assert len(train_df) + len(val_df) + len(test_df) == len(prepared_df)
    assert train_keys.isdisjoint(val_keys)
    assert train_keys.isdisjoint(test_keys)
    assert val_keys.isdisjoint(test_keys)


def test_metadata_records_group_split_strategy(tmp_path: Path, monkeypatch) -> None:
    train_df = pd.DataFrame(
        {
            "complaint_id": ["a1", "a2"],
            "consumer_complaint_narrative": ["t1 long enough text narrative", "t2 long enough text narrative"],
            "text_key": ["t1", "t2"],
            "product": ["Mortgage", "Mortgage"],
            "issue": ["Servicing", "Servicing"],
            "category": ["Mortgage|Servicing", "Mortgage|Servicing"],
        }
    )
    val_df = train_df.iloc[[0]].copy()
    test_df = train_df.iloc[[1]].copy()

    monkeypatch.setattr(prepare, "PROCESSED_DIR", tmp_path)
    prepare.save_metadata(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        prepare_stats={"rows_raw": 2},
    )

    payload = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert payload["split_strategy"] == "stratified_group_kfold_by_text_key"

