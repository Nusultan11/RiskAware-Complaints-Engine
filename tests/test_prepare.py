import pandas as pd

from risk_aware.data.prepare import build_category_slice


def test_build_category_slice_creates_target() -> None:
    raw = pd.DataFrame(
        {
            "consumer_complaint_narrative": [
                "this complaint narrative has enough length",
                "another long complaint narrative for testing",
            ],
            "product": ["Mortgage", "Debt collection"],
            "issue": ["Servicing", "Disclosure verification of debt"],
        }
    )
    prepared, stats = build_category_slice(raw)
    assert "category" in prepared.columns
    assert prepared["category"].iloc[0] == "Mortgage|Servicing"
    assert stats["rows_raw"] == 2
