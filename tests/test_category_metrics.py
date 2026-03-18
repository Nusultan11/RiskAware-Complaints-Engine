from risk_aware.evaluation.category_metrics import macro_f1


def test_macro_f1_perfect_case() -> None:
    y_true = ["A", "B", "C"]
    y_pred = ["A", "B", "C"]
    assert macro_f1(y_true, y_pred) == 1.0
