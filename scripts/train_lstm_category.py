from __future__ import annotations

import typer

from risk_aware.models.category.bilstm import BiLSTMClassifier
from risk_aware.pipelines.category_lstm_training import (
    load_npz as _load_npz,
)
from risk_aware.pipelines.category_lstm_training import run_category_lstm_training

app = typer.Typer(add_completion=False, help="Train BiLSTM category model on preprocessed NPZ tensors.")


@app.command()
def run(config_dir: str = "configs", epochs: int | None = None) -> None:
    try:
        run_category_lstm_training(config_dir=config_dir, epochs=epochs)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


if __name__ == "__main__":
    app()
