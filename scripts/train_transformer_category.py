from __future__ import annotations

import typer

from risk_aware.pipelines.category_transformer_training import run_category_transformer_training

app = typer.Typer(add_completion=False, help="Train DistilBERT category model on processed splits.")


@app.command()
def run(config_dir: str = "configs") -> None:
    run_category_transformer_training(config_dir=config_dir)


if __name__ == "__main__":
    app()
