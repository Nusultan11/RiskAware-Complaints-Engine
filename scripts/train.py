from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import typer

from risk_aware.config import load_project_configs
from risk_aware.pipelines.training import CategoryTrainer
from risk_aware.utils.serialization import save_artifact

app = typer.Typer(add_completion=False, help="Train category model for RiskAware Complaints Engine.")


def _extract_stack_cfg(task_cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    stack_name = str(task_cfg.get("stack", "tfidf_lr"))
    stacks = task_cfg.get("stacks", {})
    stack_cfg = stacks.get(stack_name, {})
    if not isinstance(stack_cfg, dict):
        raise ValueError(f"Stack config for {stack_name} must be a mapping.")
    return stack_name, stack_cfg


@app.command()
def run(config_dir: str = "configs") -> None:
    cfg = load_project_configs(config_dir=config_dir)
    base_cfg = cfg["base"]
    category_cfg = cfg["category"]

    processed_dir = Path("data/processed")
    train_path = processed_dir / "train.csv"
    if not train_path.exists():
        raise typer.BadParameter(f"Train data not found: {train_path}")

    train_df = pd.read_csv(train_path)
    text_col = str(base_cfg["data"]["text_column"])
    category_col = str(base_cfg["data"]["category_column"])

    if text_col not in train_df.columns:
        raise typer.BadParameter(f"Text column not found in train data: {text_col}")
    if category_col not in train_df.columns:
        raise typer.BadParameter(f"Category column not found in train data: {category_col}")

    train_df = train_df.dropna(subset=[text_col, category_col]).copy()
    category_labels = sorted(train_df[category_col].astype(str).unique())
    category_stack_name, category_stack_cfg = _extract_stack_cfg(category_cfg)

    category_trainer = CategoryTrainer(
        stack_name=category_stack_name,
        stack_cfg=category_stack_cfg,
        labels=category_labels,
    )
    category_model = category_trainer.train(train_df, text_col=text_col, label_col=category_col)

    model_dir = Path("artifacts/models")
    save_artifact(category_model, model_dir / "category_model.joblib")

    typer.echo("Category training completed.")
    typer.echo(f"Saved category model to: {model_dir / 'category_model.joblib'}")


if __name__ == "__main__":
    app()

