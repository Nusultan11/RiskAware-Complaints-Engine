from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from risk_aware.config import load_project_configs
from risk_aware.pipelines.category_training import CategoryTrainer
from risk_aware.utils.seed import set_global_seed
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

    set_global_seed(int(base_cfg.get("project", {}).get("seed", 42)))

    train_path = Path("data/processed/train.csv")
    if not train_path.exists():
        raise typer.BadParameter(f"Train data not found: {train_path}")

    train_df = pd.read_csv(train_path)
    text_col = str(base_cfg["data"]["text_column"])
    category_col = str(base_cfg["data"]["category_column"])

    train_df = train_df.dropna(subset=[text_col, category_col]).copy()
    labels = sorted(train_df[category_col].astype(str).unique().tolist())
    stack_name, stack_cfg = _extract_stack_cfg(category_cfg)

    trainer = CategoryTrainer(stack_name=stack_name, stack_cfg=stack_cfg, labels=labels)
    model = trainer.train(train_df, text_col=text_col, label_col=category_col)

    output_dir = Path("artifacts/category")
    save_artifact(model, output_dir / "model.joblib")
    save_artifact(model.pipeline.named_steps["tfidf"], output_dir / "vectorizer.joblib")
    save_artifact(model.labels, output_dir / "label_encoder.joblib")

    metadata = {
        "stack_name": stack_name,
        "n_train_rows": int(len(train_df)),
        "n_labels": int(len(labels)),
        "text_column": text_col,
        "target_column": category_col,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    typer.echo("Category training completed.")
    typer.echo(f"Saved artifacts to: {output_dir}")


if __name__ == "__main__":
    app()
