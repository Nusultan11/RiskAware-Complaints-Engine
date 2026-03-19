from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from risk_aware.config import load_project_configs
from risk_aware.evaluation.category_metrics import compute_category_metrics
from risk_aware.pipelines.category_training import CategoryTrainer
from risk_aware.utils.seed import set_global_seed

app = typer.Typer(add_completion=False, help="Tune TF-IDF category baseline on train/val split.")


def _load_tfidf_experiments(category_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    experiments = category_cfg.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError("`experiments` in category config must be a list.")

    filtered: list[dict[str, Any]] = []
    for exp in experiments:
        if not isinstance(exp, dict):
            continue
        if str(exp.get("stack", "")).strip() != "tfidf_lr":
            continue
        params = exp.get("params", {})
        if not isinstance(params, dict):
            continue
        name = str(exp.get("name", "")).strip()
        if not name:
            continue
        filtered.append({"name": name, "stack": "tfidf_lr", "params": params})

    if not filtered:
        raise ValueError("No tfidf_lr experiments found in configs/category.yaml")
    return filtered


@app.command()
def run(config_dir: str = "configs") -> None:
    cfg = load_project_configs(config_dir=config_dir)
    base_cfg = cfg["base"]
    category_cfg = cfg["category"]

    set_global_seed(int(base_cfg.get("project", {}).get("seed", 42)))

    text_col = str(base_cfg["data"]["text_column"])
    category_col = str(base_cfg["data"]["category_column"])

    train_path = Path("data/processed/train.csv")
    val_path = Path("data/processed/val.csv")
    if not train_path.exists():
        raise typer.BadParameter(f"Train data not found: {train_path}")
    if not val_path.exists():
        raise typer.BadParameter(f"Validation data not found: {val_path}")

    train_df = pd.read_csv(train_path).dropna(subset=[text_col, category_col]).copy()
    val_df = pd.read_csv(val_path).dropna(subset=[text_col, category_col]).copy()
    labels = sorted(train_df[category_col].astype(str).unique().tolist())

    experiments = _load_tfidf_experiments(category_cfg)
    typer.echo(f"Found {len(experiments)} tfidf_lr experiments.")

    results: list[dict[str, Any]] = []
    for idx, exp in enumerate(experiments, start=1):
        exp_name = exp["name"]
        exp_cfg = exp["params"]

        typer.echo(f"[{idx}/{len(experiments)}] Training: {exp_name}")
        trainer = CategoryTrainer(stack_name="tfidf_lr", stack_cfg=exp_cfg, labels=labels)

        start = time.perf_counter()
        model = trainer.train(train_df=train_df, text_col=text_col, label_col=category_col)
        fit_time_sec = time.perf_counter() - start

        y_true = val_df[category_col].astype(str).tolist()
        y_pred = model.predict(val_df[text_col].fillna("").tolist())
        metrics = compute_category_metrics(y_true=y_true, y_pred=y_pred)

        tfidf = model.pipeline.named_steps["tfidf"]
        n_features = int(len(getattr(tfidf, "vocabulary_", {})))

        results.append(
            {
                "experiment_name": exp_name,
                "macro_f1": float(metrics["macro_f1"]),
                "accuracy": float(metrics["accuracy"]),
                "n_features": n_features,
                "fit_time_sec": round(fit_time_sec, 3),
                "params": exp_cfg,
            }
        )

    leaderboard = pd.DataFrame(results).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    best = leaderboard.iloc[0].to_dict()

    metrics_dir = Path("reports/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    leaderboard[["experiment_name", "macro_f1", "accuracy", "n_features", "fit_time_sec"]].to_csv(
        metrics_dir / "category_tuning_results.csv", index=False
    )

    with (metrics_dir / "category_tuning_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    best_payload = {
        "experiment_name": best["experiment_name"],
        "macro_f1": float(best["macro_f1"]),
        "accuracy": float(best["accuracy"]),
        "n_features": int(best["n_features"]),
        "fit_time_sec": float(best["fit_time_sec"]),
        "params": best["params"],
    }
    with (metrics_dir / "category_tuning_best.json").open("w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2, ensure_ascii=False)

    typer.echo("Tuning completed.")
    typer.echo(f"Best experiment: {best_payload['experiment_name']}")
    typer.echo(f"Best macro_f1: {best_payload['macro_f1']:.6f}")
    typer.echo(f"Saved leaderboard to: {metrics_dir / 'category_tuning_results.csv'}")


if __name__ == "__main__":
    app()
