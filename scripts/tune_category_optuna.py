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

app = typer.Typer(add_completion=False, help="Optuna tuning for TF-IDF category baseline.")


@app.command()
def run(
    config_dir: str = "configs",
    n_trials: int = 30,
    timeout_sec: int = 3600,
) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is not installed. Run `pip install optuna` or reinstall project dependencies."
        ) from exc

    cfg = load_project_configs(config_dir=config_dir)
    base_cfg = cfg["base"]
    category_cfg = cfg["category"]

    seed = int(base_cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

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

    base_stack_cfg = category_cfg.get("stacks", {}).get("tfidf_lr", {})
    if not isinstance(base_stack_cfg, dict):
        raise ValueError("configs/category.yaml -> stacks.tfidf_lr must be a mapping.")

    def objective(trial: Any) -> float:
        stack_cfg = {
            "max_features": trial.suggest_categorical("max_features", [50000, 70000, 90000]),
            "ngram_range": [1, 2],
            "min_df": trial.suggest_int("min_df", 1, 3),
            "max_df": trial.suggest_categorical("max_df", [0.85, 0.90, 0.95]),
            "c": trial.suggest_float("c", 1.0, 12.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        }

        # Keep optional baseline fields if they exist.
        for k, v in base_stack_cfg.items():
            if k not in stack_cfg:
                stack_cfg[k] = v

        trainer = CategoryTrainer(stack_name="tfidf_lr", stack_cfg=stack_cfg, labels=labels)

        start = time.perf_counter()
        model = trainer.train(train_df=train_df, text_col=text_col, label_col=category_col)
        fit_time_sec = time.perf_counter() - start

        y_true = val_df[category_col].astype(str).tolist()
        y_pred = model.predict(val_df[text_col].fillna("").tolist())
        metrics = compute_category_metrics(y_true=y_true, y_pred=y_pred)

        tfidf = model.pipeline.named_steps["tfidf"]
        n_features = int(len(getattr(tfidf, "vocabulary_", {})))

        trial.set_user_attr("accuracy", float(metrics["accuracy"]))
        trial.set_user_attr("n_features", n_features)
        trial.set_user_attr("fit_time_sec", round(fit_time_sec, 3))
        return float(metrics["macro_f1"])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    typer.echo(f"Starting Optuna tuning: n_trials={n_trials}, timeout_sec={timeout_sec}")
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=False)

    metrics_dir = Path("reports/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    trials_rows: list[dict[str, Any]] = []
    for t in study.trials:
        row = {
            "trial_number": t.number,
            "state": str(t.state),
            "macro_f1": float(t.value) if t.value is not None else None,
            "accuracy": t.user_attrs.get("accuracy"),
            "n_features": t.user_attrs.get("n_features"),
            "fit_time_sec": t.user_attrs.get("fit_time_sec"),
        }
        row.update({f"param_{k}": v for k, v in t.params.items()})
        trials_rows.append(row)

    trials_df = pd.DataFrame(trials_rows).sort_values("macro_f1", ascending=False, na_position="last")
    trials_df.to_csv(metrics_dir / "category_optuna_trials.csv", index=False)

    best = study.best_trial
    best_payload = {
        "trial_number": best.number,
        "macro_f1": float(best.value),
        "accuracy": best.user_attrs.get("accuracy"),
        "n_features": best.user_attrs.get("n_features"),
        "fit_time_sec": best.user_attrs.get("fit_time_sec"),
        "params": best.params,
    }
    with (metrics_dir / "category_optuna_best.json").open("w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2, ensure_ascii=False)

    typer.echo("Optuna tuning completed.")
    typer.echo(f"Best trial: {best.number}")
    typer.echo(f"Best macro_f1: {best.value:.6f}")
    typer.echo(f"Saved trials to: {metrics_dir / 'category_optuna_trials.csv'}")
    typer.echo(f"Saved best to: {metrics_dir / 'category_optuna_best.json'}")


if __name__ == "__main__":
    app()
