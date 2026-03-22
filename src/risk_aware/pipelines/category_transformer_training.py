from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


TEXT_COL = "consumer_complaint_narrative"
LABEL_COL = "category"


@dataclass(slots=True)
class TransformerTrainConfig:
    model_name: str
    max_length: int
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    num_train_epochs: int
    weight_decay: float
    warmup_ratio: float
    early_stopping_patience: int


def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path("data/processed")
    train_df = pd.read_csv(base / "train.csv")
    val_df = pd.read_csv(base / "val.csv")
    test_df = pd.read_csv(base / "test.csv")

    required_cols = {TEXT_COL, LABEL_COL}
    cleaned_splits: dict[str, pd.DataFrame] = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} split is missing required columns: {sorted(missing)}")

        local_df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
        local_df[TEXT_COL] = local_df[TEXT_COL].astype(str).str.strip()
        local_df[LABEL_COL] = local_df[LABEL_COL].astype(str).str.strip()
        local_df = local_df[(local_df[TEXT_COL] != "") & (local_df[LABEL_COL] != "")].copy()
        cleaned_splits[name] = local_df.reset_index(drop=True)
    return cleaned_splits["train"], cleaned_splits["val"], cleaned_splits["test"]


def _build_label_mapping(train_df: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted(train_df[LABEL_COL].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def _make_hf_dataset(df: pd.DataFrame, label_to_id: dict[str, int]) -> Dataset:
    work_df = df[[TEXT_COL, LABEL_COL]].copy()
    work_df["labels"] = work_df[LABEL_COL].map(label_to_id)
    if work_df["labels"].isna().any():
        bad_labels = work_df.loc[work_df["labels"].isna(), LABEL_COL].unique().tolist()
        raise ValueError(f"Found labels not present in label_to_id: {bad_labels}")
    work_df["labels"] = work_df["labels"].astype(int)
    work_df = work_df[[TEXT_COL, "labels"]].reset_index(drop=True)
    return Dataset.from_pandas(work_df, preserve_index=False)


def _compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "weighted_f1": float(f1_score(labels, preds, average="weighted")),
    }


def _build_training_args(cfg: TransformerTrainConfig, output_dir: str) -> TrainingArguments:
    common_kwargs = {
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "learning_rate": cfg.learning_rate,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "num_train_epochs": cfg.num_train_epochs,
        "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "fp16": bool(torch.cuda.is_available()),
        "max_grad_norm": 1.0,
        "save_total_limit": 2,
        "report_to": "none",
    }
    try:
        return TrainingArguments(eval_strategy="epoch", **common_kwargs)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **common_kwargs)


def _load_transformer_cfg(config_dir: str) -> TransformerTrainConfig:
    from risk_aware.config import load_project_configs

    cfg = load_project_configs(config_dir=config_dir)
    bert_cfg = dict(cfg["category"].get("stacks", {}).get("bert", {}))
    return TransformerTrainConfig(
        model_name=str(bert_cfg.get("model_name", "distilbert-base-uncased")),
        max_length=int(bert_cfg.get("max_length", 256)),
        learning_rate=float(bert_cfg.get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(bert_cfg.get("batch_size", 16)),
        per_device_eval_batch_size=int(bert_cfg.get("eval_batch_size", 32)),
        num_train_epochs=int(bert_cfg.get("epochs", 3)),
        weight_decay=float(bert_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(bert_cfg.get("warmup_ratio", 0.1)),
        early_stopping_patience=int(bert_cfg.get("early_stopping_patience", 2)),
    )


def run_category_transformer_training(config_dir: str = "configs") -> dict[str, Any]:
    cfg = _load_transformer_cfg(config_dir=config_dir)

    train_df, val_df, test_df = _load_splits()
    label_to_id, id_to_label = _build_label_mapping(train_df)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_ds = _make_hf_dataset(train_df, label_to_id)
    val_ds = _make_hf_dataset(val_df, label_to_id)
    test_ds = _make_hf_dataset(test_df, label_to_id)

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch[TEXT_COL],
            padding="max_length",
            truncation=True,
            max_length=cfg.max_length,
        )

    train_ds = train_ds.map(tokenize_batch, batched=True)
    val_ds = val_ds.map(tokenize_batch, batched=True)
    test_ds = test_ds.map(tokenize_batch, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    output_dir = "artifacts/category_transformer/distilbert_baseline"
    trainer = Trainer(
        model=model,
        args=_build_training_args(cfg, output_dir=output_dir),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    train_output = trainer.train()
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    test_pred = trainer.predict(test_ds)

    test_logits = test_pred.predictions
    test_labels = test_pred.label_ids
    test_preds = np.argmax(test_logits, axis=-1)
    test_metrics = {
        "accuracy": float(accuracy_score(test_labels, test_preds)),
        "macro_f1": float(f1_score(test_labels, test_preds, average="macro")),
        "weighted_f1": float(f1_score(test_labels, test_preds, average="weighted")),
    }

    artifacts_dir = Path(output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "label_to_id.json").write_text(
        json.dumps(label_to_id, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (artifacts_dir / "id_to_label.json").write_text(
        json.dumps({str(k): v for k, v in id_to_label.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics_dir = Path("reports/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "category_transformer_metrics_val.json").write_text(
        json.dumps(val_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (metrics_dir / "category_transformer_metrics_test.json").write_text(
        json.dumps(test_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {
        "model_name": cfg.model_name,
        "max_length": cfg.max_length,
        "best_checkpoint_metric": "macro_f1",
        "val": val_metrics,
        "test": test_metrics,
        "train_runtime_sec": float(train_output.metrics.get("train_runtime", 0.0)),
    }
    (metrics_dir / "category_transformer_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary

