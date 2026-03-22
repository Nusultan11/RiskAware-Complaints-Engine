from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
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


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path("data/processed")
    train_path = base / "train.csv"
    val_path = base / "val.csv"
    test_path = base / "test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    required_cols = {TEXT_COL, LABEL_COL}
    cleaned_splits: dict[str, pd.DataFrame] = {}

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} split is missing required columns: {sorted(missing)}")

        local_df = df.copy()
        local_df = local_df.dropna(subset=[TEXT_COL, LABEL_COL])
        local_df[TEXT_COL] = local_df[TEXT_COL].astype(str).str.strip()
        local_df[LABEL_COL] = local_df[LABEL_COL].astype(str).str.strip()
        local_df = local_df[(local_df[TEXT_COL] != "") & (local_df[LABEL_COL] != "")].copy()
        cleaned_splits[name] = local_df.reset_index(drop=True)

    return cleaned_splits["train"], cleaned_splits["val"], cleaned_splits["test"]


def build_label_mapping(train_df: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted(train_df[LABEL_COL].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def tokenize_batch(batch: dict, tokenizer: AutoTokenizer, max_length: int) -> dict:
    return tokenizer(
        batch[TEXT_COL],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def make_hf_dataset(df: pd.DataFrame, label_to_id: dict[str, int]) -> Dataset:
    work_df = df[[TEXT_COL, LABEL_COL]].copy()
    work_df["labels"] = work_df[LABEL_COL].map(label_to_id)

    if work_df["labels"].isna().any():
        bad_labels = work_df.loc[work_df["labels"].isna(), LABEL_COL].unique().tolist()
        raise ValueError(f"Found labels not present in label_to_id: {bad_labels}")

    work_df["labels"] = work_df["labels"].astype(int)
    work_df = work_df[[TEXT_COL, "labels"]].reset_index(drop=True)
    return Dataset.from_pandas(work_df, preserve_index=False)


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "weighted_f1": float(f1_score(labels, preds, average="weighted")),
    }


def build_training_args() -> TrainingArguments:
    common_kwargs = {
        "output_dir": "artifacts/category_transformer/distilbert_baseline",
        "overwrite_output_dir": True,
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": 3,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "fp16": True,
        "max_grad_norm": 1.0,
        "save_total_limit": 2,
        "report_to": "none",
    }

    # transformers argument naming differs by version: eval_strategy vs evaluation_strategy
    try:
        return TrainingArguments(eval_strategy="epoch", **common_kwargs)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **common_kwargs)


def main() -> None:
    model_name = "distilbert-base-uncased"
    max_length = 256

    train_df, val_df, test_df = load_splits()
    label_to_id, id_to_label = build_label_mapping(train_df)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = make_hf_dataset(train_df, label_to_id)
    val_ds = make_hf_dataset(val_df, label_to_id)
    test_ds = make_hf_dataset(test_df, label_to_id)

    train_ds = train_ds.map(lambda batch: tokenize_batch(batch, tokenizer, max_length), batched=True)
    val_ds = val_ds.map(lambda batch: tokenize_batch(batch, tokenizer, max_length), batched=True)
    test_ds = test_ds.map(lambda batch: tokenize_batch(batch, tokenizer, max_length), batched=True)

    num_labels = len(label_to_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = build_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Temporary sanity checks before full training stage.
    print("num_labels =", num_labels)
    print("model ready")
    print("trainer ready")

    _ = (
        id_to_label,
        np.array([], dtype=np.int64),
        json.dumps({"status": "trainer_ready"}),
        Path("reports/metrics"),
        test_ds,
    )


if __name__ == "__main__":
    main()
