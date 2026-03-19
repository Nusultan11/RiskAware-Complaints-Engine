from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import typer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset

app = typer.Typer(add_completion=False, help="Diagnostic analysis for LSTM category model.")

TRAIN_CSV = Path("data/processed/train.csv")
TEST_CSV = Path("data/processed/test.csv")
TEST_NPZ = Path("artifacts/lstm_preprocessing/test.npz")
MODEL_PT = Path("artifacts/category_lstm/model.pt")
TRAIN_META = Path("artifacts/category_lstm/training_metadata.json")
OUT_DIR = Path("reports/analysis/lstm")
TARGET_COL = "category"


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        architecture: str,
        vocab_size: int,
        embedding_dim: int,
        lstm_hidden_dim: int,
        bilstm_hidden_dim: int,
        num_layers_lstm: int,
        num_layers_bilstm: int,
        n_labels: int,
        pad_idx: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if architecture not in {"bilstm_only", "lstm_bilstm"}:
            raise ValueError(f"Unsupported architecture: {architecture}")
        self.architecture = architecture

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if architecture == "lstm_bilstm":
            self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=num_layers_lstm,
                batch_first=True,
                bidirectional=False,
                dropout=dropout if num_layers_lstm > 1 else 0.0,
            )
            bilstm_input = lstm_hidden_dim
        else:
            self.lstm = None
            bilstm_input = embedding_dim

        self.bilstm = nn.LSTM(
            input_size=bilstm_input,
            hidden_size=bilstm_hidden_dim,
            num_layers=num_layers_bilstm,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers_bilstm > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bilstm_hidden_dim * 2, n_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).clamp(min=1).to(torch.int64)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        if self.architecture == "lstm_bilstm":
            packed_lstm_out, _ = self.lstm(packed_input)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True)
            packed_input = nn.utils.rnn.pack_padded_sequence(
                lstm_out, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        _, (h_n, _) = self.bilstm(packed_input)
        features = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.classifier(self.dropout(features))


def _load_checkpoint() -> tuple[nn.Module, list[str], torch.device]:
    if not MODEL_PT.exists():
        raise typer.BadParameter(f"Checkpoint not found: {MODEL_PT}")
    ckpt = torch.load(MODEL_PT, map_location="cpu")
    labels = [str(x) for x in ckpt["labels"]]
    cfg = ckpt["config"]

    model = BiLSTMClassifier(
        architecture=str(cfg.get("architecture", "bilstm_only")),
        vocab_size=int(cfg["vocab_size"]),
        embedding_dim=int(cfg["embedding_dim"]),
        lstm_hidden_dim=int(cfg.get("lstm_hidden_dim", cfg.get("hidden_dim", 128))),
        bilstm_hidden_dim=int(cfg.get("bilstm_hidden_dim", cfg.get("hidden_dim", 128))),
        num_layers_lstm=int(cfg.get("num_layers_lstm", 1)),
        num_layers_bilstm=int(cfg.get("num_layers_bilstm", 1)),
        n_labels=len(labels),
        pad_idx=int(ckpt.get("pad_idx", 0)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, labels, device


def _load_test_loader(batch_size: int = 256) -> tuple[DataLoader[tuple[torch.Tensor, ...]], np.ndarray]:
    if not TEST_NPZ.exists():
        raise typer.BadParameter(f"Test NPZ not found: {TEST_NPZ}")
    payload = np.load(TEST_NPZ)
    input_ids = torch.from_numpy(payload["input_ids"]).to(torch.int64)
    attention_mask = torch.from_numpy(payload["attention_mask"]).to(torch.int64)
    y_true_idx = payload["labels"].astype(np.int64)
    labels_t = torch.from_numpy(y_true_idx).to(torch.int64)
    dataset = TensorDataset(input_ids, attention_mask, labels_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, y_true_idx


def _predict_logits(model: nn.Module, loader: DataLoader[tuple[torch.Tensor, ...]], device: torch.device) -> np.ndarray:
    logits_all: list[np.ndarray] = []
    with torch.no_grad():
        for input_ids, attention_mask, _ in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids, attention_mask)
            logits_all.append(logits.cpu().numpy())
    return np.vstack(logits_all)


def _topk_accuracy(y_true_idx: np.ndarray, logits: np.ndarray, k: int) -> float:
    topk_idx = np.argpartition(logits, kth=-k, axis=1)[:, -k:]
    hits = np.any(topk_idx == y_true_idx[:, None], axis=1)
    return float(np.mean(hits))


@app.command()
def run(
    head_min: int = 1000,
    mid_min: int = 100,
    tail_max: int = 99,
    top_head_confusion: int = 20,
) -> None:
    for path in [TRAIN_CSV, TEST_CSV, MODEL_PT, TEST_NPZ]:
        if not path.exists():
            raise typer.BadParameter(f"Missing required file: {path}")

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    model, labels, device = _load_checkpoint()
    loader, y_true_idx = _load_test_loader(batch_size=256)
    logits = _predict_logits(model, loader, device=device)
    y_pred_idx = np.argmax(logits, axis=1)

    y_true = np.array([labels[i] for i in y_true_idx], dtype=object)
    y_pred = np.array([labels[i] for i in y_pred_idx], dtype=object)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    overall = {
        "macro_f1": float(f1_score(y_true=y_true, y_pred=y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true=y_true, y_pred=y_pred)),
        "top3_accuracy": _topk_accuracy(y_true_idx=y_true_idx, logits=logits, k=3),
        "top5_accuracy": _topk_accuracy(y_true_idx=y_true_idx, logits=logits, k=5),
        "n_test_samples": int(len(y_true)),
        "n_labels": int(len(labels)),
        "device": device.type,
    }

    train_counts = train_df[TARGET_COL].astype(str).value_counts()
    group_map: dict[str, str] = {}
    for cls, c in train_counts.items():
        if c >= head_min:
            group_map[str(cls)] = "head"
        elif c >= mid_min:
            group_map[str(cls)] = "mid"
        elif c <= tail_max:
            group_map[str(cls)] = "tail"
        else:
            group_map[str(cls)] = "mid"

    group_rows: list[dict[str, Any]] = []
    for group in ["head", "mid", "tail"]:
        group_classes = [cls for cls, g in group_map.items() if g == group]
        mask = np.isin(y_true, group_classes)
        if not np.any(mask):
            continue
        group_rows.append(
            {
                "group": group,
                "n_classes": int(len(group_classes)),
                "n_samples_test": int(mask.sum()),
                "macro_f1": float(f1_score(y_true=y_true[mask], y_pred=y_pred[mask], average="macro")),
                "accuracy": float(accuracy_score(y_true=y_true[mask], y_pred=y_pred[mask])),
            }
        )
    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(OUT_DIR / "head_mid_tail_metrics.csv", index=False)

    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    with (OUT_DIR / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(class_report, f, indent=2, ensure_ascii=False)

    p, r, f1_arr, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame(
        {
            "category": labels,
            "support_test": support.astype(int),
            "precision": p,
            "recall": r,
            "f1": f1_arr,
            "group": [group_map.get(cls, "unknown") for cls in labels],
        }
    ).sort_values("f1", ascending=True)
    per_class.to_csv(OUT_DIR / "per_class_metrics.csv", index=False)

    pred_dist = (
        pd.Series(y_pred, name="category")
        .value_counts()
        .rename_axis("category")
        .reset_index(name="pred_count")
    )
    pred_dist["pred_share"] = pred_dist["pred_count"] / len(y_pred)
    pred_dist.to_csv(OUT_DIR / "predicted_class_distribution.csv", index=False)

    head_classes = train_counts[train_counts >= head_min].index.astype(str).tolist()[:top_head_confusion]
    head_mask = np.isin(y_true, head_classes)
    cm = confusion_matrix(y_true[head_mask], y_pred[head_mask], labels=head_classes)
    cm_df = pd.DataFrame(cm, index=head_classes, columns=head_classes)
    cm_df.to_csv(OUT_DIR / "confusion_head_top20.csv", index=True)

    if TRAIN_META.exists():
        train_meta = json.loads(TRAIN_META.read_text(encoding="utf-8"))
        history = train_meta.get("history", [])
        if history:
            pd.DataFrame(history).to_csv(OUT_DIR / "training_curves.csv", index=False)

    summary = {
        "overall": overall,
        "group_thresholds": {"head_min": head_min, "mid_min": mid_min, "tail_max": tail_max},
        "outputs": [
            "head_mid_tail_metrics.csv",
            "per_class_metrics.csv",
            "predicted_class_distribution.csv",
            "confusion_head_top20.csv",
            "classification_report.json",
            "training_curves.csv (if available)",
        ],
    }
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    typer.echo("LSTM diagnostic analysis completed.")
    typer.echo(f"Overall macro_f1={overall['macro_f1']:.4f} top3={overall['top3_accuracy']:.4f} top5={overall['top5_accuracy']:.4f}")
    typer.echo(f"Saved report dir: {OUT_DIR}")


if __name__ == "__main__":
    app()
