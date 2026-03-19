from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import typer
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from risk_aware.config import load_project_configs

app = typer.Typer(add_completion=False, help="Train BiLSTM category model on preprocessed NPZ tensors.")


@dataclass(slots=True)
class LSTMTrainConfig:
    max_length: int
    vocab_size: int
    n_labels: int
    embedding_dim: int
    hidden_dim: int
    dropout: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    early_stopping_patience: int
    seed: int
    device: str


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        n_labels: int,
        pad_idx: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).clamp(min=1).to(torch.int64)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.encoder(packed)
        # BiLSTM: take forward/backward final states and concatenate.
        features = torch.cat((h_n[-2], h_n[-1]), dim=1)
        features = self.dropout(features)
        return self.classifier(features)


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_npz(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not path.exists():
        raise typer.BadParameter(f"Missing NPZ file: {path}")
    payload = np.load(path)
    input_ids = torch.from_numpy(payload["input_ids"]).to(torch.int64)
    attention_mask = torch.from_numpy(payload["attention_mask"]).to(torch.int64)
    labels = torch.from_numpy(payload["labels"]).to(torch.int64)
    return input_ids, attention_mask, labels


def _build_loader(path: Path, batch_size: int, shuffle: bool) -> DataLoader[tuple[torch.Tensor, ...]]:
    input_ids, attention_mask, labels = _load_npz(path)
    dataset = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _predict(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def _compute_metrics(y_true_idx: np.ndarray, y_pred_idx: np.ndarray, labels: list[str]) -> dict[str, Any]:
    y_true = np.array([labels[int(i)] for i in y_true_idx], dtype=object)
    y_pred = np.array([labels[int(i)] for i in y_pred_idx], dtype=object)
    return {
        "macro_f1": float(f1_score(y_true=y_true, y_pred=y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true=y_true, y_pred=y_pred)),
    }


def _load_train_cfg(config_dir: str) -> tuple[LSTMTrainConfig, list[str], int]:
    cfg = load_project_configs(config_dir=config_dir)
    base_cfg = cfg["base"]
    category_cfg = cfg["category"]
    bilstm_cfg = dict(category_cfg.get("stacks", {}).get("bilstm", {}))

    preprocessing_meta_path = Path("artifacts/lstm_preprocessing/metadata.json")
    vocab_path = Path("artifacts/lstm_preprocessing/vocab.json")
    if not preprocessing_meta_path.exists():
        raise typer.BadParameter("Run scripts/prepare_lstm_data.py first (metadata missing).")
    if not vocab_path.exists():
        raise typer.BadParameter("Run scripts/prepare_lstm_data.py first (vocab missing).")

    meta = json.loads(preprocessing_meta_path.read_text(encoding="utf-8"))
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    labels = [str(label) for label in meta["labels"]]
    pad_idx = int(vocab.get(meta.get("pad_token", "<pad>"), 0))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cfg = LSTMTrainConfig(
        max_length=int(meta["max_length"]),
        vocab_size=int(meta["actual_vocab_size"]),
        n_labels=int(meta["n_labels"]),
        embedding_dim=int(bilstm_cfg.get("embedding_dim", 300)),
        hidden_dim=int(bilstm_cfg.get("hidden_dim", 256)),
        dropout=float(bilstm_cfg.get("dropout", 0.2)),
        epochs=int(bilstm_cfg.get("epochs", 6)),
        batch_size=int(bilstm_cfg.get("batch_size", 64)),
        learning_rate=float(bilstm_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(bilstm_cfg.get("weight_decay", 1e-5)),
        grad_clip_norm=float(bilstm_cfg.get("grad_clip_norm", 1.0)),
        early_stopping_patience=int(bilstm_cfg.get("early_stopping_patience", 2)),
        seed=int(base_cfg.get("project", {}).get("seed", 42)),
        device=device,
    )
    return train_cfg, labels, pad_idx


@app.command()
def run(config_dir: str = "configs", epochs: int | None = None) -> None:
    train_cfg, labels, pad_idx = _load_train_cfg(config_dir=config_dir)
    if epochs is not None:
        if epochs < 1:
            raise typer.BadParameter("--epochs must be >= 1")
        train_cfg.epochs = epochs
    _set_torch_seed(train_cfg.seed)

    device = torch.device(train_cfg.device)
    train_loader = _build_loader(Path("artifacts/lstm_preprocessing/train.npz"), train_cfg.batch_size, True)
    val_loader = _build_loader(Path("artifacts/lstm_preprocessing/val.npz"), train_cfg.batch_size, False)
    test_loader = _build_loader(Path("artifacts/lstm_preprocessing/test.npz"), train_cfg.batch_size, False)

    model = BiLSTMClassifier(
        vocab_size=train_cfg.vocab_size,
        embedding_dim=train_cfg.embedding_dim,
        hidden_dim=train_cfg.hidden_dim,
        n_labels=train_cfg.n_labels,
        pad_idx=pad_idx,
        dropout=train_cfg.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    best_val_f1 = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for input_ids, attention_mask, target in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.grad_clip_norm)
            optimizer.step()

            batch_size = target.size(0)
            running_loss += float(loss.item()) * batch_size
            total_samples += batch_size

        avg_train_loss = running_loss / max(total_samples, 1)
        val_true, val_pred = _predict(model, val_loader, device)
        val_metrics = _compute_metrics(val_true, val_pred, labels=labels)
        val_f1 = float(val_metrics["macro_f1"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_macro_f1": val_f1,
                "val_accuracy": float(val_metrics["accuracy"]),
            }
        )
        typer.echo(
            f"epoch={epoch} train_loss={avg_train_loss:.4f} "
            f"val_macro_f1={val_f1:.4f} val_accuracy={val_metrics['accuracy']:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= train_cfg.early_stopping_patience:
                typer.echo("Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint.")

    model.load_state_dict(best_state)
    model = model.to(device)

    val_true, val_pred = _predict(model, val_loader, device)
    test_true, test_pred = _predict(model, test_loader, device)
    val_metrics = _compute_metrics(val_true, val_pred, labels=labels)
    test_metrics = _compute_metrics(test_true, test_pred, labels=labels)

    model_dir = Path("artifacts/category_lstm")
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": labels,
            "pad_idx": pad_idx,
            "config": train_cfg.__dict__,
        },
        model_dir / "model.pt",
    )

    with (model_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_f1,
                "history": history,
                "config": train_cfg.__dict__,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    metrics_dir = Path("reports/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "category_lstm_metrics_val.json").open("w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2, ensure_ascii=False)
    with (metrics_dir / "category_lstm_metrics_test.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    typer.echo("LSTM category training completed.")
    typer.echo(f"Best epoch: {best_epoch}")
    typer.echo(f"Val Macro-F1: {val_metrics['macro_f1']:.4f}")
    typer.echo(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    typer.echo(f"Artifacts saved to: {model_dir}")
    typer.echo(f"Metrics saved to: {metrics_dir}")


if __name__ == "__main__":
    app()
