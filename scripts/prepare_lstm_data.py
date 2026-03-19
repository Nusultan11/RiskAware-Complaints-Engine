from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from risk_aware.config import load_project_configs
from risk_aware.preprocessing.neural import NeuralTextPreprocessor

app = typer.Typer(add_completion=False, help="Prepare LSTM-ready tensors from processed splits.")


def _load_split(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Split not found: {path}")
    frame = pd.read_csv(path)
    missing = [c for c in [text_col, label_col] if c not in frame.columns]
    if missing:
        raise typer.BadParameter(f"Missing columns in {path.name}: {missing}")
    return frame.dropna(subset=[text_col, label_col]).copy()


def _save_npz(
    output_path: Path,
    token_ids: np.ndarray,
    attention_mask: np.ndarray,
    labels: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        input_ids=token_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


@app.command()
def run(config_dir: str = "configs") -> None:
    cfg = load_project_configs(config_dir=config_dir)
    base_cfg: dict[str, Any] = cfg["base"]
    category_cfg: dict[str, Any] = cfg["category"]

    text_col = str(base_cfg["data"]["text_column"])
    label_col = str(base_cfg["data"]["category_column"])
    bilstm_cfg = dict(category_cfg.get("stacks", {}).get("bilstm", {}))

    max_vocab_size = int(bilstm_cfg.get("max_vocab_size", 50_000))
    min_token_freq = int(bilstm_cfg.get("min_token_freq", 2))
    max_length = int(bilstm_cfg.get("max_length", 256))

    processed_dir = Path("data/processed")
    train_df = _load_split(processed_dir / "train.csv", text_col, label_col)
    val_df = _load_split(processed_dir / "val.csv", text_col, label_col)
    test_df = _load_split(processed_dir / "test.csv", text_col, label_col)

    label_values = sorted(train_df[label_col].astype(str).unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(label_values)}

    preprocessor = NeuralTextPreprocessor(
        max_vocab_size=max_vocab_size,
        min_token_freq=min_token_freq,
        max_length=max_length,
    )
    preprocessor.fit(train_df[text_col].astype(str).tolist())

    output_dir = Path("artifacts/lstm_preprocessing")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        texts = frame[text_col].astype(str).tolist()
        token_ids = preprocessor.transform(texts)
        attention_mask = preprocessor.build_attention_mask(token_ids)
        labels = frame[label_col].astype(str).map(label_to_id).to_numpy(dtype=np.int32)
        _save_npz(output_dir / f"{split_name}.npz", token_ids, attention_mask, labels)

    vocab = preprocessor.vocab
    if vocab is None:
        raise RuntimeError("Vocabulary was not fitted.")

    metadata = {
        "text_column": text_col,
        "label_column": label_col,
        "max_length": max_length,
        "max_vocab_size": max_vocab_size,
        "min_token_freq": min_token_freq,
        "actual_vocab_size": len(vocab.id_to_token),
        "pad_token": vocab.pad_token,
        "unk_token": vocab.unk_token,
        "n_labels": len(label_values),
        "labels": label_values,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with (output_dir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab.token_to_id, f, indent=2, ensure_ascii=False)

    typer.echo("LSTM preprocessing completed.")
    typer.echo(f"Saved tensors to: {output_dir}")


if __name__ == "__main__":
    app()
