from __future__ import annotations

import json
import joblib
import numpy as np
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from risk_aware.models.category.bilstm import BiLSTMClassifier
from risk_aware.preprocessing.neural import simple_tokenize


class CategoryPredictor:
    """
    Unified category inference interface for multiple model types.

    Supported:
    - tfidf_lr
    - bilstm
    """

    def __init__(
        self,
        artifacts_dir: str | Path = "artifacts",
        model_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        # Backward compatibility: explicit model_path means legacy TF-IDF loading.
        self.legacy_model = joblib.load(Path(model_path)) if model_path is not None else None

        self.artifacts_dir = Path(artifacts_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self._tfidf_model: Any | None = None
        self._bilstm_model: BiLSTMClassifier | None = None
        self._bilstm_labels: list[str] | None = None
        self._bilstm_vocab: dict[str, int] | None = None
        self._bilstm_max_length: int | None = None
        self._bilstm_pad_idx: int | None = None
        self._distilbert_model: AutoModelForSequenceClassification | None = None
        self._distilbert_tokenizer: AutoTokenizer | None = None
        self._distilbert_labels: list[str] | None = None

    def _load_tfidf(self) -> Any:
        if self._tfidf_model is not None:
            return self._tfidf_model

        model_path = self.artifacts_dir / "category" / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"TF-IDF model not found: {model_path}")
        self._tfidf_model = joblib.load(model_path)
        return self._tfidf_model

    def _load_bilstm(self) -> tuple[BiLSTMClassifier, list[str], dict[str, int], int, int]:
        if (
            self._bilstm_model is not None
            and self._bilstm_labels is not None
            and self._bilstm_vocab is not None
            and self._bilstm_max_length is not None
            and self._bilstm_pad_idx is not None
        ):
            return (
                self._bilstm_model,
                self._bilstm_labels,
                self._bilstm_vocab,
                self._bilstm_max_length,
                self._bilstm_pad_idx,
            )

        model_path = self.artifacts_dir / "category_lstm" / "model.pt"
        vocab_path = self.artifacts_dir / "lstm_preprocessing" / "vocab.json"
        if not model_path.exists():
            raise FileNotFoundError(f"BiLSTM model not found: {model_path}")
        if not vocab_path.exists():
            raise FileNotFoundError(f"BiLSTM vocab not found: {vocab_path}")

        ckpt = torch.load(model_path, map_location="cpu")
        cfg = dict(ckpt["config"])
        labels = [str(x) for x in ckpt["labels"]]
        pad_idx = int(ckpt["pad_idx"])
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

        model = BiLSTMClassifier(
            architecture=str(cfg["architecture"]),
            vocab_size=int(cfg["vocab_size"]),
            embedding_dim=int(cfg["embedding_dim"]),
            lstm_hidden_dim=int(cfg["lstm_hidden_dim"]),
            bilstm_hidden_dim=int(cfg["bilstm_hidden_dim"]),
            num_layers_lstm=int(cfg["num_layers_lstm"]),
            num_layers_bilstm=int(cfg["num_layers_bilstm"]),
            n_labels=int(cfg["n_labels"]),
            pad_idx=pad_idx,
            dropout=float(cfg["dropout"]),
        )
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(self.device)
        model.eval()

        self._bilstm_model = model
        self._bilstm_labels = labels
        self._bilstm_vocab = {str(k): int(v) for k, v in vocab.items()}
        self._bilstm_max_length = int(cfg["max_length"])
        self._bilstm_pad_idx = pad_idx

        return model, labels, self._bilstm_vocab, self._bilstm_max_length, pad_idx

    def _load_distilbert(self) -> tuple[AutoModelForSequenceClassification, AutoTokenizer, list[str]]:
        if (
            self._distilbert_model is not None
            and self._distilbert_tokenizer is not None
            and self._distilbert_labels is not None
        ):
            return self._distilbert_model, self._distilbert_tokenizer, self._distilbert_labels

        model_dir = self.artifacts_dir / "category_transformer" / "distilbert_baseline"
        id_to_label_path = model_dir / "id_to_label.json"
        if not model_dir.exists():
            raise FileNotFoundError(f"DistilBERT model dir not found: {model_dir}")
        if not id_to_label_path.exists():
            raise FileNotFoundError(f"DistilBERT labels mapping not found: {id_to_label_path}")

        id_to_label = json.loads(id_to_label_path.read_text(encoding="utf-8"))
        labels = [value for _, value in sorted(((int(k), str(v)) for k, v in id_to_label.items()), key=lambda x: x[0])]

        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = model.to(self.device)
        model.eval()

        self._distilbert_model = model
        self._distilbert_tokenizer = tokenizer
        self._distilbert_labels = labels
        return model, tokenizer, labels

    def _encode_for_bilstm(
        self,
        texts: list[str],
        vocab: dict[str, int],
        max_length: int,
        pad_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unk_idx = int(vocab.get("<unk>", 1))
        token_ids = np.full((len(texts), max_length), pad_idx, dtype=np.int64)

        for i, text in enumerate(texts):
            tokens = simple_tokenize(str(text))[:max_length]
            ids = [int(vocab.get(tok, unk_idx)) for tok in tokens]
            if ids:
                token_ids[i, : len(ids)] = np.asarray(ids, dtype=np.int64)

        attention_mask = (token_ids != pad_idx).astype(np.int64)
        input_ids_t = torch.from_numpy(token_ids).to(self.device)
        attention_mask_t = torch.from_numpy(attention_mask).to(self.device)
        return input_ids_t, attention_mask_t

    def predict(self, texts: list[str], model_type: str = "tfidf_lr") -> np.ndarray:
        if self.legacy_model is not None and model_type == "tfidf_lr":
            proba = self.legacy_model.predict_proba(texts)
            pred_idx = np.argmax(proba, axis=1)
            return np.array([self.legacy_model.labels[i] for i in pred_idx], dtype=object)

        if model_type == "tfidf_lr":
            model = self._load_tfidf()
            proba = model.predict_proba(texts)
            pred_idx = np.argmax(proba, axis=1)
            return np.array([model.labels[i] for i in pred_idx], dtype=object)

        if model_type == "bilstm":
            model, labels, vocab, max_length, pad_idx = self._load_bilstm()
            input_ids, attention_mask = self._encode_for_bilstm(texts, vocab, max_length, pad_idx)
            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
            return np.array([labels[int(i)] for i in pred_idx], dtype=object)

        if model_type == "distilbert":
            model, tokenizer, labels = self._load_distilbert()
            batch = tokenizer(
                [str(t) for t in texts],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**batch).logits
                pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
            return np.array([labels[int(i)] for i in pred_idx], dtype=object)

        raise ValueError("model_type must be one of: {'tfidf_lr', 'bilstm', 'distilbert'}")

    def predict_proba(self, texts: list[str], model_type: str = "tfidf_lr") -> np.ndarray:
        if self.legacy_model is not None and model_type == "tfidf_lr":
            return self.legacy_model.predict_proba(texts)

        if model_type == "tfidf_lr":
            model = self._load_tfidf()
            return model.predict_proba(texts)

        if model_type == "bilstm":
            model, _, vocab, max_length, pad_idx = self._load_bilstm()
            input_ids, attention_mask = self._encode_for_bilstm(texts, vocab, max_length, pad_idx)
            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                proba = torch.softmax(logits, dim=1).cpu().numpy()
            return proba

        if model_type == "distilbert":
            model, tokenizer, _ = self._load_distilbert()
            batch = tokenizer(
                [str(t) for t in texts],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**batch).logits
                proba = torch.softmax(logits, dim=1).cpu().numpy()
            return proba

        raise ValueError("model_type must be one of: {'tfidf_lr', 'bilstm', 'distilbert'}")
