from __future__ import annotations

import torch
import torch.nn as nn


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
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )
        if self.architecture == "lstm_bilstm":
            self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=num_layers_lstm,
                batch_first=True,
                bidirectional=False,
                dropout=dropout if num_layers_lstm > 1 else 0.0,
            )
            bilstm_input_dim = lstm_hidden_dim
        else:
            self.lstm = None
            bilstm_input_dim = embedding_dim

        self.bilstm = nn.LSTM(
            input_size=bilstm_input_dim,
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
            embedded,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        if self.architecture == "lstm_bilstm":
            packed_lstm_out, _ = self.lstm(packed_input)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True)
            packed_input = nn.utils.rnn.pack_padded_sequence(
                lstm_out,
                lengths=lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

        _, (h_n, _) = self.bilstm(packed_input)
        # BiLSTM: take forward/backward final states and concatenate.
        features = torch.cat((h_n[-2], h_n[-1]), dim=1)
        features = self.dropout(features)
        return self.classifier(features)

