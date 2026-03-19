# RiskAware Architecture (Category-only)

## Goal

Predict complaint `category` from `consumer_complaint_narrative` with a reproducible ML pipeline.

## Pipeline

1. `scripts/prepare_data.py`
2. `scripts/prepare_lstm_data.py` (sequence preprocessing for LSTM)
3. `scripts/train_category.py`
4. `scripts/train_lstm_category.py`
5. `scripts/evaluate_category.py`

## Data flow

- Raw source: `data/raw/cfpb_complaints.csv`
- Processed splits: `data/processed/train.csv`, `val.csv`, `test.csv`
- TF-IDF artifacts: `artifacts/category/`
- LSTM preprocessing artifacts: `artifacts/lstm_preprocessing/`
  - `train.npz`, `val.npz`, `test.npz`
  - `vocab.json`
  - `metadata.json`
- LSTM model artifacts: `artifacts/category_lstm/`
  - `model.pt`
  - `training_metadata.json`

## EDA-aligned preprocessing decisions

- Minimum narrative length filter is applied in dataset preparation.
- Text split leakage is explicitly checked (`train/val/test` overlap guard).
- For TF-IDF: anonymization token (`xxxx`) is treated as noise.
- For LSTM: anonymization token is preserved as `<anon>` and numbers as `<num>`.
- Sequence length is capped by `bilstm.max_length` from `configs/category.yaml`.

## Metrics

- Primary metric: Macro-F1.
- Validation and test metrics are saved separately to avoid accidental overwrite.
- LSTM validation/test metrics are saved to
  - `reports/metrics/category_lstm_metrics_val.json`
  - `reports/metrics/category_lstm_metrics_test.json`
