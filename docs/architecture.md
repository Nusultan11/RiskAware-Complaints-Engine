# RiskAware Architecture (Category-only)

## Goal

Predict complaint `category` from `consumer_complaint_narrative` with a reproducible and leakage-safe ML pipeline.

## Layered Architecture

### Data layer

- Raw source: `data/raw/cfpb_complaints.csv`
- Canonical preparation: [`src/risk_aware/data/prepare.py`](../src/risk_aware/data/prepare.py)
- Rules:
  - `MIN_CLASS_COUNT = 5`
  - remove conflicting `text_key`
  - group-aware split by `text_key`
  - zero overlap between `train/val/test` by `text_key`

### Preprocessing layer

- TF-IDF preprocessing: [`src/risk_aware/preprocessing/tfidf.py`](../src/risk_aware/preprocessing/tfidf.py)
- Neural preprocessing: [`src/risk_aware/preprocessing/neural.py`](../src/risk_aware/preprocessing/neural.py)

### Model layer

- Sparse baseline: TF-IDF + Logistic Regression
- DL baseline: BiLSTM ([`src/risk_aware/models/category/bilstm.py`](../src/risk_aware/models/category/bilstm.py))
- Transformer comparison: DistilBERT

### Pipeline layer

- TF-IDF training: [`src/risk_aware/pipelines/category_training.py`](../src/risk_aware/pipelines/category_training.py)
- LSTM training: [`src/risk_aware/pipelines/category_lstm_training.py`](../src/risk_aware/pipelines/category_lstm_training.py)
- Transformer training: [`src/risk_aware/pipelines/category_transformer_training.py`](../src/risk_aware/pipelines/category_transformer_training.py)

### Inference layer

- Unified interface: [`src/risk_aware/inference/category_predictor.py`](../src/risk_aware/inference/category_predictor.py)
- API:
  - `predict(texts, model_type="tfidf_lr")`
  - `predict(texts, model_type="bilstm")`
  - `predict(texts, model_type="distilbert")` (reserved)

### Scripts layer (entrypoints)

Scripts are launchers only:

- `scripts/prepare_data.py`
- `scripts/prepare_lstm_data.py`
- `scripts/train_category.py`
- `scripts/train_lstm_category.py`
- `scripts/train_transformer_category.py`
- `scripts/evaluate_category.py`
- `scripts/generate_final_comparison.py`

## Artifacts

- TF-IDF: `artifacts/category/`
- LSTM preprocessing: `artifacts/lstm_preprocessing/`
- LSTM model: `artifacts/category_lstm/`
- Transformer: `artifacts/category_transformer/`
- Final comparison: `reports/final/model_comparison.csv`, `reports/final/model_comparison.json`

## Metrics Policy

- Primary metric: **Macro-F1**
- Secondary: Accuracy, Weighted-F1
- Metrics are written into `reports/metrics/*` and aggregated into `reports/final/*`
