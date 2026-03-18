# RiskAware Category-Only Architecture

## 1. Objective

Build a reproducible NLP pipeline for CFPB complaint **category classification**.

- Input: `consumer_complaint_narrative`
- Target: `category = product|issue`
- Output metric: `Macro-F1`

## 2. Current Scope

Implemented and runnable:
- EDA on CFPB complaints (`notebooks/EDA_CFPB.ipynb`)
- Deterministic data preparation and split generation
- Category model training (`TF-IDF + Logistic Regression`)
- Category evaluation on test split
- Artifact and metrics persistence

## 3. Data Flow

1. Raw data: `data/raw/cfpb_complaints.csv`
2. Prepare stage (`src/risk_aware/data/prepare.py`):
   - keep required columns
   - drop invalid text rows
   - build `category`
   - create `complaint_id`
   - split into `train/val/test`
3. Train stage (`scripts/train.py`):
   - train category model on `train.csv`
   - save model artifact
4. Evaluate stage (`scripts/evaluate.py`):
   - evaluate on `test.csv`
   - save category metric report

## 4. Components

- Data preparation:
  - `src/risk_aware/data/prepare.py`
- Model stack:
  - `src/risk_aware/models/stacks.py`
  - `src/risk_aware/models/registry.py`
- Training orchestration:
  - `src/risk_aware/pipelines/training.py`
  - `scripts/train.py`
- Evaluation:
  - `scripts/evaluate.py`

## 5. Artifacts

Processed data:
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/metadata.json`

Model artifact:
- `artifacts/models/category_model.joblib`

Evaluation report:
- `reports/metrics/category_metrics.json`

## 6. Current Baseline

- `category_macro_f1`: `0.301357`

## 7. Next Steps

- Tune TF-IDF baseline (`C`, ngram range, min/max df)
- Implement and benchmark LSTM category stack
- Implement and benchmark Transformer category stack
- Compare all category stacks on the same split protocol

