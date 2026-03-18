# RiskAware Complaints Engine

Production-style NLP system for CFPB complaint category classification.

## Overview

The project predicts:
- `category` from complaint text (`consumer_complaint_narrative`)

Current MVP uses:
- `TF-IDF + Logistic Regression` category baseline

## Repository layout

```text
.
|-- configs/
|-- data/
|   |-- raw/
|   `-- processed/
|-- docs/
|-- notebooks/
|   `-- EDA_CFPB.ipynb
|-- reports/
|   `-- metrics/
|-- scripts/
|   |-- train.py
|   |-- evaluate.py
|   `-- serve.py
|-- src/risk_aware/
`-- tests/
```

## End-to-end workflow

1. EDA  
`notebooks/EDA_CFPB.ipynb`

2. Data preparation  
`src/risk_aware/data/prepare.py`  
Output:
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/metadata.json`

3. Training  
`scripts/train.py` (train category model on `train.csv`)  
Output:
- `artifacts/models/category_model.joblib`

4. Evaluation  
`scripts/evaluate.py` (category evaluation on `test.csv`)  
Output:
- `reports/metrics/category_metrics.json`

5. Gold-set candidate generation (human labeling)  
`scripts/create_gold_set.py`  
Output:
- `reports/labeling/gold_candidates.csv`
- `reports/labeling/gold_manifest.json`

## Setup and run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

If running scripts directly, ensure package import path:

```bash
$env:PYTHONPATH = "src"
```

Run sequence:

```bash
python src/risk_aware/data/prepare.py
python scripts/train.py
python scripts/evaluate.py
```

## Current baseline metric

From `reports/metrics/category_metrics.json`:
- `category_macro_f1`: `0.301357`

## Known limitations

- Current active scope is category-only.
- Deep text stacks (`bilstm`, `bert`) are scaffolded but not yet enabled in the training flow.

## Next iteration

- Category model tuning (`TF-IDF` hyperparameters)
- LSTM category model implementation
- Transformer category model fine-tuning and comparison
