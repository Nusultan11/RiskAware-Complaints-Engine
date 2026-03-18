# RiskAware Complaints Engine

Category-only NLP pipeline for CFPB complaint classification.

## Project structure

```text
RiskAware-Complaints-Engine/
├── .env.example
├── .gitignore
├── Makefile
├── pyproject.toml
├── README.md
├── AGENT.md (local, gitignored)
├── configs/
│   ├── base.yaml
│   └── category.yaml
├── data/
│   ├── raw/
│   │   └── cfpb_complaints.csv
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── metadata.json
├── artifacts/
│   └── category/
│       ├── model.joblib
│       ├── vectorizer.joblib
│       ├── label_encoder.joblib
│       └── training_metadata.json
├── reports/
│   └── metrics/
│       └── category_metrics.json
├── notebooks/
│   └── EDA_CFPB.ipynb
├── docs/
│   └── architecture.md
├── scripts/
│   ├── prepare_data.py
│   ├── train_category.py
│   └── evaluate_category.py
├── src/risk_aware/
│   ├── config.py
│   ├── data/prepare.py
│   ├── preprocessing/
│   ├── features/
│   ├── models/category/
│   ├── pipelines/category_training.py
│   ├── evaluation/category_metrics.py
│   ├── inference/category_predictor.py
│   └── utils/
└── tests/
    ├── test_prepare.py
    ├── test_preprocessing.py
    └── test_category_metrics.py
```

## Run

```bash
$env:PYTHONPATH = "src"
python scripts/prepare_data.py
python scripts/train_category.py
python scripts/evaluate_category.py
```
