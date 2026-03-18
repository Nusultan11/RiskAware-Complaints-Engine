# RiskAware Architecture (Category-Only)

## Goal

Predict complaint `category` from `consumer_complaint_narrative` using a reproducible, modular NLP pipeline.

## Canonical layout

```text
RiskAware-Complaints-Engine/
├── configs/
│   ├── base.yaml
│   └── category.yaml
├── data/
│   ├── raw/cfpb_complaints.csv
│   └── processed/{train.csv,val.csv,test.csv,metadata.json}
├── artifacts/category/
│   ├── model.joblib
│   ├── vectorizer.joblib
│   ├── label_encoder.joblib
│   └── training_metadata.json
├── reports/metrics/category_metrics.json
├── scripts/
│   ├── prepare_data.py
│   ├── train_category.py
│   └── evaluate_category.py
└── src/risk_aware/
    ├── config.py
    ├── data/prepare.py
    ├── preprocessing/{base.py,tfidf.py,neural.py}
    ├── features/encoders.py
    ├── models/category/{baseline.py,stacks.py,registry.py}
    ├── pipelines/category_training.py
    ├── evaluation/category_metrics.py
    ├── inference/category_predictor.py
    └── utils/{io.py,seed.py,serialization.py}
```

## Pipeline

1. `scripts/prepare_data.py`
2. `scripts/train_category.py`
3. `scripts/evaluate_category.py`

## Design notes

- `TF-IDF + LogisticRegression` is the baseline stack.
- `BiLSTM` and `Transformer` are scaffolded in `models/category/stacks.py`.
- Macro-F1 is the primary metric for model comparison.
