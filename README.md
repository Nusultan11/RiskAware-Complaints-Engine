# RiskAware Complaints Engine

Category-only NLP pipeline for CFPB complaint classification.

## Structure

```text
RiskAware-Complaints-Engine/
|- configs/
|  |- base.yaml
|  |- category.yaml
|- data/
|  |- raw/cfpb_complaints.csv
|  |- processed/{train.csv,val.csv,test.csv,metadata.json}
|- artifacts/
|  |- category/{model.joblib,vectorizer.joblib,label_encoder.joblib,training_metadata.json}
|  |- lstm_preprocessing/{train.npz,val.npz,test.npz,vocab.json,metadata.json}
|  |- category_lstm/{model.pt,training_metadata.json}
|- reports/
|  |- metrics/
|- notebooks/
|  |- EDA_CFPB.ipynb
|- scripts/
|  |- prepare_data.py
|  |- prepare_lstm_data.py
|  |- train_category.py
|  |- train_lstm_category.py
|  |- evaluate_category.py
|  |- tune_category.py
|  |- tune_category_optuna.py
|  |- error_analysis_category.py
|- src/risk_aware/
|  |- data/prepare.py
|  |- preprocessing/{tfidf.py,neural.py}
|  |- models/category/{stacks.py,registry.py}
|  |- pipelines/category_training.py
|  |- evaluation/category_metrics.py
|  |- inference/category_predictor.py
|- tests/
```

## Run

```powershell
$env:PYTHONPATH = "src"
python scripts/prepare_data.py
python scripts/prepare_lstm_data.py
python scripts/train_category.py
python scripts/train_lstm_category.py
python scripts/evaluate_category.py
```

## Notes

- Baseline model: TF-IDF + Logistic Regression.
- LSTM and Transformer stacks are scaffolded; LSTM data preprocessing is now ready.
- Primary metric: Macro-F1.
