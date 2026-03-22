# RiskAware Complaints Engine

Production-style NLP pipeline for CFPB complaint **category** classification with three model families:

- TF-IDF + Logistic Regression (main baseline)
- BiLSTM (DL baseline)
- DistilBERT (transformer comparison model)

## Final Data Pipeline

Implemented in [`src/risk_aware/data/prepare.py`](C:/Users/nurs/OneDrive/Рабочий стол/RiskAware Complaints Engine/src/risk_aware/data/prepare.py):

- `MIN_CLASS_COUNT = 5`
- conflicting `text_key` rows removed
- group-aware split by `text_key` (`train/val/test`) to prevent leakage

Processed outputs:

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/metadata.json`

## Preprocessing

- TF-IDF: [`src/risk_aware/preprocessing/tfidf.py`](C:/Users/nurs/OneDrive/Рабочий стол/RiskAware Complaints Engine/src/risk_aware/preprocessing/tfidf.py)
  - anonymization `xxxx` treated as noise
  - digits normalized to `num`
  - vectorizer uses `sublinear_tf=True`
- Neural: [`src/risk_aware/preprocessing/neural.py`](C:/Users/nurs/OneDrive/Рабочий стол/RiskAware Complaints Engine/src/risk_aware/preprocessing/neural.py)
  - `xxxx -> <anon>`
  - digits -> `<num>`
  - contractions preserved (apostrophe retained)

## Training Entry Points

```powershell
$env:PYTHONPATH = "src"
python scripts/prepare_data.py
python scripts/train_category.py
python scripts/evaluate_category.py
python scripts/prepare_lstm_data.py
python scripts/train_lstm_category.py
python scripts/train_transformer_category.py
python scripts/generate_final_comparison.py
```

`scripts/*` are thin launchers; core training logic lives in `src/risk_aware/pipelines/*`.

## Final Model Comparison (test)

See:

- `reports/final/model_comparison.csv`
- `reports/final/model_comparison.json`

Primary selection metric: **Macro-F1**.

Current role assignment:

- **Main model**: TF-IDF + Logistic Regression
- **DL baseline**: BiLSTM
- **Transformer comparison**: DistilBERT (len256/len384 experiments)
