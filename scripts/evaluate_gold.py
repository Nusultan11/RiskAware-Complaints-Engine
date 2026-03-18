import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
GOLD_PATH=Path("reports/labeling/gold_v1.csv")
MODEL_DIR=Path("artifacts/models")
THRESHOLD_PATH=Path("artifacts/thresholds/priority_threshold.joblib")
REPORT_PATH=Path("reports/metrics/gold_metrics.json")
def main():
        gold_df=pd.read_csv(GOLD_PATH)
        annotator=gold_df["annotator_id"].astype(str).str.strip()
        mask_manual=(annotator!="")
        gold_df=gold_df[mask_manual]
        if len(gold_df)==0:
                raise ValueError("No gold data found")
        else:
                print(f"Found {len(gold_df)} gold examples")
        category_model=joblib.load(MODEL_DIR/"category_model.joblib")
        legal_model=joblib.load(MODEL_DIR/"legal_model.joblib")
        priority_model=joblib.load(MODEL_DIR/"priority_model.joblib")
        threshold_payload=joblib.load(THRESHOLD_PATH)
        threshold_p1=float(threshold_payload["threshold_p1"])
        texts=gold_df["consumer_complaint_narrative"].fillna("").tolist()
        category_proba=category_model.predict_proba(texts)
        category_idx=np.argmax(category_proba,axis=1)
        category_pred=np.array([category_model.labels[i] for i in category_idx],dtype=object)
        category_macro_f1=float(f1_score(y_true=gold_df["category"].astype(str),y_pred=category_pred,average="macro"))
        legal_proba=legal_model.predict_proba(texts)
        y_score_legal=legal_proba[:,1]
        y_true_legal=gold_df["gold_legal_threat"].astype(int).to_numpy()
        legal_pred=(y_score_legal>=0.5).astype(int)
        legal_recall=float(recall_score(y_true_legal,legal_pred))
        legal_pr_auc=float(average_precision_score(y_true_legal,y_score_legal))
        print(category_macro_f1)
        print(legal_recall)
        print(legal_pr_auc)
if __name__ == "__main__":
        main()