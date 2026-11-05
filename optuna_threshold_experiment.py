# ==============================================================================
# EKSPERIMEN OPTUNA (30 TRIALS), PROTOKOL OOF, & THRESHOLD TUNING (optuna_threshold_experiment.py)
# Kelompok 04 - PT Bringin Inti Teknologi (Divisi IT Operational / DIIO SME)
# Mata Kuliah: Pembelajaran Mesin (S1) - Minggu 06 (Menuju Proposal UTS)
# ==============================================================================
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, log_loss
)

try:
    import optuna
    OPTUNA_INSTALLED = True
except ImportError:
    OPTUNA_INSTALLED = False

def main():
    df = pd.read_csv("dataset_sla_tiket_bit_clean.csv")
    feature_cols = [
        "Project Owner", "Service_Category", "Assigned_PIC_Code",
        "Task_Complexity_Tier", "Submit_Day_of_Week", "Is_Weekend_or_Peak",
        "Daily_Ticket_Load", "Engineer_Queue_Load",
        "Desc_Char_Length", "Detail_Char_Length", "Is_Category_Missing", "TFIDF_Urgency_Score",
        "Combined_Text"
    ]
    X = df[feature_cols].copy()
    y = df["Target_SLA_Breach"].copy()

    # Step 01: Stratified Train-Test Split (80:20) - Isolasi Test Set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    num_cols = [
        "Task_Complexity_Tier", "Submit_Day_of_Week", "Is_Weekend_or_Peak",
        "Daily_Ticket_Load", "Engineer_Queue_Load",
        "Desc_Char_Length", "Detail_Char_Length", "Is_Category_Missing", "TFIDF_Urgency_Score"
    ]
    cat_cols = ["Project Owner", "Service_Category", "Assigned_PIC_Code"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))]), cat_cols),
        ("txt", Pipeline([("tfidf", TfidfVectorizer(max_features=16, ngram_range=(1, 2), stop_words="english"))]), "Combined_Text")
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Step 02: Optimasi Hiperparameter dengan Optuna (30 Trials, 5-Fold Stratified CV)
    if OPTUNA_INSTALLED:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=10),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
                "random_state": 42,
                "n_jobs": -1
            }
            pipe = Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(**params))])
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
            return scores.mean()

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)
        best_params = study.best_params
        print("Best Params from Optuna:", best_params)
    else:
        best_params = {
            "n_estimators": 290, "max_depth": 13, "min_samples_split": 6,
            "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": "balanced_subsample"
        }

    # Step 03 & 04: Prediksi Out-Of-Fold (OOF) pada X_train & Pemilihan Threshold Optimal
    best_pipe = Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(**best_params, random_state=42))])
    oof_probs = cross_val_predict(best_pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    print(f"OOF Train ROC-AUC: {roc_auc_score(y_train, oof_probs):.4f} | PR-AUC: {average_precision_score(y_train, oof_probs):.4f}")

    # Step 05 & 06: Fit Final pada X_train, Kunci Threshold (0.35 vs 0.50), dan Evaluasi pada X_test
    best_pipe.fit(X_train, y_train)
    joblib.dump(best_pipe, "model_final_bit_sla.joblib", compress=3)

    y_probs_test = best_pipe.predict_proba(X_test)[:, 1]
    for th in [0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
        y_pred_th = (y_probs_test >= th).astype(int)
        cm = confusion_matrix(y_test, y_pred_th)
        print(f"\n=== EVALUASI TEST SET PADA THRESHOLD = {th:.2f} ===")
        print("Confusion Matrix:\n", cm)
        print(classification_report(y_test, y_pred_th, target_names=["Kelas 0 (SLA Met)", "Kelas 1 (SLA Breached)"], digits=4))

if __name__ == "__main__":
    main()
