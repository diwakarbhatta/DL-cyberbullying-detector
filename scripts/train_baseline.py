"""
Baseline Model: TF-IDF + Logistic Regression
Provides the benchmark score to beat.
Evaluates using Macro-F1, Weighted-F1, and Confusion Matrix.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessor import clean_text, normalize_leetspeak

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

def preprocess_series(series: pd.Series) -> pd.Series:
    return series.astype(str).apply(lambda t: normalize_leetspeak(clean_text(t)))

def train_and_evaluate_baseline():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 60)
    print("TRAINING BASELINE MODEL: TF-IDF + LOGISTIC REGRESSION")
    print("=" * 60)

    train_path = os.path.join(DATA_DIR, "train.csv")
    val_path = os.path.join(DATA_DIR, "val.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"Loaded {len(train_df):,} train, {len(val_df):,} val, {len(test_df):,} test samples.")

    print("\nPreprocessing text...")
    X_train = preprocess_series(train_df["tweet_text"])
    y_train = train_df["cyberbullying_type"]

    X_val = preprocess_series(val_df["tweet_text"])
    y_val = val_df["cyberbullying_type"]

    X_test = preprocess_series(test_df["tweet_text"])
    y_test = test_df["cyberbullying_type"]

    classes = sorted(y_train.unique().tolist())
    print(f"Classes ({len(classes)}): {classes}")

    # Build Pipeline
    print("\nBuilding TF-IDF + Logistic Regression pipeline...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.5,
            random_state=42
        ))
    ])

    print("Fitting model on training data...")
    pipeline.fit(X_train, y_train)

    # Evaluate on Validation & Test
    val_preds = pipeline.predict(X_val)
    val_macro_f1 = f1_score(y_val, val_preds, average="macro")
    print(f"Validation Macro-F1: {val_macro_f1:.4f}")

    print("\nEvaluating on Test Set...")
    test_preds = pipeline.predict(X_test)
    test_probs = pipeline.predict_proba(X_test)

    macro_f1 = f1_score(y_test, test_preds, average="macro")
    weighted_f1 = f1_score(y_test, test_preds, average="weighted")
    acc = accuracy_score(y_test, test_preds)

    print("-" * 60)
    print(f"BENCHMARK BASELINE RESULTS:")
    print(f"  • Macro-F1 Score:    {macro_f1:.4f}  <-- SCORE TO BEAT")
    print(f"  • Weighted-F1 Score: {weighted_f1:.4f}")
    print(f"  • Accuracy:          {acc:.4f} ({acc*100:.2f}%)")
    print("-" * 60)

    # Detailed Classification Report
    report_dict = classification_report(y_test, test_preds, target_names=classes, output_dict=True)
    report_text = classification_report(y_test, test_preds, target_names=classes, digits=4)
    print("\nClassification Report:\n", report_text)

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"Actual_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])
    print("\nConfusion Matrix:\n", cm_df)

    # Save Pipeline and Metrics
    model_save_path = os.path.join(MODELS_DIR, "baseline_tfidf_lr.joblib")
    joblib.dump(pipeline, model_save_path)
    print(f"\nModel pipeline saved to: {model_save_path}")

    metrics_save_path = os.path.join(REPORTS_DIR, "baseline_metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump({
            "model": "TF-IDF + Logistic Regression (Baseline)",
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "accuracy": round(acc, 4),
            "classes": classes,
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict
        }, f, indent=2)

    cm_save_path = os.path.join(REPORTS_DIR, "baseline_confusion_matrix.csv")
    cm_df.to_csv(cm_save_path)
    print(f"Metrics saved to: {metrics_save_path} and {cm_save_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_and_evaluate_baseline()
