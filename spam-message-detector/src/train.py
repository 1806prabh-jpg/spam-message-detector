"""
Training script for the Spam Message Detector.

This script:
1.  Loads the SMS Spam dataset from data/spam.csv
2.  Preprocesses the text (lowercase, special-char removal, stopwords, ...)
3.  Splits the data into train/test sets (stratified)
4.  Builds a scikit-learn Pipeline that bundles TF-IDF + classifier
5.  Trains two classifiers (Multinomial Naive Bayes, Logistic Regression)
6.  Evaluates both (accuracy, precision, recall, F1, confusion matrix)
7.  Saves the best pipeline to models/spam_classifier.pkl
8.  Saves evaluation metrics + confusion-matrix plot to models/

Run:  python -m src.train    (from the project root)
"""

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Make `src` importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess_text  # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "spam_classifier.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.txt")
CM_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
CM_NB_PATH = os.path.join(MODEL_DIR, "confusion_matrix_nb.png")
CM_LR_PATH = os.path.join(MODEL_DIR, "confusion_matrix_lr.png")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the SMS spam dataset.

    Expected CSV columns: label (ham/spam), message (raw text).
    """
    if not os.path.exists(path):
        print(f"ERROR: Dataset not found at {path}")
        print("Please place the SMS Spam Collection dataset as data/spam.csv")
        sys.exit(1)

    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")

    # Handle different column naming conventions
    if "label" not in df.columns or "message" not in df.columns:
        # Try common alternatives
        if df.shape[1] >= 2:
            df.columns = ["label", "message"] + list(df.columns[2:])
        else:
            print("ERROR: Dataset does not have the expected columns (label, message).")
            sys.exit(1)

    # Normalise labels to lowercase
    df["label"] = df["label"].str.lower().str.strip()
    # Keep only ham and spam rows
    df = df[df["label"].isin(["ham", "spam"])].copy()
    # Create binary target: spam=1, ham=0
    df["target"] = (df["label"] == "spam").astype(int)
    # Preprocess the message text
    df["clean_message"] = df["message"].apply(preprocess_text)
    # Drop any rows that became empty after preprocessing
    df = df[df["clean_message"].str.len() > 0].reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------
def build_pipeline(classifier_name: str) -> Pipeline:
    """
    Build a scikit-learn Pipeline: TfidfVectorizer -> Classifier.

    Using a Pipeline guarantees that the TF-IDF vectorizer is fitted
    ONLY on the training data (no data leakage from the test set).
    The same fitted vectorizer is then applied to the test set.
    """
    tfidf = TfidfVectorizer(
        max_features=5000,       # keep the top 5,000 terms by frequency
        ngram_range=(1, 2),      # unigrams + bigrams (captures "free entry")
        min_df=2,                # ignore terms that appear in < 2 documents
        max_df=0.95,             # ignore terms in > 95% of docs (corpus-specific)
        sublinear_tf=True,       # use 1 + log(tf) instead of raw tf
    )

    if classifier_name == "naive_bayes":
        clf = MultinomialNB(alpha=0.1)   # small smoothing alpha
    elif classifier_name == "logistic_regression":
        clf = LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",     # handle class imbalance
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    return Pipeline([("tfidf", tfidf), ("clf", clf)])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model: Pipeline, X_test, y_test, name: str) -> dict:
    """Compute metrics and return them as a dict."""
    y_pred = model.predict(X_test)

    # For probability/confidence
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except (AttributeError, NotImplementedError):
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  [[TN={cm[0,0]:>4}  FP={cm[0,1]:>4}]")
    print(f"   [FN={cm[1,0]:>4}  TP={cm[1,1]:>4}]]")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    return {
        "name": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "y_proba": y_proba,
    }


def save_confusion_matrix(cm, name: str, path: str):
    """Render and save a confusion-matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Ham", "Spam"])
    ax.set_yticklabels(["Ham", "Spam"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    # Annotate cells
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    df = load_data()
    print(f"  Loaded {len(df)} messages")
    print(f"  Spam: {df['target'].sum()}  |  Ham: {(df['target'] == 0).sum()}")

    X = df["clean_message"]
    y = df["target"]

    # Stratified split preserves the spam/ham ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ---- Train both models ------------------------------------------------
    results = {}
    for clf_name, label in [
        ("naive_bayes", "Multinomial Naive Bayes"),
        ("logistic_regression", "Logistic Regression"),
    ]:
        print(f"\nTraining {label}...")
        pipe = build_pipeline(clf_name)
        pipe.fit(X_train, y_train)
        res = evaluate(pipe, X_test, y_test, label)
        results[clf_name] = {"pipeline": pipe, "metrics": res}
        # Save individual confusion matrix
        cm_path = CM_NB_PATH if clf_name == "naive_bayes" else CM_LR_PATH
        save_confusion_matrix(res["confusion_matrix"], label, cm_path)

    # ---- Select the best model by F1-score -------------------------------
    best_clf = max(results, key=lambda k: results[k]["metrics"]["f1"])
    best_name = results[best_clf]["metrics"]["name"]
    best_pipe = results[best_clf]["pipeline"]
    best_metrics = results[best_clf]["metrics"]

    print(f"\n{'='*60}")
    print(f"  BEST MODEL: {best_name} (F1={best_metrics['f1']:.4f})")
    print(f"{'='*60}")

    # ---- Save the best pipeline ------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "pipeline": best_pipe,
            "model_name": best_name,
            "label_map": {0: "NOT SPAM", 1: "SPAM"},
        }, f)
    print(f"  Saved model to {MODEL_PATH}")

    # ---- Save the main confusion matrix ----------------------------------
    save_confusion_matrix(
        best_metrics["confusion_matrix"], best_name, CM_PATH
    )

    # ---- Save metrics to a text file -------------------------------------
    with open(METRICS_PATH, "w") as f:
        f.write("Spam Message Detector — Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {DATA_PATH}\n")
        f.write(f"Total messages: {len(df)}\n")
        f.write(f"Spam: {int(df['target'].sum())}  |  Ham: {int((df['target'] == 0).sum())}\n")
        f.write(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}\n\n")

        for clf_name, data in results.items():
            m = data["metrics"]
            f.write(f"{m['name']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Accuracy : {m['accuracy']:.4f}\n")
            f.write(f"  Precision: {m['precision']:.4f}\n")
            f.write(f"  Recall   : {m['recall']:.4f}\n")
            f.write(f"  F1-score : {m['f1']:.4f}\n")
            cm = m["confusion_matrix"]
            f.write(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}\n\n")

        f.write(f"Selected model: {best_name}\n")
    print(f"  Saved metrics to {METRICS_PATH}")

    return results


if __name__ == "__main__":
    main()
