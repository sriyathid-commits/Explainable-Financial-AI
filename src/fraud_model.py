import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    cm = confusion_matrix(y_test, predictions)

    print("\nConfusion Matrix:")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    display.plot()
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()

    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(FIGURES_DIR / filename)
    plt.close()

    return {
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc
    }


def train_creditcard_models():

    print("\n" + "=" * 70)
    print("CREDIT CARD FRAUD MODEL TRAINING")
    print("=" * 70)

    file_path = PROCESSED_DIR / "creditcard_features.csv"

    df = pd.read_csv(file_path)

    print(f"Dataset shape: {df.shape}")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")
    print(f"Fraud in training: {y_train.sum()}")
    print(f"Fraud in testing : {y_test.sum()}")

    # ---------------------------------------------------------
    # Logistic Regression
    # ---------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    print("\nTraining Logistic Regression...")

    logistic_model.fit(
        X_train_scaled,
        y_train
    )

    logistic_results = evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_test_scaled,
        y_test
    )

    dump(
        logistic_model,
        MODEL_DIR / "logistic_regression.pkl"
    )

    dump(
        scaler,
        MODEL_DIR / "standard_scaler.pkl"
    )

    # ---------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------

    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    print("\nTraining Random Forest...")

    random_forest.fit(
        X_train,
        y_train
    )

    rf_results = evaluate_model(
        "Random Forest",
        random_forest,
        X_test,
        y_test
    )

    dump(
        random_forest,
        MODEL_DIR / "random_forest.pkl"
    )

    # ---------------------------------------------------------
    # XGBoost
    # ---------------------------------------------------------

    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()

    scale_pos_weight = negative / positive

    print(
        f"\nXGBoost scale_pos_weight: "
        f"{scale_pos_weight:.2f}"
    )

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    )

    print("\nTraining XGBoost...")

    xgb_model.fit(
        X_train,
        y_train
    )

    xgb_results = evaluate_model(
        "XGBoost",
        xgb_model,
        X_test,
        y_test
    )

    dump(
        xgb_model,
        MODEL_DIR / "xgboost.pkl"
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    results = pd.DataFrame([
        logistic_results,
        rf_results,
        xgb_results
    ])

    results_file = (
        BASE_DIR
        / "outputs"
        / "reports"
        / "model_results.csv"
    )

    results_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        results_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("MODEL RESULTS")
    print("=" * 70)

    print(results.to_string(index=False))

    print(
        f"\nResults saved to: {results_file}"
    )

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    train_creditcard_models()