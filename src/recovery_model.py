import pandas as pd
import numpy as np

from pathlib import Path
from joblib import load


BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_recovery_probability(risk_probability, amount):
    """
    Estimate recovery probability using a transparent
    rule-based assumption.

    This is a proposed recovery model.
    It is NOT a ground-truth label from the dataset.
    """

    if risk_probability >= 0.90:
        base_probability = 0.20

    elif risk_probability >= 0.70:
        base_probability = 0.40

    elif risk_probability >= 0.50:
        base_probability = 0.60

    elif risk_probability >= 0.30:
        base_probability = 0.75

    else:
        base_probability = 0.85

    # Large transactions receive a slightly lower
    # assumed recovery probability.
    if amount > 5000:
        base_probability -= 0.05

    elif amount > 10000:
        base_probability -= 0.10

    return max(0.05, min(base_probability, 0.95))


def select_recovery_strategy(
    risk_probability,
    recovery_probability,
    amount
):
    """
    Select a recovery action using transparent business rules.
    """

    if risk_probability >= 0.90:
        return "Manual Review"

    if risk_probability >= 0.70:
        return "Alternative Payment"

    if recovery_probability >= 0.70 and amount <= 5000:
        return "Retry Payment"

    if recovery_probability >= 0.50:
        return "Send Reminder"

    return "No Action"


def main():

    print("\n" + "=" * 70)
    print("REVENUE RECOVERY MODEL")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------

    data_file = (
        PROCESSED_DIR / "creditcard_features.csv"
    )

    print("\nLoading transaction data...")

    df = pd.read_csv(data_file)

    print(f"Dataset shape: {df.shape}")

    # ---------------------------------------------------------
    # 2. LOAD XGBOOST MODEL
    # ---------------------------------------------------------

    model_file = MODEL_DIR / "xgboost.pkl"

    print("\nLoading trained XGBoost model...")

    model = load(model_file)

    print("XGBoost model loaded successfully.")

    # ---------------------------------------------------------
    # 3. PREPARE FEATURES
    # ---------------------------------------------------------

    X = df.drop(columns=["Class"])

    print(
        f"Number of model features: {X.shape[1]}"
    )

    # ---------------------------------------------------------
    # 4. GENERATE RISK PROBABILITY
    # ---------------------------------------------------------

    print("\nGenerating fraud risk probabilities...")

    risk_probability = model.predict_proba(X)[:, 1]

    df["risk_probability"] = risk_probability

    # ---------------------------------------------------------
    # 5. CALCULATE RISK CATEGORY
    # ---------------------------------------------------------

    df["risk_category"] = pd.cut(
        df["risk_probability"],
        bins=[
            -np.inf,
            0.30,
            0.50,
            0.70,
            0.90,
            np.inf
        ],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Very High",
            "Critical"
        ]
    )

    # ---------------------------------------------------------
    # 6. RECOVERY PROBABILITY
    # ---------------------------------------------------------

    print("\nCalculating recovery probability...")

    df["recovery_probability"] = [
        calculate_recovery_probability(
            risk,
            amount
        )
        for risk, amount
        in zip(
            df["risk_probability"],
            df["Amount"]
        )
    ]

    # ---------------------------------------------------------
    # 7. RECOVERABLE AMOUNT
    # ---------------------------------------------------------

    df["recoverable_amount"] = (
        df["Amount"] *
        df["recovery_probability"]
    )

    # ---------------------------------------------------------
    # 8. EXPECTED RECOVERY
    # ---------------------------------------------------------

    df["expected_recovery"] = (
        df["recoverable_amount"] *
        df["recovery_probability"]
    )

    # ---------------------------------------------------------
    # 9. RECOVERY STRATEGY
    # ---------------------------------------------------------

    print("\nSelecting recovery strategies...")

    df["recovery_strategy"] = [
        select_recovery_strategy(
            risk,
            recovery,
            amount
        )
        for risk, recovery, amount
        in zip(
            df["risk_probability"],
            df["recovery_probability"],
            df["Amount"]
        )
    ]

    # ---------------------------------------------------------
    # 10. DECISION
    # ---------------------------------------------------------

    def make_decision(risk, recovery):

        if risk >= 0.90:
            return "BLOCK / REVIEW"

        if risk >= 0.70:
            return "REVIEW"

        if recovery >= 0.70:
            return "RECOVER"

        return "MONITOR"

    df["decision"] = [
        make_decision(
            risk,
            recovery
        )
        for risk, recovery
        in zip(
            df["risk_probability"],
            df["recovery_probability"]
        )
    ]

    # ---------------------------------------------------------
    # 11. SAVE FULL RECOVERY DATA
    # ---------------------------------------------------------

    output_file = (
        REPORTS_DIR /
        "recovery_predictions.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved: {output_file}"
    )

    # ---------------------------------------------------------
    # 12. SUMMARY
    # ---------------------------------------------------------

    summary = pd.DataFrame({
        "Metric": [
            "Total Transactions",
            "Total Transaction Amount",
            "Average Transaction Amount",
            "Average Risk Probability",
            "Average Recovery Probability",
            "Total Expected Recovery"
        ],
        "Value": [
            len(df),
            df["Amount"].sum(),
            df["Amount"].mean(),
            df["risk_probability"].mean(),
            df["recovery_probability"].mean(),
            df["expected_recovery"].sum()
        ]
    })

    summary_file = (
        REPORTS_DIR /
        "recovery_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    # ---------------------------------------------------------
    # 13. STRATEGY SUMMARY
    # ---------------------------------------------------------

    strategy_summary = (
        df["recovery_strategy"]
        .value_counts()
        .reset_index()
    )

    strategy_summary.columns = [
        "Recovery_Strategy",
        "Transaction_Count"
    ]

    strategy_file = (
        REPORTS_DIR /
        "recovery_strategy_summary.csv"
    )

    strategy_summary.to_csv(
        strategy_file,
        index=False
    )

    # ---------------------------------------------------------
    # 14. DECISION SUMMARY
    # ---------------------------------------------------------

    decision_summary = (
        df["decision"]
        .value_counts()
        .reset_index()
    )

    decision_summary.columns = [
        "Decision",
        "Transaction_Count"
    ]

    decision_file = (
        REPORTS_DIR /
        "decision_summary.csv"
    )

    decision_summary.to_csv(
        decision_file,
        index=False
    )

    # ---------------------------------------------------------
    # 15. PRINT RESULTS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RECOVERY SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("RECOVERY STRATEGIES")
    print("=" * 70)

    print(
        strategy_summary.to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("DECISIONS")
    print("=" * 70)

    print(
        decision_summary.to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("REVENUE RECOVERY MODEL COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")
    print("  recovery_predictions.csv")
    print("  recovery_summary.csv")
    print("  recovery_strategy_summary.csv")
    print("  decision_summary.csv")


if __name__ == "__main__":
    main()