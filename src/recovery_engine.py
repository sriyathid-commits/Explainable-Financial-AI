import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "outputs" / "reports"


def decide_action(row):

    risk = row["risk_probability"]
    recovery = row["recovery_probability"]
    amount = row["Amount"]

    if risk >= 0.90:
        return "BLOCK_AND_MANUAL_REVIEW"

    if risk >= 0.70:
        return "ALTERNATIVE_PAYMENT"

    if recovery >= 0.70 and amount <= 5000:
        return "RETRY_PAYMENT"

    if recovery >= 0.50:
        return "SEND_REMINDER"

    return "NO_ACTION"


def assign_priority(risk):

    if risk >= 0.90:
        return "CRITICAL"

    if risk >= 0.70:
        return "HIGH"

    if risk >= 0.50:
        return "MEDIUM"

    return "LOW"


def main():

    print("\n" + "=" * 70)
    print("DECISION AND RECOVERY ENGINE")
    print("=" * 70)

    input_file = REPORTS_DIR / "recovery_predictions.csv"

    print("\nLoading recovery predictions...")

    df = pd.read_csv(input_file)

    print(f"Transactions loaded: {len(df):,}")

    print("\nApplying decision engine...")

    df["engine_action"] = df.apply(
        decide_action,
        axis=1
    )

    df["priority"] = df["risk_probability"].apply(
        assign_priority
    )

    df["expected_value"] = (
        df["Amount"] *
        df["recovery_probability"]
    )

    output_file = REPORTS_DIR / "decision_engine_output.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print(f"\nSaved: {output_file}")

    action_summary = (
        df["engine_action"]
        .value_counts()
        .reset_index()
    )

    action_summary.columns = [
        "Action",
        "Transaction_Count"
    ]

    action_file = REPORTS_DIR / "engine_action_summary.csv"

    action_summary.to_csv(
        action_file,
        index=False
    )

    priority_summary = (
        df["priority"]
        .value_counts()
        .reset_index()
    )

    priority_summary.columns = [
        "Priority",
        "Transaction_Count"
    ]

    priority_file = REPORTS_DIR / "priority_summary.csv"

    priority_summary.to_csv(
        priority_file,
        index=False
    )

    high_priority = df[
        df["priority"].isin(
            ["CRITICAL", "HIGH"]
        )
    ].copy()

    high_priority = high_priority.sort_values(
        by="expected_value",
        ascending=False
    )

    high_priority_file = (
        REPORTS_DIR /
        "high_priority_transactions.csv"
    )

    high_priority.head(1000).to_csv(
        high_priority_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("ACTION SUMMARY")
    print("=" * 70)

    print(
        action_summary.to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("PRIORITY SUMMARY")
    print("=" * 70)

    print(
        priority_summary.to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("HIGH-PRIORITY TRANSACTIONS")
    print("=" * 70)

    print(
        f"High-priority cases: {len(high_priority):,}"
    )

    print(
        f"Top 1,000 saved to: {high_priority_file}"
    )

    print("\n" + "=" * 70)
    print("DECISION ENGINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()