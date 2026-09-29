import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def engineer_creditcard_features():
    print("\n" + "=" * 70)
    print("CREDIT CARD FEATURE ENGINEERING")
    print("=" * 70)

    input_file = PROCESSED_DIR / "creditcard_clean.csv"
    output_file = PROCESSED_DIR / "creditcard_features.csv"

    df = pd.read_csv(input_file)

    df["Amount_Log"] = __import__("numpy").log1p(df["Amount"])

    df["Time_Hour"] = (df["Time"] / 3600) % 24

    df["Time_Day"] = df["Time"] // 86400

    df.to_csv(output_file, index=False)

    print(f"Original features: {len(df.columns) - 3}")
    print(f"Final features: {len(df.columns)}")
    print(f"Saved: {output_file}")


def engineer_paysim_features():
    print("\n" + "=" * 70)
    print("PAYSIM FEATURE ENGINEERING")
    print("=" * 70)

    input_file = PROCESSED_DIR / "paysim_clean.csv"
    output_file = PROCESSED_DIR / "paysim_features.csv"

    df = pd.read_csv(input_file)

    df["balance_change_orig"] = (
        df["oldbalanceOrg"] - df["newbalanceOrig"]
    )

    df["balance_change_dest"] = (
        df["newbalanceDest"] - df["oldbalanceDest"]
    )

    df["amount_to_orig_balance"] = (
        df["amount"] / (df["oldbalanceOrg"] + 1)
    )

    df["amount_to_dest_balance"] = (
        df["amount"] / (df["oldbalanceDest"] + 1)
    )

    df["orig_balance_error"] = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    )

    df["dest_balance_error"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )

    df["is_large_transaction"] = (
        df["amount"] > df["amount"].quantile(0.95)
    ).astype(int)

    df = pd.get_dummies(
        df,
        columns=["type"],
        prefix="type",
        dtype=int
    )

    df.to_csv(output_file, index=False)

    print(f"Final features: {len(df.columns)}")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    engineer_creditcard_features()
    engineer_paysim_features()

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)