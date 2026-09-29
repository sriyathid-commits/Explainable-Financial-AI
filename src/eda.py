import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def creditcard_eda():
    print("\n" + "=" * 70)
    print("CREDIT CARD EDA")
    print("=" * 70)

    df = pd.read_csv(PROCESSED_DIR / "creditcard_features.csv")

    # 1. Fraud distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Class")
    plt.title("Credit Card Fraud Distribution")
    plt.xlabel("Transaction Class")
    plt.ylabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "creditcard_fraud_distribution.png")
    plt.close()

    # 2. Transaction amount distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df,
        x="Amount",
        bins=100,
        log_scale=True
    )
    plt.title("Credit Card Transaction Amount Distribution")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "creditcard_amount_distribution.png")
    plt.close()

    # 3. Fraud vs amount
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Class", y="Amount")
    plt.yscale("log")
    plt.title("Transaction Amount: Fraud vs Legitimate")
    plt.xlabel("Class")
    plt.ylabel("Amount (log scale)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "creditcard_amount_fraud_comparison.png")
    plt.close()

    # 4. Top correlations with fraud
    correlations = df.corr(numeric_only=True)["Class"].abs()
    correlations = correlations.sort_values(ascending=False)

    top_features = correlations.head(11).index

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        df[top_features].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )
    plt.title("Top Feature Correlations")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "creditcard_correlation_heatmap.png")
    plt.close()

    print("Credit Card EDA completed.")


def paysim_eda():
    print("\n" + "=" * 70)
    print("PAYSIM EDA")
    print("=" * 70)

    df = pd.read_csv(PROCESSED_DIR / "paysim_features.csv")

    # 5. Transaction types
    plt.figure(figsize=(9, 6))
    sns.countplot(
        data=df,
        x="type_CASH_OUT"
    )
    plt.close()

    type_columns = [
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER"
    ]

    type_counts = {
        col.replace("type_", ""): int(df[col].sum())
        for col in type_columns
    }

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=list(type_counts.keys()),
        y=list(type_counts.values())
    )
    plt.title("PaySim Transaction Types")
    plt.xlabel("Transaction Type")
    plt.ylabel("Number of Transactions")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "paysim_transaction_types.png")
    plt.close()

    # 6. Fraud count
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="isFraud")
    plt.title("PaySim Fraud Distribution")
    plt.xlabel("Fraud")
    plt.ylabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "paysim_fraud_distribution.png")
    plt.close()

    # 7. Transaction amount vs fraud
    sample = df.sample(
        min(200000, len(df)),
        random_state=42
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=sample,
        x="isFraud",
        y="amount"
    )
    plt.yscale("log")
    plt.title("PaySim Transaction Amount: Fraud vs Legitimate")
    plt.xlabel("Fraud")
    plt.ylabel("Amount (log scale)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "paysim_amount_fraud_comparison.png")
    plt.close()

    # 8. Fraud by transaction type
    fraud_by_type = {}

    for col in type_columns:
        transaction_type = col.replace("type_", "")
        fraud_by_type[transaction_type] = df.loc[
            df[col] == 1,
            "isFraud"
        ].sum()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=list(fraud_by_type.keys()),
        y=list(fraud_by_type.values())
    )
    plt.title("Fraud Transactions by Transaction Type")
    plt.xlabel("Transaction Type")
    plt.ylabel("Number of Fraud Transactions")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "paysim_fraud_by_type.png")
    plt.close()

    print("PaySim EDA completed.")


if __name__ == "__main__":
    creditcard_eda()
    paysim_eda()

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 70)