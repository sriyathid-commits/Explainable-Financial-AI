import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

datasets = {
    "Credit Card Fraud": "creditcard.csv",
    "PaySim": "paysim.csv"
}

for name, filename in datasets.items():
    path = os.path.join(RAW_DIR, filename)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    if not os.path.exists(path):
        print(f"ERROR: {filename} not found")
        continue

    df = pd.read_csv(path)

    print(f"\nFile: {filename}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nNumeric summary:")
    print(df.describe().T)

    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if categorical:
        print("\nCategorical columns:")
        for col in categorical:
            print(f"\n{col}:")
            print(df[col].value_counts().head(10))

    if "Class" in df.columns:
        print("\nFraud distribution:")
        print(df["Class"].value_counts())
        print("\nFraud percentage:")
        print(df["Class"].value_counts(normalize=True) * 100)

    if "isFraud" in df.columns:
        print("\nPaySim fraud distribution:")
        print(df["isFraud"].value_counts())
        print("\nPaySim fraud percentage:")
        print(df["isFraud"].value_counts(normalize=True) * 100)

print("\n" + "=" * 70)
print("DATASET INSPECTION COMPLETE")
print("=" * 70)