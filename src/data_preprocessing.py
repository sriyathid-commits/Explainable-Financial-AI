import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_creditcard():
    print("\n" + "=" * 70)
    print("PREPROCESSING CREDIT CARD DATASET")
    print("=" * 70)

    input_file = RAW_DIR / "creditcard.csv"
    output_file = PROCESSED_DIR / "creditcard_clean.csv"

    df = pd.read_csv(input_file)

    print(f"Original shape: {df.shape}")

    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicate_count}")

    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Shape after removing duplicates: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")


def preprocess_paysim():
    print("\n" + "=" * 70)
    print("PREPROCESSING PAYSIM DATASET")
    print("=" * 70)

    input_file = RAW_DIR / "paysim.csv"
    output_file = PROCESSED_DIR / "paysim_clean.csv"

    df = pd.read_csv(input_file)

    print(f"Original shape: {df.shape}")

    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicate_count}")

    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Shape after removing duplicates: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")


if __name__ == "__main__":
    preprocess_creditcard()
    preprocess_paysim()

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)