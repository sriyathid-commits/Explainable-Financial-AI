import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "deep_learning"
RESULTS_PATH = MODEL_DIR / "deep_learning_results.json"
LOCAL_DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"


st.set_page_config(page_title="Deep Learning", page_icon="🧠", layout="wide")
st.title("Deep Learning Fraud Analysis")
st.warning(
    "Analytical prototype only. This page is NOT connected to live banking or "
    "UPI systems and must not be used to make live transaction decisions."
)

required_files = [
    MODEL_DIR / "lstm_fraud_model.keras",
    MODEL_DIR / "autoencoder.keras",
    MODEL_DIR / "deep_learning_scaler.pkl",
    MODEL_DIR / "feature_columns.json",
    RESULTS_PATH,
]
missing_files = [path.name for path in required_files if not path.is_file()]
if missing_files:
    st.info(
        "Train the models first with `python src\\train_deep_learning.py`. "
        f"Missing artifacts: {', '.join(missing_files)}"
    )
    st.stop()


@st.cache_resource(max_entries=1)
def load_models(model_directory: str):
    model_path = Path(model_directory)
    with (model_path / "feature_columns.json").open(encoding="utf-8") as file:
        feature_columns = json.load(file)
    with (model_path / "deep_learning_results.json").open(encoding="utf-8") as file:
        results = json.load(file)
    return (
        tf.keras.models.load_model(model_path / "lstm_fraud_model.keras"),
        tf.keras.models.load_model(model_path / "autoencoder.keras"),
        joblib.load(model_path / "deep_learning_scaler.pkl"),
        feature_columns,
        results,
    )


@st.cache_data(max_entries=1)
def load_transactions(path: str, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    data = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    expected_columns = [*feature_columns, "Class"]
    missing_columns = [column for column in expected_columns if column not in data]
    if missing_columns:
        raise ValueError(
            "Credit-card CSV is missing required columns: "
            + ", ".join(missing_columns)
        )
    return data


lstm_model, autoencoder, scaler, feature_columns, results = load_models(str(MODEL_DIR))
st.subheader("LSTM fraud-risk model")
lstm = results["lstm"]
lstm_metrics = st.columns(5)
for column, label, key in zip(
    lstm_metrics,
    ("Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"),
    ("precision", "recall", "f1", "roc_auc", "pr_auc"),
):
    column.metric(label, f"{lstm[key]:.4f}")
st.caption(
    f"Measured on {lstm['test_samples']:,} chronological holdout sequences "
    f"of length {lstm['sequence_length']}."
)
st.dataframe(
    pd.DataFrame(
        lstm["confusion_matrix"],
        index=["Actual normal", "Actual fraud"],
        columns=["Predicted normal", "Predicted fraud"],
    ),
    width="stretch",
)

st.subheader("Autoencoder anomaly detection")
autoencoder_results = results["autoencoder"]
autoencoder_metrics = st.columns(3)
for column, label, key in zip(
    autoencoder_metrics,
    ("Precision", "Recall", "F1"),
    ("precision", "recall", "f1"),
):
    column.metric(label, f"{autoencoder_results[key]:.4f}")
st.metric(
    "95th-percentile normal reconstruction-error threshold",
    f"{autoencoder_results['anomaly_threshold']:.8f}",
)
st.caption(
    f"Measured on {autoencoder_results['test_samples']:,} chronological "
    "holdout transactions."
)
st.dataframe(
    pd.DataFrame(
        autoencoder_results["confusion_matrix"],
        index=["Actual normal", "Actual fraud"],
        columns=["Predicted normal", "Predicted anomalous"],
    ),
    width="stretch",
)

data_path = LOCAL_DATA_PATH
if not data_path.is_file():
    recorded_path = Path(results["dataset"]["source_path"])
    if recorded_path.is_file():
        data_path = recorded_path
    else:
        st.error(
            "Credit-card data was not found at the project path or the path "
            "recorded during training."
        )
        st.stop()

transactions = load_transactions(str(data_path), tuple(feature_columns))
st.subheader("Inspect a dataset transaction")
row_index = st.number_input(
    "Transaction row index",
    min_value=0,
    max_value=len(transactions) - 1,
    value=0,
    step=1,
)
transaction = transactions.iloc[int(row_index)]
left, right = st.columns(2)
left.write(f"**Time:** {transaction['Time']}")
right.write(f"**Amount:** {transaction['Amount']}")
st.caption(f"Dataset label for evaluation context: {int(transaction['Class'])}")

scaled_transaction = scaler.transform(
    transaction.loc[feature_columns].to_frame().T
).astype(np.float32)
reconstruction = autoencoder.predict(scaled_transaction, verbose=0)
reconstruction_error = float(
    np.mean(np.square(scaled_transaction - reconstruction), axis=1)[0]
)
threshold = float(autoencoder_results["anomaly_threshold"])
st.metric("Reconstruction error", f"{reconstruction_error:.8f}")
if reconstruction_error >= threshold:
    st.error(
        f"Classified as ANOMALOUS: reconstruction error is at or above the "
        f"saved threshold ({threshold:.8f})."
    )
else:
    st.success(
        f"Classified as NORMAL: reconstruction error is below the saved "
        f"threshold ({threshold:.8f})."
    )
