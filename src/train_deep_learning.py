"""Train LSTM fraud-risk and autoencoder anomaly-detection models."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import dump
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models" / "deep_learning"
SEQUENCE_LENGTH = 10
RANDOM_SEED = 42


def build_lstm_model(feature_count: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, feature_count)),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def build_autoencoder(feature_count: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(feature_count,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(feature_count, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
    }


def train(data_path: Path) -> dict:
    if not data_path.is_file():
        raise FileNotFoundError(f"Credit-card dataset not found: {data_path}")

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    data = pd.read_csv(data_path)
    if "Class" not in data.columns:
        raise ValueError("Dataset must contain a 'Class' target column.")

    original_rows = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    duplicates_removed = original_rows - len(data)
    feature_columns = [column for column in data.columns if column != "Class"]

    features = data[feature_columns]
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("All transaction feature columns must be numeric.")
    if features.isna().any().any():
        raise ValueError("Dataset contains missing feature values.")

    labels = data["Class"].to_numpy(dtype=np.int32)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("'Class' must contain only binary 0/1 labels.")

    split_at = int(len(data) * 0.8)
    if split_at <= SEQUENCE_LENGTH or split_at >= len(data):
        raise ValueError("Dataset is too small for the chronological train/test split.")
    y_train_rows = labels[:split_at]
    y_test_rows = labels[split_at:]
    if np.unique(y_train_rows).size != 2 or np.unique(y_test_rows).size != 2:
        raise ValueError(
            "Chronological 80/20 split must contain both classes in train and test."
        )

    scaler = StandardScaler()
    scaler.fit(features.iloc[:split_at])
    scaled_features = scaler.transform(features).astype(np.float32)

    # Each sequence predicts the label of its final transaction; the chronological
    # split keeps test labels out of training while allowing prior rows as context.
    windows = np.lib.stride_tricks.sliding_window_view(
        scaled_features, SEQUENCE_LENGTH, axis=0
    ).transpose(0, 2, 1)
    target_labels = labels[SEQUENCE_LENGTH - 1 :]
    train_window_count = split_at - SEQUENCE_LENGTH + 1
    x_train = np.ascontiguousarray(windows[:train_window_count])
    y_train = target_labels[:train_window_count]
    x_test = np.ascontiguousarray(windows[train_window_count:])
    y_test = target_labels[train_window_count:]
    if np.unique(y_test).size != 2:
        raise ValueError("The sequence test split must contain both classes.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(scaler, MODEL_DIR / "deep_learning_scaler.pkl")
    (MODEL_DIR / "feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2), encoding="utf-8"
    )

    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    if positive_count == 0:
        raise ValueError("Training data must contain fraud examples.")
    class_weights = {
        0: 1.0,
        1: float(negative_count / positive_count),
    }

    lstm_model = build_lstm_model(len(feature_columns))
    lstm_history = lstm_model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=512,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ],
        verbose=2,
    )
    lstm_probabilities = lstm_model.predict(x_test, batch_size=1024, verbose=0).ravel()
    lstm_predictions = (lstm_probabilities >= 0.5).astype(np.int32)
    lstm_results = classification_metrics(y_test, lstm_predictions)
    lstm_results.update(
        {
            "roc_auc": float(roc_auc_score(y_test, lstm_probabilities)),
            "pr_auc": float(average_precision_score(y_test, lstm_probabilities)),
            "sequence_length": SEQUENCE_LENGTH,
            "test_samples": int(len(y_test)),
            "epochs_trained": len(lstm_history.epoch),
        }
    )
    lstm_model.save(MODEL_DIR / "lstm_fraud_model.keras")

    normal_train = scaled_features[:split_at][y_train_rows == 0]
    validation_at = int(len(normal_train) * 0.9)
    if validation_at < 1 or validation_at >= len(normal_train):
        raise ValueError("Not enough normal training data for autoencoder validation.")
    autoencoder_train = normal_train[:validation_at]
    autoencoder_validation = normal_train[validation_at:]

    autoencoder = build_autoencoder(len(feature_columns))
    autoencoder_history = autoencoder.fit(
        autoencoder_train,
        autoencoder_train,
        validation_data=(autoencoder_validation, autoencoder_validation),
        epochs=100,
        batch_size=512,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ],
        verbose=2,
    )
    validation_reconstruction = autoencoder.predict(
        autoencoder_validation, batch_size=2048, verbose=0
    )
    validation_errors = np.mean(
        np.square(autoencoder_validation - validation_reconstruction), axis=1
    )
    anomaly_threshold = float(np.percentile(validation_errors, 95))

    test_rows = scaled_features[split_at:]
    test_reconstruction = autoencoder.predict(test_rows, batch_size=2048, verbose=0)
    test_errors = np.mean(np.square(test_rows - test_reconstruction), axis=1)
    anomaly_predictions = (test_errors >= anomaly_threshold).astype(np.int32)
    autoencoder_results = classification_metrics(y_test_rows, anomaly_predictions)
    autoencoder_results.update(
        {
            "anomaly_threshold": anomaly_threshold,
            "threshold_percentile": 95,
            "test_samples": int(len(y_test_rows)),
            "epochs_trained": len(autoencoder_history.epoch),
        }
    )
    autoencoder.save(MODEL_DIR / "autoencoder.keras")

    results = {
        "dataset": {
            "source_path": str(data_path.resolve()),
            "rows_before_deduplication": int(original_rows),
            "duplicates_removed": int(duplicates_removed),
            "rows_after_deduplication": int(len(data)),
            "features": int(len(feature_columns)),
            "chronological_train_rows": int(split_at),
            "chronological_test_rows": int(len(data) - split_at),
        },
        "lstm": lstm_results,
        "autoencoder": autoencoder_results,
    }
    (MODEL_DIR / "deep_learning_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    print(f"\nModels and results saved to: {MODEL_DIR}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"CSV path (default: {DEFAULT_DATA_PATH})",
    )
    args = parser.parse_args()
    train(args.data_path)


if __name__ == "__main__":
    main()
