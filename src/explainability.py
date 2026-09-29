import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from pathlib import Path
from joblib import load
from lime.lime_tabular import LimeTabularExplainer


BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("\n" + "=" * 70)
    print("EXPLAINABLE AI - SHAP + LIME")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------

    data_file = PROCESSED_DIR / "creditcard_features.csv"

    print("\nLoading dataset...")

    df = pd.read_csv(data_file)

    print(f"Dataset shape: {df.shape}")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    feature_names = X.columns.tolist()

    print(f"Number of features: {len(feature_names)}")

    # ---------------------------------------------------------
    # 2. LOAD TRAINED XGBOOST MODEL
    # ---------------------------------------------------------

    model_file = MODEL_DIR / "xgboost.pkl"

    print("\nLoading XGBoost model...")

    model = load(model_file)

    print("XGBoost model loaded successfully.")

    # ---------------------------------------------------------
    # 3. SELECT SAMPLE FOR SHAP
    # ---------------------------------------------------------

    print("\nSelecting samples for SHAP...")

    sample_size = min(2000, len(X))

    X_sample = X.sample(
        n=sample_size,
        random_state=42
    )

    print(f"SHAP sample size: {len(X_sample)}")

    # ---------------------------------------------------------
    # 4. SHAP TREE EXPLAINER
    # ---------------------------------------------------------

    print("\nCreating SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    # ---------------------------------------------------------
    # 5. HANDLE SHAP OUTPUT FORMAT
    # ---------------------------------------------------------

    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]
    else:
        shap_values_plot = shap_values

    print("SHAP values calculated.")

    # ---------------------------------------------------------
    # 6. SHAP SUMMARY DOT PLOT
    # ---------------------------------------------------------

    print("\nCreating SHAP summary plot...")

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values_plot,
        X_sample,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "shap_summary_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved: shap_summary_plot.png")

    # ---------------------------------------------------------
    # 7. SHAP BAR PLOT
    # ---------------------------------------------------------

    print("\nCreating SHAP feature importance plot...")

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values_plot,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved: shap_feature_importance.png")

    # ---------------------------------------------------------
    # 8. CALCULATE GLOBAL FEATURE IMPORTANCE
    # ---------------------------------------------------------

    print("\nCalculating global feature importance...")

    mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": mean_abs_shap
    })

    importance_df = importance_df.sort_values(
        by="Mean_Absolute_SHAP",
        ascending=False
    )

    importance_file = (
        REPORTS_DIR / "shap_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_file,
        index=False
    )

    print(
        f"Saved: {importance_file}"
    )

    print("\nTop 10 features:")

    print(
        importance_df.head(10).to_string(index=False)
    )

    # ---------------------------------------------------------
    # 9. FIND ONE FRAUD TRANSACTION
    # ---------------------------------------------------------

    print("\nFinding fraud transaction...")

    fraud_indices = df.index[df["Class"] == 1]

    fraud_index = fraud_indices[0]

    fraud_row = X.loc[[fraud_index]]

    fraud_shap_values = explainer.shap_values(
        fraud_row
    )

    if isinstance(fraud_shap_values, list):
        fraud_shap_values = fraud_shap_values[1]

    fraud_shap_values = np.asarray(
        fraud_shap_values
    ).reshape(-1)

    fraud_explanation = pd.DataFrame({
        "Feature": feature_names,
        "Feature_Value": fraud_row.iloc[0].values,
        "SHAP_Value": fraud_shap_values
    })

    fraud_explanation["Absolute_SHAP"] = (
        fraud_explanation["SHAP_Value"].abs()
    )

    fraud_explanation = fraud_explanation.sort_values(
        by="Absolute_SHAP",
        ascending=False
    )

    fraud_file = (
        REPORTS_DIR / "fraud_transaction_shap.csv"
    )

    fraud_explanation.to_csv(
        fraud_file,
        index=False
    )

    print(
        f"Saved: {fraud_file}"
    )

    # ---------------------------------------------------------
    # 10. FIND ONE LEGITIMATE TRANSACTION
    # ---------------------------------------------------------

    print("\nFinding legitimate transaction...")

    legitimate_indices = df.index[df["Class"] == 0]

    legitimate_index = legitimate_indices[0]

    legitimate_row = X.loc[[legitimate_index]]

    legitimate_shap_values = explainer.shap_values(
        legitimate_row
    )

    if isinstance(legitimate_shap_values, list):
        legitimate_shap_values = legitimate_shap_values[1]

    legitimate_shap_values = np.asarray(
        legitimate_shap_values
    ).reshape(-1)

    legitimate_explanation = pd.DataFrame({
        "Feature": feature_names,
        "Feature_Value": legitimate_row.iloc[0].values,
        "SHAP_Value": legitimate_shap_values
    })

    legitimate_explanation["Absolute_SHAP"] = (
        legitimate_explanation["SHAP_Value"].abs()
    )

    legitimate_explanation = legitimate_explanation.sort_values(
        by="Absolute_SHAP",
        ascending=False
    )

    legitimate_file = (
        REPORTS_DIR / "legitimate_transaction_shap.csv"
    )

    legitimate_explanation.to_csv(
        legitimate_file,
        index=False
    )

    print(
        f"Saved: {legitimate_file}"
    )

    # ---------------------------------------------------------
    # 11. LIME EXPLAINER
    # ---------------------------------------------------------

    print("\nCreating LIME explainer...")

    lime_sample = X.sample(
        n=min(5000, len(X)),
        random_state=42
    )

    lime_explainer = LimeTabularExplainer(
        training_data=lime_sample.values,
        feature_names=feature_names,
        class_names=["Legitimate", "Fraud"],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )

    # ---------------------------------------------------------
    # 12. LIME FRAUD EXPLANATION
    # ---------------------------------------------------------

    print("\nCreating LIME fraud explanation...")

    fraud_instance = fraud_row.iloc[0].values

    lime_explanation = lime_explainer.explain_instance(
        fraud_instance,
        model.predict_proba,
        num_features=10
    )

    lime_file = (
        REPORTS_DIR / "lime_fraud_explanation.csv"
    )

    lime_data = []

    for feature, weight in lime_explanation.as_list(
        label=1
    ):
        lime_data.append({
            "Feature": feature,
            "Weight": weight
        })

    lime_df = pd.DataFrame(lime_data)

    lime_df.to_csv(
        lime_file,
        index=False
    )

    print(
        f"Saved: {lime_file}"
    )

    # ---------------------------------------------------------
    # 13. SAVE LIME HTML EXPLANATION
    # ---------------------------------------------------------

    lime_html = (
        REPORTS_DIR / "lime_fraud_explanation.html"
    )

    lime_explanation.save_to_file(
        str(lime_html)
    )

    print(
        f"Saved: {lime_html}"
    )

    # ---------------------------------------------------------
    # 14. FINAL SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPLAINABLE AI COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")

    print("\nFigures:")
    print("  shap_summary_plot.png")
    print("  shap_feature_importance.png")

    print("\nReports:")
    print("  shap_feature_importance.csv")
    print("  fraud_transaction_shap.csv")
    print("  legitimate_transaction_shap.csv")
    print("  lime_fraud_explanation.csv")
    print("  lime_fraud_explanation.html")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()