import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from joblib import load


BASE_DIR = Path(__file__).resolve().parent.parent

REPORTS_DIR = BASE_DIR / "outputs" / "reports"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
MODEL_DIR = BASE_DIR / "models"


st.set_page_config(
    page_title="Explainable Financial AI",
    page_icon="💳",
    layout="wide"
)


st.title("💳 Explainable AI for Financial Decision-Making")
st.markdown(
    "### Fraud Detection • Explainable AI • Revenue Recovery"
)

st.divider()


@st.cache_data
def load_reports():

    recovery = pd.read_csv(
        REPORTS_DIR / "recovery_predictions.csv"
    )

    decisions = pd.read_csv(
        REPORTS_DIR / "decision_engine_output.csv"
    )

    model_results = pd.read_csv(
        REPORTS_DIR / "model_results.csv"
    )

    shap_importance = pd.read_csv(
        REPORTS_DIR / "shap_feature_importance.csv"
    )

    return (
        recovery,
        decisions,
        model_results,
        shap_importance
    )


recovery, decisions, model_results, shap_importance = (
    load_reports()
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Model Performance",
        "Risk & Recovery",
        "Explainability",
        "Transaction Analyzer"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Financial AI Dashboard")

    total_transactions = len(recovery)

    total_amount = recovery["Amount"].sum()

    avg_risk = (
        recovery["risk_probability"].mean()
    )

    expected_recovery = (
        recovery["expected_recovery"].sum()
    )

    high_priority = (
        decisions["priority"]
        .isin(["HIGH", "CRITICAL"])
        .sum()
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Transaction Value",
        f"₹{total_amount:,.0f}"
    )

    col3.metric(
        "Average Risk",
        f"{avg_risk * 100:.2f}%"
    )

    col4.metric(
        "Expected Recovery",
        f"₹{expected_recovery:,.0f}"
    )

    col5.metric(
        "High Priority",
        f"{high_priority:,}"
    )

    st.divider()

    st.subheader("Recovery Strategy Distribution")

    strategy_counts = (
        recovery["recovery_strategy"]
        .value_counts()
    )

    st.bar_chart(strategy_counts)

    st.subheader("Decision Distribution")

    decision_counts = (
        decisions["engine_action"]
        .value_counts()
    )

    st.bar_chart(decision_counts)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("📈 Fraud Detection Model Performance")

    st.dataframe(
        model_results,
        use_container_width=True
    )

    st.subheader("Evaluation Metrics")

    st.bar_chart(
        model_results.set_index("Model")[
            ["Precision", "Recall", "F1"]
        ]
    )

    st.subheader("ROC-AUC and PR-AUC")

    st.bar_chart(
        model_results.set_index("Model")[
            ["ROC_AUC", "PR_AUC"]
        ]
    )

    st.subheader("Confusion Matrices")

    col1, col2, col3 = st.columns(3)

    with col1:

        image = (
            FIGURES_DIR /
            "logistic_regression_confusion_matrix.png"
        )

        st.image(
            str(image),
            caption="Logistic Regression"
        )

    with col2:

        image = (
            FIGURES_DIR /
            "random_forest_confusion_matrix.png"
        )

        st.image(
            str(image),
            caption="Random Forest"
        )

    with col3:

        image = (
            FIGURES_DIR /
            "xgboost_confusion_matrix.png"
        )

        st.image(
            str(image),
            caption="XGBoost"
        )


# ============================================================
# RISK & RECOVERY
# ============================================================

elif page == "Risk & Recovery":

    st.header("💰 Risk & Revenue Recovery")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Risk Categories")

        risk_counts = (
            recovery["risk_category"]
            .value_counts()
        )

        st.bar_chart(risk_counts)

    with col2:

        st.subheader("Recovery Strategies")

        recovery_counts = (
            recovery["recovery_strategy"]
            .value_counts()
        )

        st.bar_chart(recovery_counts)

    st.divider()

    st.subheader("Expected Recovery")

    st.metric(
        "Estimated Expected Recovery",
        f"₹{recovery['expected_recovery'].sum():,.0f}"
    )

    st.info(
        "Recovery values are model/framework estimates based "
        "on explicit assumptions. The source dataset does not "
        "contain observed recovery outcomes."
    )

    st.subheader("High-Priority Transactions")

    high_priority = decisions[
        decisions["priority"]
        .isin(["HIGH", "CRITICAL"])
    ]

    st.dataframe(
        high_priority[
            [
                "Amount",
                "risk_probability",
                "recovery_probability",
                "expected_value",
                "engine_action",
                "priority"
            ]
        ].head(100),
        use_container_width=True
    )


# ============================================================
# EXPLAINABILITY
# ============================================================

elif page == "Explainability":

    st.header("🧠 Explainable AI")

    st.subheader("Global SHAP Feature Importance")

    shap_image = (
        FIGURES_DIR /
        "shap_feature_importance.png"
    )

    st.image(
        str(shap_image),
        use_container_width=True
    )

    st.subheader("SHAP Summary Plot")

    shap_summary = (
        FIGURES_DIR /
        "shap_summary_plot.png"
    )

    st.image(
        str(shap_summary),
        use_container_width=True
    )

    st.subheader("Top Features")

    st.dataframe(
        shap_importance.head(15),
        use_container_width=True
    )

    st.subheader("LIME Fraud Explanation")

    lime_file = (
        REPORTS_DIR /
        "lime_fraud_explanation.html"
    )

    with open(
        lime_file,
        "r",
        encoding="utf-8"
    ) as file:

        html = file.read()

    st.components.v1.html(
        html,
        height=700,
        scrolling=True
    )


# ============================================================
# TRANSACTION ANALYZER
# ============================================================

elif page == "Transaction Analyzer":

    st.header("🔍 Transaction Risk Analyzer")

    st.write(
        "Enter transaction information to inspect "
        "risk and recovery outputs."
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

    risk = st.slider(
        "Fraud Risk Probability",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01
    )

    recovery_probability = st.slider(
        "Recovery Probability",
        min_value=0.05,
        max_value=0.95,
        value=0.80,
        step=0.01
    )

    if st.button("Analyze Transaction"):

        expected_recovery = (
            amount *
            recovery_probability
        )

        if risk >= 0.90:

            category = "Critical"
            action = "BLOCK_AND_MANUAL_REVIEW"

        elif risk >= 0.70:

            category = "Very High"
            action = "ALTERNATIVE_PAYMENT"

        elif recovery_probability >= 0.70:

            category = "Low / Recoverable"
            action = "RETRY_PAYMENT"

        elif recovery_probability >= 0.50:

            category = "Moderate"
            action = "SEND_REMINDER"

        else:

            category = "Low Recovery"
            action = "NO_ACTION"

        st.divider()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Risk Probability",
            f"{risk * 100:.2f}%"
        )

        col2.metric(
            "Recovery Probability",
            f"{recovery_probability * 100:.2f}%"
        )

        col3.metric(
            "Expected Recovery",
            f"₹{expected_recovery:,.2f}"
        )

        st.subheader("Risk Category")

        st.write(category)

        st.subheader("Recommended Engine Action")

        st.write(action)

        st.info(
            "This analyzer demonstrates the project's "
            "decision framework. It is not a live banking "
            "or payment-processing system."
        )