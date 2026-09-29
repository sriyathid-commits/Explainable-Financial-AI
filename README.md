# Explainable Financial AI

### Explainable AI for Fraud Detection, Financial Risk Analysis & Revenue Recovery

> An end-to-end machine learning platform that combines fraud detection, explainable AI, financial risk analysis, recovery intelligence, and automated decision-making.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)]()
[![SHAP](https://img.shields.io/badge/XAI-SHAP-purple)]()
[![LIME](https://img.shields.io/badge/XAI-LIME-green)]()
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)]()
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)]()

---

## Overview

**Explainable Financial AI** is an end-to-end financial intelligence system designed to detect suspicious transactions, estimate financial risk, explain model decisions, and generate recovery-oriented actions.

The project combines:

* Fraud detection
* Machine learning
* Explainable AI
* Financial behavior analysis
* Revenue recovery modeling
* Rule-based decision intelligence
* Interactive analytics

The current version uses public financial datasets for model development and evaluation.

The architecture is designed to evolve toward a **real-time transaction intelligence platform**, including future support for authorized payment-provider APIs, webhooks, streaming transaction events, and UPI-focused risk analysis.

---

# Product Vision

Traditional financial ML systems often answer:

> **"Is this transaction risky?"**

This project aims to answer a broader set of questions:

> **Why is it risky?**

> **What financial impact could it have?**

> **What action should the system consider?**

> **How can the decision be explained to a human operator?**

The system therefore follows:

```text
Transaction
     ↓
Data Preprocessing
     ↓
Feature Engineering
     ↓
Risk / Fraud Prediction
     ↓
Explainable AI
     ↓
Recovery Intelligence
     ↓
Decision Engine
     ↓
Dashboard
```

---

# Key Features

## 1. Fraud Detection

Multiple machine learning models are evaluated:

* Logistic Regression
* Random Forest
* XGBoost

Evaluation focuses on metrics suitable for highly imbalanced financial datasets:

* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC
* Confusion Matrix

---

## 2. Explainable AI

The project integrates:

### SHAP

Used to analyze model-level feature contributions and explain individual predictions.

### LIME

Used to generate local explanations for individual transaction predictions.

Example explanation:

```text
Transaction Risk: High

Important contributing features:
- V14
- V4
- V12
- V10
- V11
```

Feature importance values represent model behavior and should not be interpreted as causal relationships.

---

## 3. Recovery Intelligence

The recovery layer estimates potential recoverable value using:

* Transaction amount
* Model-derived risk
* Explicit recovery assumptions
* Recovery probability
* Business rules

Possible strategies include:

```text
RETRY_PAYMENT
SEND_REMINDER
ALTERNATIVE_PAYMENT
BLOCK_AND_MANUAL_REVIEW
```

### Important

The public datasets used in this project do not contain historical recovery outcomes.

Therefore:

> Recovery values are **framework estimates based on explicit assumptions**, not observed financial recoveries.

---

## 4. Decision Engine

The decision engine converts model outputs into operational actions.

```text
Risk + Recovery Intelligence
             ↓
       Decision Engine
             ↓
 ┌─────────────────────────┐
 │ Retry Payment            │
 │ Send Reminder            │
 │ Alternative Payment      │
 │ Manual Review             │
 │ Block / Review            │
 └─────────────────────────┘
```

The decision engine is a prototype decision-support layer and is not connected to real banking infrastructure.

---

# Datasets

## Credit Card Fraud Detection

Source:

**ULB Credit Card Fraud Detection Dataset**

Main fields include:

```text
Time
V1 ... V28
Amount
Class
```

Where:

```text
Class = 0 → Legitimate
Class = 1 → Fraud
```

The dataset is highly imbalanced, so accuracy is not used as the primary evaluation metric.

---

## PaySim

PaySim is a synthetic mobile-money transaction dataset.

Main fields include:

```text
step
type
amount
nameOrig
oldbalanceOrg
newbalanceOrig
nameDest
oldbalanceDest
newbalanceDest
isFraud
isFlaggedFraud
```

Identifiers such as `nameOrig` and `nameDest` are not used directly as predictive features.

---

# Model Results

The current Credit Card Fraud experiment produced the following test-set results:

| Model               | Precision | Recall |     F1 | ROC-AUC | PR-AUC |
| ------------------- | --------: | -----: | -----: | ------: | -----: |
| Logistic Regression |    0.0550 | 0.8737 | 0.1036 |  0.9684 | 0.6695 |
| Random Forest       |    0.8471 | 0.7579 | 0.8000 |  0.9741 | 0.7903 |
| XGBoost             |    0.9136 | 0.7789 | 0.8409 |  0.9809 | 0.8191 |

These measurements come from the project's held-out test split.

Because the dataset is highly imbalanced, **PR-AUC, precision, recall, and F1** are particularly important when interpreting the results.

---

# Explainability Results

The current SHAP analysis identified the following features among the highest mean absolute SHAP contributions:

```text
V14
V4
V12
V10
V11
V3
V8
V1
V26
V5
```

These values describe how strongly the features contributed to model predictions in the analyzed sample. They do not establish that a feature causes fraud.

---

# System Architecture

Current architecture:

```text
┌───────────────────────┐
│ Financial Transactions│
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│ Data Preprocessing    │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│ Feature Engineering   │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│ Fraud Detection       │
│ LR / RF / XGBoost     │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│ Explainable AI        │
│ SHAP + LIME           │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│ Recovery Model        │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│ Decision Engine       │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│ Streamlit Dashboard   │
└───────────────────────┘
```

---

# Dashboard

The current application provides:

### Dashboard

* Total transactions
* Transaction value
* Average risk
* Expected recovery estimate
* High-priority cases
* Recovery strategy distribution
* Decision distribution

### Model Performance

* Model comparison
* Precision
* Recall
* F1
* ROC-AUC
* PR-AUC
* Confusion matrices

### Risk & Recovery

* Risk categories
* Recovery strategies
* Expected recovery
* High-priority transactions

### Explainability

* SHAP feature importance
* SHAP summary plot
* LIME explanation

### Transaction Analyzer

Interactive transaction-risk analysis using configurable risk and recovery inputs.

---

# Project Structure

```text
Explainable-Financial-AI/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   ├── creditcard.csv
│   │   └── paysim.csv
│   │
│   └── processed/
│       ├── creditcard_clean.csv
│       ├── paysim_clean.csv
│       ├── creditcard_features.csv
│       └── paysim_features.csv
│
├── models/
│
├── notebooks/
│
├── outputs/
│   ├── figures/
│   └── reports/
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── fraud_model.py
│   ├── explainability.py
│   ├── recovery_model.py
│   └── recovery_engine.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

Large datasets, generated reports, model binaries, and the virtual environment are excluded from Git through `.gitignore`.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/sriyathid-commits/Explainable-Financial-AI.git
cd Explainable-Financial-AI
```

## 2. Create a virtual environment

Windows:

```cmd
python -m venv venv
```

Activate:

```cmd
venv\Scripts\activate
```

## 3. Install dependencies

```cmd
pip install -r requirements.txt
```

---

# Dataset Setup

Download the required public datasets separately and place them in:

```text
data/raw/
```

Expected files:

```text
data/raw/creditcard.csv
data/raw/paysim.csv
```

The datasets are intentionally excluded from Git because of their size.

---

# Run the Pipeline

Run preprocessing:

```cmd
python src\data_preprocessing.py
```

Run feature engineering:

```cmd
python src\feature_engineering.py
```

Run exploratory analysis:

```cmd
python src\eda.py
```

Train fraud models:

```cmd
python src\fraud_model.py
```

Generate explanations:

```cmd
python src\explainability.py
```

Generate recovery predictions:

```cmd
python src\recovery_model.py
```

Run the decision engine:

```cmd
python src\recovery_engine.py
```

---

# Run the Dashboard

Start Streamlit:

```cmd
streamlit run app\app.py
```

Open locally:

```text
http://localhost:8501
```

The localhost URL is available only on the machine running the application.

---

# Technology Stack

### Programming

* Python
* SQL-ready architecture

### Machine Learning

* Scikit-learn
* XGBoost
* Imbalanced-learn

### Explainable AI

* SHAP
* LIME

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn
* Plotly

### Application

* Streamlit

### Deployment Architecture — Planned

* FastAPI
* PostgreSQL
* Redis
* WebSockets
* Render
* Vercel

---

# Real-Time UPI Vision

The next stage of the project is to evolve the current batch-processing architecture into a real-time transaction intelligence platform.

Planned architecture:

```text
                 UPI / Payment Event
                         │
                         ▼
                ┌─────────────────┐
                │ API / Webhook   │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Feature Engine  │
                └────────┬────────┘
                         ↓
            ┌────────────┴────────────┐
            ↓                         ↓
      Fraud Model              Anomaly Model
            │                         │
            └────────────┬────────────┘
                         ↓
                  Risk Fusion
                         ↓
                   SHAP / XAI
                         ↓
                  Decision Engine
                         ↓
                Real-Time Dashboard
```

The future system is intended to support:

* Real-time transaction ingestion
* Transaction streaming
* Behavioral anomaly detection
* Velocity analysis
* Risk scoring
* Explainable alerts
* Investigation workflows
* Real-time dashboards
* Authorized payment-provider integrations
* UPI-focused transaction intelligence

Actual banking or payment-network access will require appropriate authorization and supported APIs/sandboxes.

---

# Deployment Roadmap

### Current

```text
Local Python
     ↓
Streamlit
     ↓
localhost:8501
```

### Planned

```text
                    GitHub
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
       Vercel                     Render
    Next.js Frontend             FastAPI
          │                         │
          └───────────┬─────────────┘
                      ↓
               Financial AI Engine
                      │
              ┌───────┴───────┐
              ↓               ↓
          PostgreSQL         Redis
```

---

# Responsible Use

This project is a research and engineering prototype.

It should not be used to make autonomous financial, credit, banking, or payment decisions without appropriate validation, governance, security controls, regulatory compliance, and human oversight.

The current recovery system produces **assumption-based estimates**, not guarantees of financial recovery.

The datasets used for development are public/synthetic sources and do not represent live bank transaction data.

No real customer banking credentials or private financial information should be used in development.

---

# Future Development

* [ ] FastAPI inference service
* [ ] Real-time transaction ingestion
* [ ] WebSocket event streaming
* [ ] PostgreSQL transaction store
* [ ] Redis caching/streaming
* [ ] UPI transaction simulator
* [ ] Behavioral anomaly detection
* [ ] Real-time alert engine
* [ ] Investigation workspace
* [ ] Next.js dashboard
* [ ] Vercel deployment
* [ ] Render backend deployment
* [ ] Model monitoring
* [ ] Data drift monitoring
* [ ] Authentication and role-based access
* [ ] Authorized payment-provider sandbox integration

---

# Project Status

**Current stage:** ML + Explainability + Recovery + Decision Engine + Streamlit Dashboard

**Next stage:** Real-time financial transaction platform

**Long-term direction:** Explainable, real-time financial risk and transaction intelligence infrastructure.

---

## Author

**Sriyathi Dharavath**

B.E. Artificial Intelligence & Data Science
Chaitanya Bharathi Institute of Technology, Hyderabad

GitHub: https://github.com/sriyathid-commits

---

## Disclaimer

This project is intended for educational, research, and software-engineering purposes.

It does not provide banking services, process real financial transactions, access private bank accounts, or guarantee fraud detection or financial recovery outcomes.
