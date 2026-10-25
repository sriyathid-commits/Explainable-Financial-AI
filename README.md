# Explainable Financial AI

### Explainable AI for Fraud Detection, Financial Risk Analysis & Revenue Recovery

An end-to-end machine learning platform combining fraud detection, explainable AI, financial risk analysis, recovery intelligence, and automated decision support.

## Overview

Explainable Financial AI detects suspicious transactions, estimates financial risk, explains model predictions, and generates recovery-oriented recommendations using public financial datasets.

### Key Features

- **Fraud Detection:** Logistic Regression, Random Forest, and XGBoost.
- **Explainable AI:** SHAP feature importance and LIME local explanations.
- **Recovery Intelligence:** Assumption-based recovery estimates and strategy recommendations.
- **Decision Engine:** Prototype recommendations for retries, reminders, alternative payments, and manual review.
- **Interactive Dashboard:** Streamlit-based analytics and transaction-risk analysis.
- **Deep Learning:** LSTM fraud-risk modeling and autoencoder-based anomaly detection.

## System Architecture

```text
Financial Transactions
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Fraud Detection / Anomaly Detection
        ↓
Explainable AI
        ↓
Recovery Intelligence
        ↓
Decision Engine
        ↓
Streamlit Dashboard
```

## Datasets

### Credit Card Fraud Detection

The ULB credit-card dataset contains anonymized PCA features (`V1`–`V28`), transaction time, amount, and a fraud label (`Class`).

- `Class = 0`: Legitimate transaction
- `Class = 1`: Fraudulent transaction

### PaySim

PaySim is a synthetic mobile-money transaction dataset. It includes transaction types, amounts, balance changes, and fraud labels.

The datasets must be downloaded separately and placed in `data/raw/`.

```text
data/raw/creditcard.csv
data/raw/paysim.csv
```

Large datasets are excluded from Git.

## Traditional Machine Learning Results

Recorded results from the project's held-out credit-card test split:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0550 | 0.8737 | 0.1036 | 0.9684 | 0.6695 |
| Random Forest | 0.8471 | 0.7579 | 0.8000 | 0.9741 | 0.7903 |
| XGBoost | 0.9136 | 0.7789 | 0.8409 | 0.9809 | 0.8191 |

These are recorded experiment results, not a guarantee of future performance. Precision, recall, F1, and PR-AUC are important because fraud datasets are highly imbalanced.

## Explainable AI

SHAP analysis identified the following features among the highest mean absolute contributions in the analyzed sample:

`V14`, `V4`, `V12`, `V10`, `V11`, `V3`, `V8`, `V1`, `V26`, `V5`.

These anonymized features describe model behavior; they do not establish causation.

LIME provides local explanations for individual predictions.

## Deep Learning

The deep-learning prototype includes:

- An LSTM model for sequence-based fraud-risk prediction.
- An autoencoder for anomaly detection using reconstruction error.
- A scaler and feature-column metadata for preprocessing.
- Saved evaluation results and an autoencoder threshold derived from the 95th percentile of normal reconstruction errors.

The training script reads `data/raw/creditcard.csv` and saves artifacts under `models/deep_learning/`.

Run training:

```cmd
python src\train_deep_learning.py
```

The autoencoder is an anomaly-detection signal, not a standalone reliable fraud classifier. Deep-learning results should be evaluated alongside the traditional models.

## Recovery Intelligence and Decision Engine

The recovery layer estimates potential recoverable value using transaction amounts, model-derived risk, recovery probabilities, and explicit assumptions.

Possible strategies include:

- `RETRY_PAYMENT`
- `SEND_REMINDER`
- `ALTERNATIVE_PAYMENT`
- `BLOCK_AND_MANUAL_REVIEW`

The public datasets do not contain observed recovery outcomes. Recovery values are estimates, not confirmed financial recoveries. The decision engine is a prototype and is not connected to live banking infrastructure.

## Project Structure

```text
Explainable-Financial-AI/
├── app/
│   ├── app.py
│   └── pages/
│       └── 2_Deep_Learning.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── deep_learning/
├── notebooks/
├── outputs/
│   ├── figures/
│   └── reports/
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── fraud_model.py
│   ├── explainability.py
│   ├── recovery_model.py
│   ├── recovery_engine.py
│   └── train_deep_learning.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```cmd
git clone https://github.com/sriyathid-commits/Explainable-Financial-AI.git
cd Explainable-Financial-AI
```

Create and activate a virtual environment on Windows:

```cmd
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```cmd
pip install -r requirements.txt
```

Download the required datasets separately and place them in `data/raw/`.

## Run the Pipeline

Run the following commands from the project root, as needed:

```cmd
python src\data_preprocessing.py
python src\feature_engineering.py
python src\eda.py
python src\fraud_model.py
python src\explainability.py
python src\recovery_model.py
python src\recovery_engine.py
```

Train the deep-learning models separately:

```cmd
python src\train_deep_learning.py
```

## Run the Dashboard

```cmd
streamlit run app\app.py
```

Open `http://localhost:8501` in your browser.

The dashboard includes transaction analytics, model evaluation, risk and recovery information, explainability, transaction analysis, and a deep-learning page.

## Technology Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn, XGBoost, Imbalanced-learn
- **Deep learning:** TensorFlow / Keras
- **Explainability:** SHAP, LIME
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Dashboard:** Streamlit

## Future Development

- [ ] FastAPI inference service
- [ ] Real-time transaction-event ingestion
- [ ] WebSocket streaming
- [ ] PostgreSQL transaction storage
- [ ] Redis caching or streaming
- [ ] UPI transaction simulator
- [ ] Behavioral anomaly detection
- [ ] Real-time alert and investigation workflows
- [ ] Model and data-drift monitoring
- [ ] Authentication and role-based access
- [ ] Authorized payment-provider sandbox integration

These are planned enhancements, not claims of currently implemented live integrations.

## Responsible Use

This project is an educational and research prototype. It is not connected to live banking or UPI systems and does not process real payments. It should not make autonomous financial decisions without suitable validation, security, governance, regulatory compliance, and human oversight.

## Author

**Sriyathi Dharavath**  
B.E. Artificial Intelligence & Data Science  
Chaitanya Bharathi Institute of Technology, Hyderabad

GitHub: https://github.com/sriyathid-commits

---

*For educational, research, and software-engineering purposes only. Fraud predictions and recovery estimates are not guarantees.*
