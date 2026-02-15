#----------------------------
# ML Assignment 2 - app.py
#----------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Heart Disease ML App", layout="wide")
st.title("❤️ Heart Disease Classification — ML Assignment 2")
st.write("Upload a dataset, choose a model, and view predictions + evaluation metrics.")

# ---------------------------------------------------------
# Load Saved Models
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
        "Decision Tree": joblib.load("models/decision_tree.pkl"),
        "KNN": joblib.load("models/knn.pkl"),
        "Naive Bayes": joblib.load("models/naive_bayes.pkl"),
        "Random Forest": joblib.load("models/random_forest.pkl"),
        "XGBoost": joblib.load("models/xgboost.pkl")
    }
    scaler = joblib.load("models/scaler.pkl")
    return models, scaler

models, scaler = load_models()

# ---------------------------------------------------------
# Sidebar — Model Selection
# ---------------------------------------------------------
st.sidebar.header("⚙️ Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a model for prediction:",
    list(models.keys())
)

model = models[selected_model_name]

# ---------------------------------------------------------
# Dataset Upload
# ---------------------------------------------------------
st.header("📤 Upload Dataset (CSV)")
uploaded_file = st.file_uploader("Upload a CSV file with the same columns as training data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔎 Preview of Uploaded Data")
    st.dataframe(df.head())

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------
    X = df.drop(columns=["target"], errors="ignore")
    X_scaled = scaler.transform(X)

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]

    df["Prediction"] = preds
    df["Probability"] = probs

    st.subheader("🔍 Predictions")
    st.dataframe(df)

    # ---------------------------------------------------------
    # Evaluation Metrics (if target exists)
    # ---------------------------------------------------------
    st.header("📊 Evaluation Metrics")

    if "target" in df.columns:
        y_true = df["target"]
        y_pred = preds

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, probs),
            "MCC": matthews_corrcoef(y_true, y_pred)
        }

        st.write(metrics)

        # ---------------------------------------------------------
        # Confusion Matrix
        # ---------------------------------------------------------
        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    else:
        st.info("To compute evaluation metrics, include a 'target' column in your CSV.")

else:
    st.warning("Please upload a CSV file to continue.")

