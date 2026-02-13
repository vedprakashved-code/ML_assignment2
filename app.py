#----------------------------
# ML Assignment 2 - app.py
#----------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Classifier",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction App")
st.write("Upload test data and evaluate the trained ML model.")

# -----------------------------
# Load Saved Model + Scaler
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# -----------------------------
# File Upload
# -----------------------------
st.header("📤 Upload Test CSV File")

uploaded_file = st.file_uploader("Upload a CSV file with the same columns as training data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # -----------------------------
    # Preprocess
    # -----------------------------
    X = scaler.transform(df)

    # -----------------------------
    # Predictions
    # -----------------------------
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    df["Prediction"] = preds
    df["Probability"] = probs

    st.subheader("🔍 Predictions")
    st.dataframe(df)

    # -----------------------------
    # Metrics (Optional: If user uploads true labels)
    # -----------------------------
    st.header("📊 Evaluation Metrics (Optional)")

    if "target" in df.columns:
        y_true = df["target"]
        y_pred = preds

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, probs)
        }

        st.write(metrics)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    else:
        st.info("To compute metrics, include a 'target' column in your CSV.")

else:
    st.warning("Please upload a CSV file to proceed.")