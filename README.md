# ML_assignment2
ML_assignment2 for 6 models and application
# ❤️ ML Assignment 2 — Heart Disease Classification (BITS WILP)

This project implements an end‑to‑end Machine Learning workflow using the **Heart Disease UCI Dataset**.  
It includes model training, evaluation using six metrics, and deployment as an interactive **Streamlit web app**.

---

## 1. 📌 Problem Statement
The objective is to build and compare multiple machine learning models to predict the presence of heart disease based on clinical attributes.  
The project also includes deploying the best-performing model as a Streamlit application.

---

## 2. 📊 Dataset Description
**Dataset:** Heart Disease UCI (Kaggle version)  
**Rows:** ~918  
**Features:** 12+ clinical attributes  
**Target:**  
- `0` → No heart disease  
- `1` → Heart disease present  

### Key Features
- Age  
- Sex  
- Chest pain type  
- Resting blood pressure  
- Cholesterol  
- Fasting blood sugar  
- Resting ECG  
- Max heart rate  
- Exercise-induced angina  
- Oldpeak  
- Slope  
- Major vessels  
- Thalassemia  

---

## 3. 🤖 Models Implemented
The following **six classification models** were trained and evaluated:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors  
4. Gaussian Naive Bayes  
5. Random Forest Classifier  
6. XGBoost Classifier  

---

## 4. 📈 Evaluation Metrics
Each model was evaluated using the six required metrics:

- **Accuracy**  
- **AUC (ROC Area Under Curve)**  
- **Precision**  
- **Recall**  
- **F1 Score**  
- **MCC (Matthews Correlation Coefficient)**  

---

## 5. 📋 Model Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.809756 | 0.929810 | 0.761905 | 0.914286 | 0.831169 | 0.630908 |
| Decision Tree | 0.985366 | 0.985714 | 1.000000 | 0.971429 | 0.985507 | 0.971151 |
| KNN | 0.863415 | 0.962905 | 0.873786 | 0.857143 | 0.865385 | 0.726935 |
| Naive Bayes | 0.829268 | 0.904286 | 0.807018 | 0.876190 | 0.840183 | 0.660163 |
| Random Forest | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| XGBoost | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

Logistic Regression	|	| 0.803279	| 0.871212 | 0.800000 |	0.848485 |	0.823529 |	0.603071 |
> *(Replace “…” with values from your notebook output.)*

---

## 6. 📝 Observations
- Tree‑based models (Random Forest, XGBoost) generally perform better due to their ability to capture non‑linear relationships.  
- Logistic Regression provides a strong baseline with interpretable coefficients.  
- Naive Bayes performs well when feature independence assumptions hold.  
- KNN performance depends heavily on scaling and neighborhood size.  
- XGBoost often achieves the best AUC due to gradient boosting optimization.

---

## 7. 🌐 Streamlit App

### 🔗 **Live App Link**  
*(Add your Streamlit Cloud link here)*

### Features
- Upload CSV test data  
- Automatic scaling using saved `scaler.pkl`  
- Predictions + probabilities  
- Optional evaluation metrics (if `target` column exists)  
- Confusion matrix visualization  

---

## 8. 🛠 How to Run Locally

### **1. Clone the repository**
```bash
git clone <your-repo-url>
cd ml-assignment-2
