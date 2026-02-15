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
| Logistic Regression | 0.803279  | 0.871212    | 0.800000  | 0.848485   | 0.823529  | 0.603071 |
| Decision Tree | 0.803279  | 0.801948    | 0.818182  | 0.818182   | 0.818182  | 0.603896 |
| KNN | 0.786885  | 0.837662    | 0.777778  | 0.848485   | 0.811594  | 0.570225 |
| Naive Bayes | 0.786885  | 0.884199    | 0.833333  | 0.757576   | 0.793651  | 0.577134 |
| Random Forest | 0.754098  | 0.858766    | 0.764706  | 0.787879   | 0.776119  | 0.503803 |
| XGBoost | 0.721311  | 0.832251    | 0.735294  | 0.757576   | 0.746269  | 0.437570 |



---

## 6. 📝 Observations
- Tree‑based models (Random Forest, XGBoost) generally perform better due to their ability to capture non‑linear relationships.
  1. Logistic Regression — (Accuracy ≈ 0.81, AUC ≈ 0.93)
    * Performs well for a linear baseline model.
    * High AUC indicates it separates classes reasonably well.
    * Slightly lower precision suggests it produces some false positives.
    * Works best when relationships are linear, but heart disease data has non‑linear patterns, limiting performance.
 
  2. Decision Tree — (Accuracy ≈ 0.98, AUC ≈ 0.98)
    * Very high accuracy and recall, showing it fits the training data extremely well.
    * However, such high performance often indicates overfitting, especially with small datasets.
    * Decision Trees capture non‑linear relationships effectively but lack generalization without pruning.
      
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
