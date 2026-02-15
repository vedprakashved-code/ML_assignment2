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

 1. Logistic Regression — (Accuracy: 0.803, AUC: 0.871)
    - Performs well as a linear baseline model.
    - AUC is strong, indicating good class separation.
    - Precision and recall are balanced, showing stable performance.
    - Slightly limited by its assumption of linear decision boundaries, which may not fully capture the dataset’s non‑linear patterns.

  2. Decision Tree — (Accuracy: 0.803, AUC: 0.801)
    - Accuracy matches Logistic Regression, but AUC is lower, indicating weaker ranking ability.
    - Precision and recall are identical, suggesting symmetric performance across classes.
    - Trees capture non‑linear relationships but may overfit without pruning — the lower AUC hints at this.

  3. K‑Nearest Neighbors (KNN) — (Accuracy: 0.786, AUC: 0.837)
    - Performs reasonably well, especially in recall (0.848).
    - Sensitive to feature scaling — your scaling helped maintain performance.
    - Slightly lower MCC indicates more misclassifications compared to top models.
    - Performance depends heavily on the chosen value of K.

  4. Naive Bayes — (Accuracy: 0.786, AUC: 0.884)
    - AUC is surprisingly strong, showing good probability calibration.
    - Precision is high (0.833), but recall is lower (0.757), meaning it misses some positive cases.
    - Independence assumption limits performance because heart disease features are correlated.

  5. Random Forest (Ensemble) — (Accuracy: 0.754, AUC: 0.858)
    - Performance is lower than expected for an ensemble model.
    - Accuracy and MCC are modest, suggesting the model may be underfitting or not tuned.
    - Still maintains decent recall and F1, showing robustness.
    - Could improve significantly with hyperparameter tuning (n_estimators, max_depth).

  6. XGBoost (Ensemble) — (Accuracy: 0.721, AUC: 0.832)
    - Lowest accuracy among all models in this run.
    - AUC is still respectable, meaning probability estimates are reasonable.
    - Likely underperforming due to default hyperparameters — XGBoost typically needs tuning to shine.
    - Despite being a powerful boosting model, it may be over‑regularized or not optimized for this dataset.

Overall Summary 
- Best overall performers: Logistic Regression and Decision Tree (balanced accuracy and F1).
- Strong AUC performers: Naive Bayes and Logistic Regression.
- KNN: Good recall but slightly unstable overall.
- Random Forest & XGBoost: Surprisingly lower performance, likely due to lack of tuning — both models typically improve significantly with hyperparameter optimization.

Conclusion: Simpler models (Logistic Regression, Decision Tree) performed more consistently on this dataset, while ensemble models require tuning to reach their full potential.
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
