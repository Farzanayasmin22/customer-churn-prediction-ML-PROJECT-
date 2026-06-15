
# 🔄 Customer Churn Prediction
Customer Churn Prediction using Ensemble Models, PCA and Neural Network

## 📌 Project Overview
This project was designed to apply advanced machine learning algorithms 
hands-on and identify the best model for predicting customer churn.
The goal is to predict which customers are likely to leave a telecom 
company before they actually leave — so the business can take 
preventive action like offering discounts or better plans.

---

## 📂 Dataset
- **Source:** Telco Customer Churn Dataset — Kaggle
- **Total Records:** 7,043 customers
- **Target Column:** Churn (Yes = Left, No = Stayed)
- **Features:** 21 columns including customer demographics, 
  services used, and payment details

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Matplotlib

---

## 🔬 Project Pipeline

### Step 1 — Data Loading & Exploration
- Loaded the dataset and explored shape, columns and missing values
- Visualized churn distribution — found data is imbalanced
  (5000 stayed vs 1900 left)

### Step 2 — Data Preprocessing
- Dropped customerID column (no prediction value)
- Fixed TotalCharges column (stored as text, converted to numeric)
- Encoded Churn column — Yes → 1, No → 0
- Applied get_dummies to convert all text columns to numbers
- Applied StandardScaler to scale all numeric features

### Step 3 — Dimensionality Reduction (PCA)
- Applied PCA to reduce features from 30 → 17
- Retained 95% of the original information
- Plotted cumulative explained variance graph

### Step 4 — Ensemble ML Models
- Trained Random Forest, AdaBoost and XGBoost
- Compared accuracy scores using bar chart

### Step 5 — Neural Network (TensorFlow)
- Built a Sequential Neural Network with:
  - Two hidden layers (64 and 32 neurons) with ReLU activation
  - Dropout layers (30%) to prevent overfitting
  - Output layer with Sigmoid activation
- Trained for 30 epochs
- Plotted training vs validation accuracy graph

### Step 6 — Model Evaluation
- Classification Report (Precision, Recall, F1 Score)
- ROC AUC Score for all models
- Cross Validation (5 folds) for reliable accuracy

### Step 7 — Final Comparison
- Compared all models side by side
- Identified best model based on ROC AUC Score

---

## 📊 Results

| Model | Accuracy | CV Accuracy | ROC AUC |
|-------|----------|-------------|---------|
| Random Forest | 77.65% | 77.5% | 0.790 |
| AdaBoost | 78.04% | 78.88% | 0.810 |
| XGBoost | 76.73% | 77.08% | 0.790 |
| **Neural Network** | **78.27%** | — | **0.822** |

---

## ✅ Conclusion
All four models performed consistently around 78% accuracy.
Based on the **ROC AUC Score**, the **Neural Network was the 
best model from this analysis** with a score of **0.822** — 
meaning it correctly distinguishes a churned customer from a 
stayed customer 82% of the time.

With further hyperparameter tuning, especially on XGBoost, 
the performance could potentially be improved even further.

---


**Farzana**
- GitHub: [@Farzanayasmin22](https://github.com/Farzanayasmin22)
