# 📉 Customer Churn Prediction & Retention Intelligence

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange?logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-2.3-purple?logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)

**DEPI Graduation Project — Group 2**  
*Predict telecom customer churn before customers leave — clean, notebook-first, and ready for presentation.*

</div>

---

## 🧩 Project Overview

This project builds an end-to-end **Machine Learning pipeline** to predict whether a telecom customer will **Churn** (`1`) or **Stay** (`0`) based on their demographics, subscribed services, contract type, and billing history.

All project stages are organized into two clear, self-contained **Jupyter Notebooks** for easy walkthrough and presentation:
1. **`EDA_Customer_Churn.ipynb`** — Data understanding, quality checks, cleaning, visualizations, and insights.
2. **`model_building.ipynb`** — Leakage-safe preprocessing, feature engineering, 5-fold CV hyperparameter tuning (**KNN** & **SVM**), evaluation, and saving the final `.pkl` models.

---

## 🎯 Business Problem & Expected Impact

Acquiring a new telecom customer costs significantly more than retaining an existing one. By identifying high-risk customers early, the retention team can:
- 🎁 Offer targeted contract upgrades or loyalty discounts
- 📞 Proactively reach out to high-risk month-to-month / fiber-optic users
- 📊 Focus retention budget on customers with the highest churn risk

---

## 📁 Repository Structure

```
customer-churn-retention/
├── .gitignore
├── README.md
├── requirements.txt
├── Telco_Customer_Churn_Cleaned.csv   # Raw input dataset (7,043 rows, 21 columns)
├── cleaned_data.csv                   # Cleaned dataset produced by EDA notebook
├── test_data.csv                      # Isolated 20% held-out test split (1,409 rows)
├── EDA_Customer_Churn.ipynb           # Full EDA & data cleaning notebook
├── model_building.ipynb               # Preprocessing, KNN/SVM tuning, evaluation & export
├── customer_churn_model.pkl           # Recommended end-to-end SVM pipeline (raw → prediction)
├── customer_churn_knn_model.pkl       # Tuned end-to-end KNN pipeline
├── customer_churn_preprocessor.pkl    # Fitted FeatureEngineer + ColumnTransformer
└── report.html                        # Executive summary report (open in browser)
```

---

## ⚙️ Tech Stack

- **Data Analysis & Visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Preprocessing & Modeling:** `scikit-learn` (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `KNeighborsClassifier`, `SVC`, `GridSearchCV`)
- **Serialization:** `joblib` (`.pkl`)
- **Notebooks:** `Jupyter`

---

## 🔄 End-to-End Process

```
Telco_Customer_Churn_Cleaned.csv (7,043 × 21)
   │
   ▼  [EDA_Customer_Churn.ipynb]
1. Data Cleaning           → Drop customerID, impute 11 blank TotalCharges (tenure=0 → 0.0)
   │                         Export cleaned_data.csv
   ▼  [model_building.ipynb]
2. Stratified 80/20 Split  → 5,634 train / 1,409 held-out test (saved separately to test_data.csv)
   │
3. Feature Engineering     → +6 domain features (AvgMonthlySpend, PriceHikeRatio,
   │                         TotalServicesCount, IsAutoPayment, HasHighRiskContract, TenureCohort)
   │
4. Preprocessing           → StandardScaler (6 numerical) + OneHotEncoder (16 categorical) + 3 binary = 54 features
   │
5. 5-Fold Stratified CV    → KNN (20 candidates) & SVM (24 candidates), selected by mean CV F1
   │
6. Held-Out Evaluation     → Evaluated once on test_data.csv (no data leakage)
   │
7. Model Export            → customer_churn_model.pkl, customer_churn_knn_model.pkl, customer_churn_preprocessor.pkl
```

---

## 📊 Data — Before & After EDA

| Stage | Shape | Key Changes |
|---|---|---|
| **Before EDA** (`Telco_Customer_Churn_Cleaned.csv`) | `7,043 × 21` | Contains `customerID`; `TotalCharges` has 11 whitespace entries (`tenure == 0`) |
| **After EDA** (`cleaned_data.csv`) | `7,043 × 20` | `customerID` dropped; `TotalCharges` numeric (`0.0` for `tenure == 0`); 0 nulls, 0 duplicates |
| **After Preprocessing** (`model_building.ipynb`) | `5,634 × 54` (train) / `1,409 × 54` (test) | +6 engineered features, scaled numericals, one-hot encoded categories |

**Key EDA Findings:**
- **Class Imbalance:** 26.5% Churn (`1,869`) vs 73.5% Stay (`5,174`)
- **Contract Type:** Month-to-month contracts churn at **~42.7%** vs **~2.8%** for 2-year contracts
- **Tenure:** New customers (`0–12m`) have the highest churn rate
- **Payment Method:** Electronic check users churn at **~45.3%** vs ~15–17% for automatic payments

---

## 🤖 Expected & Achieved Results (Held-Out Test Set: 1,409 Customers)

| Metric | KNN (`k=41, uniform, p=1`) | SVM (`RBF, C=1, balanced`) |
|---|---:|---:|
| Accuracy | 0.7906 | 0.7502 |
| Precision | 0.6106 | 0.5189 |
| **Recall (Churn)** | 0.5829 | **0.8075** |
| **F1 Score** | 0.5964 | **0.6318** |
| **ROC-AUC** | **0.8391** | 0.8277 |

> **Recommendation:** **SVM** (`customer_churn_model.pkl`) is recommended when catching potential churners (**80.75% Recall**) is the top priority.

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/0Seif0/customer-churn-retention.git
cd customer-churn-retention

# 2. Create & activate virtual environment (Python 3.10 or 3.11)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open notebooks
jupyter notebook
```

### Quick Inference in Python
```python
import joblib
import pandas as pd

model = joblib.load("customer_churn_model.pkl")
test_df = pd.read_csv("test_data.csv").drop(columns=["Churn"])
predictions = model.predict(test_df.head(5))
print(predictions)  # 1 = Likely Churn, 0 = Likely Stay
```
