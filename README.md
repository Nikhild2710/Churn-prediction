# Telco Customer Churn Prediction  

This project predicts **customer churn** using the [Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). The dataset contains 7,043 customers with 21 features describing demographics, services, account details, and churn status.  

---

## 📂 Dataset Overview  
- **Rows:** ~7,043  
- **Target:** `Churn` (Yes = customer left, No = customer stayed)  
- **Feature groups:**
  - **Demographic:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`  
  - **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`  
  - **Account info:** `Contract`, `PaperlessBilling`, `PaymentMethod`  
  - **Numeric features:** `tenure`, `MonthlyCharges`, `TotalCharges`  

---

## 🛠️ Steps I Followed  

### 1. Data Loading  
- Pulled the dataset directly from **Kaggle API** into Google Colab.  
- Stored it under a Drive directory (`/content/drive/MyDrive/churn-predictor`) for persistence.  

### 2. Data Cleaning  
- Converted `TotalCharges` → numeric (coerced errors to NaN).  
- Imputed missing values in `TotalCharges` with the **median**.  
- Dropped `customerID` (identifier, not predictive).  
- Converted `Churn` into binary (Yes=1, No=0).  

### 3. Exploratory Analysis  
- Checked **class imbalance** → ~26.5% churn rate.  
- Correlation analysis:  
  - `tenure` strongly **negatively** correlated with churn (-0.35).  
  - `MonthlyCharges` and `SeniorCitizen` weakly **positively** correlated with churn.  
  - `TotalCharges` **negatively** correlated with churn.  
- Insight: **short tenure + high charges** → highest churn risk.  

### 4. Preprocessing Pipeline  
- Used **`ColumnTransformer`**:  
  - Numeric features → `StandardScaler`.  
  - Categorical features → `OneHotEncoder`.  
- Wrapped transformations + model in a **single Pipeline**.  

### 5. Baseline Model (XGBoost)  
- Trained **XGBoostClassifier** with class imbalance handling (`scale_pos_weight ≈ 3`).  
- Performance:  
  - **Accuracy:** ~75%  
  - **Precision (churners):** ~0.52  
  - **Recall (churners):** ~0.76  
  - **F1 (churners):** ~0.62  



### 7. Model Explainability (SHAP)  
- **Feature Importance (global):**  
  - Top churn drivers: short tenure, month-to-month contract, high monthly charges, no online security, no tech support, fiber optic service.  
- **SHAP Summary Plot:**  
  - Red = higher feature value, Blue = lower.  
  - Right side → increases churn probability, Left side → decreases churn probability.  
- **SHAP Force Plot (local):**  
  - For each customer, shows baseline churn probability, and which features push the prediction up/down.

 <img width="793" height="940" alt="image" src="https://github.com/user-attachments/assets/f2a14b44-fe42-458f-8155-691de36d3b06" />

<img width="1570" height="362" alt="image" src="https://github.com/user-attachments/assets/ae37a15b-3e63-4e41-afd7-575898f5d937" />


---

## 📈 Key Findings  
- Customers with **short tenure**, **month-to-month contracts**, **fiber optic service**, and **no add-on support/security services** are most likely to churn.  
- Customers with **long tenure**, **two-year contracts**, and **higher total charges (loyal long-term spenders)** are least likely to churn.  

---

  

---

## 📜 References  
- Dataset: [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- SHAP documentation: [https://shap.readthedocs.io](https://shap.readthedocs.io)  
