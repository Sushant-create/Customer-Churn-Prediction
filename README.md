# Customer Churn Prediction using Logistic Regression

## Objective
A telecommunications company wants to identify customers who are likely to **churn** (leave the service) based on their demographic information and service usage patterns. This project builds and evaluates a **Logistic Regression** model to predict customer churn, so the company can proactively target at-risk customers with retention offers.

## Dataset
**Telco Customer Churn Dataset** (7,043 customers, 21 columns)
Kaggle Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

- **Target variable:** `Churn` (Yes / No)
- **Numerical features:** `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`
- **Categorical features:** `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`
- `customerID` is an identifier column and was dropped before modeling.

## Libraries Used
- `pandas`, `numpy` — data loading and manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — preprocessing (`LabelEncoder`, `StandardScaler`, `train_test_split`), modeling (`LogisticRegression`), and evaluation metrics (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report`)

## Methodology
1. **Data Understanding** — Loaded the dataset, inspected the first five records, and identified numerical, categorical, and target features.
2. **Data Preprocessing**
   - Converted `TotalCharges` to numeric (it was stored as text and had 11 blank entries for brand-new customers).
   - Imputed the 11 missing `TotalCharges` values with the column median.
   - Dropped the non-predictive `customerID` column.
   - Encoded the binary target `Churn` (Yes → 1, No → 0) and label-encoded all remaining categorical columns.
   - Split the data into 80% training / 20% testing (stratified on `Churn` to preserve class balance), then standardized features with `StandardScaler`.
3. **Model Development** — Trained a `LogisticRegression` model (max_iter=1000) on the scaled training data and generated predictions on the held-out test set.
4. **Model Evaluation** — Assessed performance with accuracy, precision, recall, F1-score, and a confusion matrix; also examined model coefficients to identify the strongest churn drivers.

## Results

| Metric      | Score  |
|-------------|--------|
| Accuracy    | 0.799  |
| Precision   | 0.643  |
| Recall      | 0.548  |
| F1-Score    | 0.592  |

**Confusion Matrix**

|                     | Predicted: No Churn | Predicted: Churn |
|---------------------|:--------------------:|:-----------------:|
| **Actual: No Churn** | 921                  | 114               |
| **Actual: Churn**    | 169                  | 205               |

**Observations:**
1. The model reaches **~80% overall accuracy**, but recall for churners (0.55) is much lower than recall for non-churners (0.89) — a typical effect of the moderate class imbalance in the dataset (~73% No-Churn vs ~27% Churn).
2. **Precision (0.64)** means roughly two out of three customers flagged as "will churn" actually churn; the **F1-score (0.59)** captures the trade-off between catching churners and avoiding false positives.
3. Based on model coefficients, **short tenure, month-to-month contracts, and high monthly charges** are the strongest drivers pushing predictions toward churn, while **long tenure, one/two-year contracts, and add-on services (OnlineSecurity, TechSupport)** are the strongest drivers of retention.

## Conclusion
This project used Logistic Regression to predict telecom customer churn from demographic and service-usage data, achieving an overall **accuracy of ~80%**, with a **precision of 0.64**, **recall of 0.55**, and **F1-score of 0.59** for the churn class. The key factors driving churn were **short customer tenure, month-to-month contracts, and higher monthly charges**, while **longer contract commitments, longer tenure, and subscription to services like online security and tech support** were associated with retention. These findings suggest the company should focus retention efforts on new, month-to-month customers and consider incentivizing longer contracts or bundled security/support services. A key **limitation of Logistic Regression** is that it assumes a roughly linear relationship between the features and the log-odds of churn, so it may struggle to capture more complex, non-linear interactions between variables (e.g., how contract type and tenure jointly affect churn risk) — models like Random Forest or Gradient Boosting could potentially capture such patterns better, at some cost to interpretability.

## Files in this Repository
- `Customer_Churn_Prediction.ipynb` — Full notebook with code, outputs, and visualizations for Tasks 1–5
- `Telco-Customer-Churn.csv` — Dataset used (mirrors the Kaggle dataset above)
- `confusion_matrix.png` — Confusion matrix visualization
- `feature_coefficients.png` — Logistic Regression coefficient plot showing churn drivers
- `README.md` — This file
