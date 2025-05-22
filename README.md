# 📊 Telecom Customer Churn Prediction and Key Factor Analysis

## 🚀 Project Overview
This project aims to predict customer churn in a telecom company and identify the key factors that significantly influence churn. Predicting and preventing customer churn is crucial for telecom companies as it directly impacts their revenue and customer base.

## 💾 Data Source
The `churn-bigml-80.csv` dataset is sourced from [BigML](https://www.bigml.com/blog/2012/10/17/churn-prediction/) and contains telecom customer data.
- **Number of Samples:** 2666
- **Number of Features:** 20
- **Target Variable:** Churn (Boolean: True/False indicating churn)

## 🛠️ Tech Stack
- Python 3.9+
- Pandas (Data manipulation and analysis)
- NumPy (Numerical operations)
- Scikit-learn (Data preprocessing, model evaluation)
- XGBoost (Gradient Boosting Machine Learning Model)
- Imbalanced-learn (Handling imbalanced datasets - SMOTE)
- Matplotlib, Seaborn (Data visualization)
- Jupyter Notebook (Analysis environment)

## 📈 Key Analysis and Modeling Steps

### 1. Data Exploration and Preprocessing
- **Missing Values:** Confirmed no missing values in the dataset, ensuring data quality.
- **Column Dropping:** The `Phone` column was removed as it's a unique identifier and not relevant for predictive modeling.
- **Categorical Feature Encoding:** Categorical features such as `State`, `Area code`, `International plan`, and `Voice mail plan` were transformed into numerical format using `One-Hot Encoding`. `drop_first=True` was used to prevent multicollinearity.
- **Numerical Feature Scaling:** Numerical features with varying scales (e.g., `Total day minutes`, `Customer service calls`) were standardized using `StandardScaler`. This prevents features with larger scales from dominating the model training.
- **Target Variable Imbalance Handling:** The `Churn` class (churned customers) constituted only approximately 14.6% of the total dataset, indicating a significant class imbalance. To address this, `SMOTE` (Synthetic Minority Over-sampling Technique) was applied to the training data to balance the class distribution.

### 2. Model Selection and Training (XGBoost)
- **Model Choice:** `XGBoost` was selected as the predictive model due to its robust performance in classification tasks and its capabilities in handling imbalanced datasets.
- **Hyperparameter Tuning:** `GridSearchCV` was employed to systematically find the optimal combination of hyperparameters for the XGBoost model, including `n_estimators`, `learning_rate`, and `max_depth`. The model was optimized based on the `ROC-AUC` score.

### 3. Model Evaluation
The optimized XGBoost model was evaluated on the unseen test dataset using various metrics suitable for imbalanced classification problems:
- **Accuracy:** X.XX%
- **Precision:** X.XX
- **Recall:** X.XX (Particular emphasis was placed on Recall, as minimizing false negatives - missing actual churners - is crucial for churn prevention.)
- **F
