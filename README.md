# 📊 Telecom Customer Churn Prediction and Key Factor Analysis

## 🚀 Project Overview
This project aims to predict customer churn in a telecom company and identify the key factors that significantly influence churn. Predicting and preventing customer churn is crucial for telecom companies as it directly impacts their revenue and customer base.

## 💾 Data Source
The `churn-bigml-80.csv` dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets?select=churn-bigml-80.csv) and contains telecom customer data.
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
- **Accuracy:** 0.94
- **Precision:** 0.9
- **Recall:** 0.69 (Particular emphasis was placed on Recall, as minimizing false negatives - missing actual churners - is crucial for churn prevention.)
- **F-Score** 0.78

- **Confusion Matrix**
- ![image](https://github.com/user-attachments/assets/3db4a7a7-a03f-41e9-b3fd-bed8c4c78833)
- 
- **Actual Churn** 78
- **Predicted Churn** 60
- **Correctly Predictied Churn** 54

- ### 4. Model Interpretation: Identifying Key Churn Factors
The `Feature Importance` from the trained `XGBoost` model was analyzed to identify the most influential factors contributing to customer churn.
- **`Total day charge`**: Customers with higher daily call charges showed a notably higher churn rate. This suggests that dissatisfaction with call costs could be a primary reason for churn.
- **`Customer service calls`**: An increasing number of calls to customer service correlated with a higher likelihood of churn. Frequent calls may indicate unresolved issues or growing dissatisfaction with the service.
- **`International plan`**: Customers subscribed to an international plan exhibited a higher churn rate. This might suggest issues related to the quality or pricing of international calling services.
- ... (Add other significant features you found and provide business insights for each)

### 🚀 Conclusion and Business Implications
This project successfully built an `XGBoost` model capable of effectively predicting telecom customer churn and identified the core factors influencing it. The model can empower telecom companies to proactively identify high-risk customers and implement targeted retention strategies, such as addressing high `Total day charge` concerns, improving resolution for frequent `Customer service calls`, or offering special incentives to `International plan` users. Ultimately, this leads to improved customer retention and increased revenue.

### 💡 Future Enhancements
- **Data Enrichment:** Integrate more diverse customer behavior data (e.g., website visit logs, app usage patterns) to further enhance predictive accuracy.
- **Real-time Prediction System:** Deploy the prediction model as an API to enable real-time monitoring of churn risk and immediate intervention strategies.
- **Exploring Other Models/Ensembles:** Investigate other advanced boosting models like LightGBM, CatBoost, or ensemble techniques such as Stacking to potentially achieve further performance improvements.

## 🏃‍♀️ How to Run
1.  Clone this repository: `git clone https://github.com/nhs102/telecom-churn-prediction.git`
2.  Navigate to the project directory: `cd telecom-churn-prediction`
3.  Install the required libraries: `pip install -r requirements.txt`
4.  Launch Jupyter Notebook and open the `telecom_churn_prediction.ipynb` file: `jupyter notebook`

## 📧 Contact
- LinkedIn: [Click Here](https://www.linkedin.com/in/shawn-nam-b79614204/)
- Email: tjr001136@gmail.com


