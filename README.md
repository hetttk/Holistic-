# Holistic Data Preparer
# Credit Risk Data Preprocessing & Feature Engineering

## Project Overview

This project focuses on preparing a customer credit risk dataset for machine learning. The objective is to perform data cleaning, missing value handling, outlier detection, feature engineering, encoding, scaling, and transformation to create a clean and ML-ready dataset for loan default prediction.

The project simulates a real-world fintech use case where customer information is collected from multiple sources and must be processed before model development.

---

## Problem Statement

The goal is to predict whether a customer is likely to default on a loan using demographic, financial, and behavioral attributes.

### Target Variable

- `default_flag`
  - 0 = No Default
  - 1 = Default

---

## Dataset Features

### Demographic Features

- customer_id
- age
- gender
- region
- education_level
- employment_type

### Financial Features

- annual_income
- loan_amount
- credit_score

### Behavioral Features

- repayment_history
- transaction_count
- spending_ratio

### Date Feature

- join_date

### Target Feature

- default_flag

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- SQLite

---

# Project Workflow

## 1. Data Acquisition

Data was collected and integrated from multiple sources:

<img width="1106" height="607" alt="image" src="https://github.com/user-attachments/assets/d42a1cd8-2634-4da0-a667-37a6afed6907" />

- CSV File
- JSON Data
- SQL Database
- Dummy API

The datasets were merged into a unified customer credit risk dataset.

---

## 2. Data Understanding

Dataset exploration was performed using:

<img width="1195" height="572" alt="image" src="https://github.com/user-attachments/assets/ebef16bc-8e5d-4a1a-b2f1-4ff728cadbf0" />

<img width="438" height="510" alt="image" src="https://github.com/user-attachments/assets/456a228d-1722-43e1-8012-c4296c125a5b" />

```python
df.head()
df.tail()
df.info()
df.describe()
df.shape
```

Key activities:

- Data type inspection
- Missing value analysis
- Statistical summary
- Dataset profiling

---

## 3. Missing Value Handling

<img width="1030" height="652" alt="image" src="https://github.com/user-attachments/assets/f61cb9e3-c6a9-4807-92d5-59c65a4ab87b" />

Missing values were identified in:

- age
- annual_income
- credit_score
- gender
- employment_type

### Techniques Used

#### Numerical Features

- Median Imputation

#### Categorical Features

- Most Frequent Imputation

Result:

- All missing values were successfully handled.

---

## 4. Outlier Detection

Outliers were detected in:

- annual_income
<img width="799" height="470" alt="e82bfaad-dbd8-4cc0-ba58-8726fe31cd4c" src="https://github.com/user-attachments/assets/8ea1bc20-3e71-448e-b63e-a2ad36660cbc" />
<img width="520" height="455" alt="56008242-a7e3-4fba-8cc8-9cc760c8493c" src="https://github.com/user-attachments/assets/6dc02111-3780-4bb2-a1ee-66d38858689d" />



### Method Used

#### Z-Score Method
<img width="455" height="247" alt="image" src="https://github.com/user-attachments/assets/c04750a9-d369-4f5c-a561-68017cb49580" />

Outliers were identified using standard deviation from the mean.

Activities performed:

- Z-Score calculation
- Outlier visualization
- Distribution analysis
- Boxplot comparison

---

## 5. Date Feature Engineering

The join_date column was converted into datetime format.

New features extracted:

- join_year
- join_month
- join_day
- join_weekday

These features help machine learning models capture temporal patterns.

---

## 6. Categorical Encoding

### Ordinal Encoding

Applied to:

- education_level
''' from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[['Primary', 'Secondary', 'Graduate', 'Post-Graduate']])
df[['education_level']] = oe.fit_transform(df[['education_level']]) '''


Order:

```text
Primary
Secondary
Graduate
Post-Graduate
```

### Label Encoding

Applied to:

- gender

### One-Hot Encoding

Applied to:

- region
- loan_purpose

---

## 7. Feature Scaling

### Standardization

Implemented using:

```python
StandardScaler()
```

Applied to:

- age
- annual_income
- loan_amount
- credit_score
- transaction_count
- spending_ratio

Purpose:

- Normalize feature scales
- Improve model performance
- Prevent dominance of large-valued features

---

## 8. Data Transformation

### Log Transformation

Applied to:

- annual_income

Purpose:

- Reduce skewness
- Compress extreme values
- Improve distribution

Implemented using:

```python
FunctionTransformer(np.log1p)
```

---

## 9. Feature Engineering

Three new features were created.

### Debt-to-Income Ratio

```python
loan_amount / annual_income
```

Purpose:

Measures customer's debt burden.

---

### Average Monthly Transactions

```python
transaction_count / 6
```

Purpose:

Provides monthly customer activity.

---

### Spending-to-Income Ratio

```python
spending_ratio / 100
```

Purpose:

Measures the proportion of income spent.

---

## Final Dataset

The final dataset:

- Contains no missing values
- Includes encoded categorical features
- Includes standardized numerical features
- Includes engineered business features
- Is ready for machine learning modeling

---

## Suitable Machine Learning Models

The processed dataset can be used for:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- Support Vector Machine (SVM)

---

## Project Outcome

Successfully developed an end-to-end data preprocessing and feature engineering pipeline for customer credit risk analysis. The dataset was cleaned, transformed, encoded, standardized, and enriched with meaningful features, making it suitable for machine learning model development and loan default prediction.
