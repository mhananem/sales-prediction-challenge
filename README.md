# IronKaggle — Sales Prediction Challenge

## Overview

Pair programming competition where the goal was to build a predictive model for daily store sales using historical retail data. The model was trained on 640,000+ rows of data and used to generate predictions on a held-out dataset within a 1-hour window.

**Team:** Hanane & [Partner name]  
**Date:** [Date]  
**Target variable:** `Sales` (daily sales per store)  
**Metric:** RMSE (Root Mean Squared Error)

---

## Dataset

**Training data:** [`sales.csv`](https://raw.githubusercontent.com/data-bootcamp-v4/data/main/sales.csv) — 640,840 rows  
**Prediction data:** [`ironkaggle_notarget.csv`](https://raw.githubusercontent.com/data-bootcamp-v4/data/main/ironkaggle_notarget.csv) — 71,205 rows

| Column | Description |
|---|---|
| `Store_ID` | Unique identifier for each shop |
| `Day_of_week` | Day encoded 1–7 |
| `Date` | Date of the record |
| `Nb_customers_on_day` | Number of customers that visited |
| `Open` | 0 = closed, 1 = open |
| `Promotion` | 0 = no promotion, 1 = promotion active |
| `State_holiday` | 0 = none, a/b/c = different state holidays |
| `School_holiday` | 0 = no school holiday, 1 = school holiday |
| `Sales` | Daily sales (target variable, train only) |

---

## Methodology

### 1. Exploratory Data Analysis
- Inspected shapes, dtypes, and null values across both datasets
- Checked value distributions for all categorical and binary columns
- Verified the Open/Sales relationship (closed shops always have Sales = 0)

### 2. Data Cleaning
- Dropped 31 anomalous rows where `Open = 1` but `Sales = 0`
- Confirmed no null values in either dataset

### 3. Feature Engineering
- Parsed `Date` (object) into three numeric features: `Month`, `Year`, `Day`
- Encoded `State_holiday` from categorical (0, a, b, c) to numeric (0, 1, 2, 3)

### 4. Filtering
- Removed `Open = 0` rows from training data — closed shops always have Sales = 0, training on them would distort the model
- Final training set: **531,985 rows**

### 5. Modelling
- **Model:** Random Forest Regressor (ensemble/bagging method)
- **Why Random Forest:** handles mixed feature types natively, robust to outliers, no normalisation required, strong performance on tabular data
- **No normalisation applied** — not needed for tree-based models which rely on split thresholds, not distances or gradients
- Train/test split: 80/20 with `random_state=42`

```python
features = ["Store_ID", "Day_of_week", "Nb_customers_on_day",
            "Promotion", "State_holiday", "School_holiday",
            "Month", "Year", "Day"]

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
```

### 6. Prediction
- Applied same preprocessing to `df_predict`
- Forced `Sales = 0` for all rows where `Open = 0`
- Exported `True_index` + predicted `Sales` to CSV

---

## Results

| Metric | Score |
|---|---|
| MAE | 645.81 |
| RMSE | 955.24 |
| R² | 0.9049 |

The model explains **90.5% of the variance** in daily sales. The average prediction error is ±645 sales units against a mean sales value of ~6,965 — approximately **13.7% of the mean**, which is a strong result for retail sales forecasting without store-level metadata.

---

## Repository Structure

```
├── notebook.ipynb        # Full analysis and modelling notebook
├── my_predictions.csv    # Final predictions submitted
└── README.md
```

---

## Dependencies

```
pandas
numpy
scikit-learn
matplotlib
```
