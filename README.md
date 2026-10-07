# Road Accident Risk Prediction

This repository contains a machine learning workflow designed to analyze environmental, roadway, temporal, and historical factors to accurately predict road accident risk levels (`accident_risk`).

![Road Safety Banner](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSp-MN26pXrgld5kvQDYRt5M_gpg5gJIaQpxT6KKue9IK0SjBan6gFfZyPE&s=10)

---

## 📌 Project Overview

Traffic accidents are influenced by various conditions such as road curvature, speed limits, lighting, weather, and time of day. The goal of this project is to build a robust predictive regression pipeline using tabular dataset analysis, exploratory data analysis (EDA), feature pre-processing, and modern gradient boosting algorithms like **LightGBM** and **XGBoost**.

---

## 📊 Dataset Overview

The dataset contains over **517,000 recorded road entries** with numerical, categorical, and boolean attributes.

* **Target Variable:** `accident_risk` (Continuous float value)

### Feature Summary
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | Integer | Unique identifier for each entry |
| `road_type` | Categorical | Type of road (`urban`, `rural`, `highway`) |
| `num_lanes` | Integer | Number of lanes (1 to 4) |
| `curvature` | Float | Degree of road curvature (0.00 to 1.00) |
| `speed_limit` | Integer | Posted speed limit in mph/kmh |
| `lighting` | Categorical | Lighting condition (`daylight`, `dim`, `night`) |
| `weather` | Categorical | Weather condition (`clear`, `rainy`, `foggy`) |
| `road_signs_present` | Boolean | Presence of traffic signs |
| `public_road` | Boolean | Whether the road is a public road |
| `time_of_day` | Categorical | Time of day (`morning`, `afternoon`, `evening`, etc.) |
| `holiday` | Boolean | Indicator for official holidays |
| `school_season` | Boolean | Indicator for active school season |
| `num_reported_accidents` | Integer | Number of previously reported accidents on the road |

---

## 🔍 Exploratory Data Analysis (Key Findings)

From the correlation analysis and statistical exploration:
* **Curvature** shows the strongest positive correlation with accident risk ($\approx 0.54$).
* **Speed Limit** shows a significant positive correlation with accident risk ($\approx 0.43$).
* **Number of Reported Accidents** positively correlates with risk ($\approx 0.21$).
* The dataset contains **no missing values** across all 517,754 rows.

---

## 🛠️ Tech Stack & Requirements

* **Python 3.8+**
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`, `lightgbm`, `xgboost`



## 🚀 Pipeline Workflow

1. **Exploratory Data Analysis (EDA):** Checking distributions, summary statistics, missing values, and feature correlations.
2. **Preprocessing:** Encoding categorical attributes and scaling numerical inputs using `StandardScaler`.
3. **Model Training & Evaluation:** Training gradient boosting regressors (`LGBMRegressor`, `XGBRegressor`) evaluated via Root Mean Squared Error (RMSE).
4. **Feature Importance Analysis:** Evaluating key predictors using permutation importance techniques.
