# Multi-Domain Prediction & Analysis Project

## Project Overview

This project consists of three distinct data analysis and prediction modules using Python and machine learning techniques:

1. **Traffic Accident Analysis** – Predicting the likelihood of traffic accidents based on vehicle, driver, and environmental factors.
2. **Household Power Consumption Forecasting** – Predicting next-day energy demand using historical household electricity consumption data.
3. **Air Quality and Pollution Analysis** – Predicting Air Quality Index (AQI) using pollution and environmental sensor data.

Each module includes **data preprocessing, feature engineering, visualization, model training, evaluation, and model saving** for future use.

---

## 1. Traffic Accident Analysis

### Objective

Predict traffic accidents using historical traffic, vehicle, driver, and environmental data.

### Dataset

`dataset_traffic_accident_prediction1.csv` – contains features like:

* Traffic_Density
* Speed_Limit
* Number_of_Vehicles
* Driver_Alcohol
* Driver_Age
* Driver_Experience
* Weather
* Road_Type
* Time_of_Day
* Accident_Severity
* Road_Condition
* Vehicle_Type
* Road_Light_Condition
* Target: `Accident` (binary: accident occurred or not)

### Workflow

1. **Data Cleaning & Imputation**

   * Fill missing numeric values with median.
   * Fill categorical missing values with mode.
   * Drop duplicates.

2. **Exploratory Data Analysis**

   * Histograms for numeric features with mean/median lines.
   * Boxplots to detect outliers.
   * Pairplots for feature relationships.
   * Spearman correlation heatmap for numeric features.

3. **Feature Engineering**

   * One-hot encoding for categorical variables.
   * Created a new feature `Age_vs_Experience = Driver_Age - Driver_Experience`.
   * Dropped original `Driver_Age` and `Driver_Experience`.

4. **Modeling**

   * RandomForestClassifier
   * XGBClassifier

5. **Evaluation**

   * Accuracy score
   * Confusion matrix
   * Classification report

6. **Model Saving**

   * `joblib.dump(classifier, "model/traffic_model.pkl")`

---

## 2. Household Power Consumption Forecasting

### Objective

Forecast next-day household energy demand using historical power consumption data.

### Dataset

`household_power_consumption.txt` – contains:

* Date, Time
* Global_active_power
* Global_reactive_power
* Voltage
* Global_intensity
* Sub_metering_1/2/3

### Workflow

1. **Data Preparation**

   * Convert Date & Time to `datetime`.
   * Sort by `DateTime`.
   * Convert numeric columns to float.
   * Interpolate missing values linearly.
   * Create target `NextDay_Global_active_power` by shifting `Global_active_power` by 1440 minutes (1 day).

2. **Feature Selection**

   * Features: `Global_active_power`, `Global_reactive_power`, `Voltage`, `Global_intensity`
   * Target: `NextDay_Global_active_power`

3. **Train-Test Split**

   * 80%-20%, no shuffle (time series)

4. **Modeling**

   * RandomForestRegressor with 100 estimators

5. **Evaluation**

   * RMSE
   * R² score
   * Plot actual vs predicted for visual comparison

6. **Model Saving**

   * `joblib.dump(model, "model/energy_consumption.pkl")`

---

## 3. Air Quality and Pollution Analysis

### Objective

Predict Air Quality Index (AQI) based on air pollution and environmental sensor readings.

### Dataset

`air_quality_cleaned.csv` – contains columns like:

* CO(GT), NO2(GT), PT08.S5(O3), etc.
* Date, Time

### Workflow

1. **Data Cleaning**

   * Strip whitespace & lowercase column names
   * Remove unnamed columns
   * Convert numeric columns, coerce errors
   * Fill missing numeric values with median

2. **Feature Engineering**

   * Create synthetic AQI:
     `AQI = 0.4*CO(GT) + 0.3*NO2(GT) + 0.3*PT08.S5(O3)`
   * Categorize AQI into classes:

     * Good, Moderate, Unhealthy, Very Unhealthy, Severe, Hazardous

3. **Modeling**

   * XGBRegressor with:

     * `n_estimators=250`
     * `learning_rate=0.05`
     * `max_depth=6`
     * `subsample=0.8`
     * `colsample_bytree=0.8`

4. **Evaluation**

   * RMSE
   * R² score
   * Scatter plot: Actual vs Predicted AQI

5. **Model Saving**

   * `joblib.dump(xgb_aqi, "model/aqi.pkl")`

---

## Requirements

* Python 3.9+
* Libraries:

  * `pandas`, `numpy`, `matplotlib`, `seaborn`
  * `scikit-learn`
  * `xgboost`
  * `joblib`

Install using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

---

## Usage

1. Clone the project and place the datasets in the specified folder.
2. Run each analysis module separately:

   * `traffic_accident_analysis.ipynb`
   * `household_power_forecasting.ipynb`
   * `air_quality_analysis.ipynb`
3. Trained models will be saved in `model/` folder for later predictions.

---

## Conclusion

This project provides:

* Predictive modeling for traffic accidents, household energy demand, and air quality.
* Complete preprocessing, feature engineering, and evaluation pipeline.
* Ready-to-use saved models for real-time predictions.
