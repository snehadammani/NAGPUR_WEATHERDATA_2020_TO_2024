# NAGPUR_WEATHERDATA_2020_TO_2024
## Weather Prediction for Nagpur (2020–2026) using ML & LSTM

This project focuses on analyzing historical weather data for Nagpur and predicting future weather patterns using Machine Learning and Deep Learning techniques. The goal is to compare traditional regression models with time-series forecasting models and evaluate their performance.

---

## Project Objectives

- Analyze historical weather data (2020–2024)
- Perform exploratory data analysis and visualization
- Predict weather parameters using Machine Learning models
- Apply LSTM for short-term time-series forecasting
- Compare RandomForest, XGBoost, and LSTM models
- Predict weather for future dates (1 Jan 2026 – 7 Jan 2026)

---

## Dataset Description

- Location: Nagpur, India
- Time Period: 2020–2024
- Data Format: CSV
- Key Features:
  - Maximum Temperature
  - Minimum Temperature
  - Humidity
  - Wind Speed
  - Pressure
  - Rain Information

Missing values were removed and the data was cleaned before analysis and modeling.

---

## Exploratory Data Analysis (EDA)

### 1. Distribution of Weather Variables
![Distribution of Weather Variables](images/Distribution%20plots%20for%20key%20weather%20variables.png)

This plot shows the distribution of major weather parameters such as temperature, humidity, wind speed, and pressure.

---

### 2. Correlation Heatmap
![Correlation Heatmap](images/Correlation%20Heatmap%20with%20Values.png)

This heatmap visualizes relationships between numerical weather variables.

---

### 3. Outlier Analysis
![Outlier Analysis](images/Key%20numeric%20columns%20where%20outliers%20usually%20exist.png)

This plot highlights columns where outliers are commonly present.

---

### 4. Rain vs No-Rain Days
![Rain vs No-Rain Days](images/Count%20of%20Rain%20vs%20No-Rain%20days.png)

This bar chart shows the distribution of rainy and non-rainy days.

---

## Machine Learning Models

### RandomForest Regressor
- Used as a baseline regression model
- Input: Date-based engineered features
- Strength: Good generalization and stable predictions

### XGBoost Regressor
- Boosting-based regression model
- Strong learner but sensitive to overfitting
- Overfitting observed due to limited input features

---

## Model Comparison Visualization
![Model Comparison](images/RF%20vs%20XGBoost%20vs%20LSTM%20–%20Max%20Temperature%20Prediction%20Comparison.png)

This graph compares predictions from RandomForest, XGBoost, and LSTM models.

---

## Deep Learning Model: LSTM

### LSTM Training Approach
- Sliding window of last 7 days
- Predicts the next day’s weather
- Captures temporal dependency

### LSTM Future Prediction (2026)
![LSTM Prediction](images/LSTM%20Weather%20Prediction%20(1%20Jan%202026%20–%207%20Jan%202026).png)

This graph shows LSTM-based predictions for maximum and minimum temperature.

---

## Model Performance Summary

The following tables summarize the performance of all implemented models using standard evaluation metrics.  
Lower values of MAE and RMSE indicate better performance, while higher R² values indicate better generalization.

---

### RandomForest Metrics

| Metric | Train | Test |
|------|------|------|
| MAE | 1.88 | 2.68 |
| RMSE | 2.99 | 4.21 |
| R² | 0.88 | 0.76 |

**Observation:**  
RandomForest shows strong generalization with the highest test R² score and relatively low error on unseen data.

---

### XGBoost Metrics

| Metric | Train | Test |
|------|------|------|
| MAE | 1.59 | 2.96 |
| RMSE | 2.53 | 4.73 |
| R² | 0.92 | 0.69 |

**Observation:**  
XGBoost performs very well on training data but shows reduced performance on test data, indicating overfitting.

---

### LSTM Metrics

| Metric | Train | Validation |
|------|------|-----------|
| Loss (MSE) | 0.0086 | 0.0069 |
| MAE | 0.0627 | 0.0574 |
| R² | — | — |

**Note:**  
LSTM metrics are calculated on scaled time-series data, so direct comparison with regression RMSE values is not applicable.

---

### Final Comparison Summary

| Model | MAE | RMSE | R² |
|------|------|------|----|
| RandomForest (Test) | 2.68 | 4.21 | 0.76 |
| XGBoost (Test) | 2.96 | 4.73 | 0.69 |
| LSTM (Validation) | 0.0574 | — | — |

---

**Overall Conclusion:**  
RandomForest achieved the best balance between error and generalization for regression-based prediction.  
XGBoost showed signs of overfitting.  
LSTM demonstrated stable short-term time-series forecasting with low validation loss.



---

## Future Scope

- Integration with real-time weather APIs
- Longer time-series forecasting
- Deployment as a web-based dashboard
- Additional atmospheric parameters
