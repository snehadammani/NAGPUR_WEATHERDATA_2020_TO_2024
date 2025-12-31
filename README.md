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

![Distribution of Weather Variables](https://github.com/snehadammani/NAGPUR_WEATHERDATA_2020_TO_2024/blob/e8cfdaa2bc7a63678d6dd513081c8d555f3e08c4/Distribution%20plots%20for%20key%20weather%20variables.png?raw=true)

This plot shows the distribution of major weather parameters such as temperature, humidity, wind speed, and pressure.

---

### 2. Correlation Heatmap

![Correlation Heatmap with Values](https://github.com/snehadammani/NAGPUR_WEATHERDATA_2020_TO_2024/blob/5653eba0933db3224438e5c453042281ecf78aed/Correlation%20Heatmap%20with%20Values.png?raw=true)

The correlation heatmap visualizes the relationship between numerical weather variables.

---

### 3. Outlier Analysis

![Outlier Analysis](https://github.com/snehadammani/NAGPUR_WEATHERDATA_2020_TO_2024/blob/5653eba0933db3224438e5c453042281ecf78aed/Key%20numeric%20columns%20where%20outliers%20usually%20exist.png?raw=true)

This visualization highlights columns where outliers commonly appear.

---

### 4. Rain vs No-Rain Days Analysis

![Rain vs No-Rain Days](https://github.com/snehadammani/NAGPUR_WEATHERDATA_2020_TO_2024/blob/01660c1113c034f2c14ae56068e7c78314581f73/Count%20of%20Rain%20vs%20No-Rain%20days.png)

This plot provides insight into rainfall patterns in the dataset.

---

## Model Comparison Visualization

![RF vs XGBoost vs LSTM – Max Temperature Prediction Comparison](https://github.com/snehadammani/NAGPUR_WEATHERDATA_2020_TO_2024/blob/5653eba0933db3224438e5c453042281ecf78aed/RF%20vs%20XGBoost%20vs%20LSTM%20%E2%80%93%20Max%20Temperature%20Prediction%20Comparison.png?raw=true)

This graph compares the predicted maximum temperature from RandomForest, XGBoost, and LSTM models.

---

## Deep Learning Model: LSTM

### LSTM Training Approach
- Sliding window of last 7 days
- Predicts the next day’s weather
- Captures temporal dependency in time-series data

### LSTM Future Prediction (2026)

![LSTM Weather Prediction](https://github.com/snehadammani/NAGPUR_WEATHERDATA_2020_TO_2024/blob/5653eba0933db3224438e5c453042281ecf78aed/LSTM%20Weather%20Prediction%20(1%20Jan%202026%20%E2%80%93%207%20Jan%202026).png?raw=true)

This graph shows LSTM-based predictions for maximum and minimum temperature from 1 January 2026 to 7 January 2026.

---

## Model Performance Summary

### RandomForest Metrics

| Metric | Train | Test |
|------|------|------|
| MAE | 1.88 | 2.68 |
| RMSE | 2.99 | 4.21 |
| R² | 0.88 | 0.76 |

---

### XGBoost Metrics

| Metric | Train | Test |
|------|------|------|
| MAE | 1.59 | 2.96 |
| RMSE | 2.53 | 4.73 |
| R² | 0.92 | 0.69 |

---

### LSTM Metrics

| Metric | Train | Validation |
|------|------|-----------|
| Loss (MSE) | 0.0086 | 0.0069 |
| MAE | 0.0627 | 0.0574 |
| R² | — | — |

---

### Final Comparison Summary

| Model | MAE | RMSE | R² |
|------|------|------|----|
| RandomForest (Test) | 2.68 | 4.21 | 0.76 |
| XGBoost (Test) | 2.96 | 4.73 | 0.69 |
| LSTM (Validation) | 0.0574 | — | — |

---

## Conclusion

RandomForest performs well for regression-based weather prediction, while LSTM is suitable for short-term time-series forecasting. Combining ML and DL approaches provides a comprehensive understanding of weather prediction behavior.

---

## Future Scope

- Integration with real-time weather APIs
- Longer time-series forecasting
- Deployment as a web-based dashboard
- Incorporation of additional atmospheric parameters
