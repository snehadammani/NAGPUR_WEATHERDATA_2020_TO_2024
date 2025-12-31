import pandas as pd
import numpy as np
from joblib import load

# -----------------------------
# 1. Load trained XGBoost model
# -----------------------------
model = load("xgboost_weather.pkl")
print("✅ XGBoost model loaded")

# -----------------------------
# 2. USER INPUT (DATE or DATE RANGE)
# -----------------------------
# Single date example:
# start_date = "2025-01-01"
# end_date = "2025-01-01"

# Date range example:
start_date = "2025-01-01"
end_date = "2025-01-08"

date_range = pd.date_range(start=start_date, end=end_date)
df_input = pd.DataFrame({"date": date_range})

# -----------------------------
# 3. Feature Engineering (SAME AS TRAINING)
# -----------------------------
df_input['year'] = df_input['date'].dt.year
df_input['month'] = df_input['date'].dt.month
df_input['day'] = df_input['date'].dt.day
df_input['dayofweek'] = df_input['date'].dt.dayofweek
df_input['weekofyear'] = df_input['date'].dt.isocalendar().week.astype(int)

# Cyclic encoding
df_input['month_sin'] = np.sin(2 * np.pi * df_input['month'] / 12)
df_input['month_cos'] = np.cos(2 * np.pi * df_input['month'] / 12)

df_input['dow_sin'] = np.sin(2 * np.pi * df_input['dayofweek'] / 7)
df_input['dow_cos'] = np.cos(2 * np.pi * df_input['dayofweek'] / 7)

# -----------------------------
# 4. Feature Selection (MATCH TRAINING)
# -----------------------------
X_test = df_input[
    [
        'year',
        'weekofyear',
        'month_sin', 'month_cos',
        'dow_sin', 'dow_cos'
    ]
]

# -----------------------------
# 5. Prediction
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# 6. Output Formatting
# -----------------------------
result = pd.concat(
    [
        df_input['date'],
        pd.DataFrame(
            predictions,
            columns=[
                "Pred_MaxTemp",
                "Pred_MinTemp",
                "Pred_Humidity",
                "Pred_WindSpeed",
                "Pred_Pressure"
            ]
        )
    ],
    axis=1
)

print("\n🌤️ XGBoost Weather Predictions:")
print(result)

# -----------------------------
# 7. Save Predictions
# -----------------------------
result.to_csv("xgboost_predictions.csv", index=False)
print("\n📁 Predictions saved as xgboost_predictions.csv")
