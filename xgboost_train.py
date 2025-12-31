import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump
import json
from xgboost import XGBRegressor

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("new_weather_cleaned_csv.csv")
df = df.dropna()

# -----------------------------
# 2. Date Feature Engineering
# -----------------------------
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
elif 'datetime' in df.columns:
    df['date'] = pd.to_datetime(df['datetime'])
else:
    raise KeyError("CSV must contain a 'date' or 'datetime' column")

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)

# detect correct max temperature column
maxt_col = 'tempmax' if 'tempmax' in df.columns else ('maxtemp' if 'maxtemp' in df.columns else None)

if maxt_col is None:
    raise KeyError("No max temperature column found (tempmax / maxtemp)")

# outlier control using detected column
df = df[df[maxt_col] < df[maxt_col].quantile(0.99)]


# -----------------------------
# 2.1 Cyclic Encoding
# -----------------------------
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

# -----------------------------
# 3. Features & Targets
# -----------------------------
X = df[
    [
        'year',
        'weekofyear',
        'month_sin', 'month_cos',
        'dow_sin', 'dow_cos'
    ]
]

maxt_col = 'tempmax' if 'tempmax' in df.columns else ('maxtemp' if 'maxtemp' in df.columns else None)
mint_col = 'tempmin' if 'tempmin' in df.columns else ('mintemp' if 'mintemp' in df.columns else None)
windspeed_col = 'windspeed_capped' if 'windspeed_capped' in df.columns else ('windspeed' if 'windspeed' in df.columns else None)

y = df[[maxt_col, mint_col, 'humidity', windspeed_col, 'pressure']].copy()
y.columns = ['maxtemp', 'mintemp', 'humidity', 'windspeed', 'pressure']

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. XGBoost Model
# -----------------------------
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

# -----------------------------
# 6. Predictions
# -----------------------------
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

# -----------------------------
# 7. Metrics
# -----------------------------
metrics = {
    "train_MAE": mean_absolute_error(y_train, y_train_pred),
    "test_MAE": mean_absolute_error(y_test, y_test_pred),

    "train_RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
    "test_RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),

    "train_R2": r2_score(y_train, y_train_pred),
    "test_R2": r2_score(y_test, y_test_pred)
}

# -----------------------------
# 8. Save Model & Metrics
# -----------------------------
dump(xgb_model, "xgboost_weather.pkl")

with open("xgboost_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ XGBoost model saved as xgboost_weather.pkl")
print("📊 Metrics saved as xgboost_metrics.json")
print(metrics)
