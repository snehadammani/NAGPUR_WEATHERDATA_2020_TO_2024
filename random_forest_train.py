import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump
import json

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("new_weather_cleaned_csv.csv")
df = df.dropna()

# -----------------------------
# 2. Date Feature Engineering
# -----------------------------
# support datasets that use either 'date' or 'datetime'
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

# -----------------------------
# 🔥 2.1 Cyclic Encoding (NEW)
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

# map available column names in the CSV to expected target names
maxt_col = 'tempmax' if 'tempmax' in df.columns else ('maxtemp' if 'maxtemp' in df.columns else None)
mint_col = 'tempmin' if 'tempmin' in df.columns else ('mintemp' if 'mintemp' in df.columns else None)
windspeed_col = 'windspeed_capped' if 'windspeed_capped' in df.columns else ('windspeed' if 'windspeed' in df.columns else None)

if not all([maxt_col, mint_col, windspeed_col, 'humidity' in df.columns, 'pressure' in df.columns]):
    missing = []
    if not maxt_col:
        missing.append('tempmax/maxtemp')
    if not mint_col:
        missing.append('tempmin/mintemp')
    if not windspeed_col:
        missing.append('windspeed_capped/windspeed')
    if 'humidity' not in df.columns:
        missing.append('humidity')
    if 'pressure' not in df.columns:
        missing.append('pressure')
    raise KeyError(f"Missing expected target columns: {', '.join(missing)}")

y = df[[maxt_col, mint_col, 'humidity', windspeed_col, 'pressure']].copy()
y.columns = ['maxtemp', 'mintemp', 'humidity', 'windspeed', 'pressure']

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. RandomForest Model
# -----------------------------
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# -----------------------------
# 6. Predictions
# -----------------------------
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

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
dump(rf_model, "randomforest_weather.pkl")

with open("randomforest_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ RandomForest model saved as randomforest_weather.pkl")
print("📊 Metrics saved as randomforest_metrics.json")
print(metrics)
