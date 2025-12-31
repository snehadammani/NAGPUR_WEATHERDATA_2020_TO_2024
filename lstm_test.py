import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load

# -----------------------------
# 1. Load model & scaler
# -----------------------------
model = load_model("lstm_weather_model.h5")
scaler = load("lstm_scaler.pkl")

print("✅ LSTM model & scaler loaded")

# -----------------------------
# 2. USER INPUT: DATE RANGE
# -----------------------------
start_date = "2026-01-01"
end_date = "2026-01-07"

future_dates = pd.date_range(start=start_date, end=end_date)
n_future = len(future_dates)

# -----------------------------
# 3. Load dataset (for history)
# -----------------------------
df = pd.read_csv("new_weather_cleaned_csv.csv")
df = df.dropna()

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
elif 'datetime' in df.columns:
    df['date'] = pd.to_datetime(df['datetime'])

df = df.sort_values("date")

# -----------------------------
# 4. Select same features as training
# -----------------------------
maxt_col = 'tempmax' if 'tempmax' in df.columns else 'maxtemp'
mint_col = 'tempmin' if 'tempmin' in df.columns else 'mintemp'
windspeed_col = 'windspeed_capped' if 'windspeed_capped' in df.columns else 'windspeed'

features = df[
    [
        maxt_col,
        mint_col,
        'humidity',
        windspeed_col,
        'pressure'
    ]
].values

# -----------------------------
# 5. Scale features
# -----------------------------
features_scaled = scaler.transform(features)

# -----------------------------
# 6. Prepare initial window
# -----------------------------
WINDOW_SIZE = 7
current_window = features_scaled[-WINDOW_SIZE:].copy()

predictions = []

# -----------------------------
# 7. Recursive Prediction
# -----------------------------
for _ in range(n_future):
    input_window = current_window.reshape(1, WINDOW_SIZE, current_window.shape[1])
    
    pred_scaled = model.predict(input_window, verbose=0)
    pred = scaler.inverse_transform(pred_scaled)[0]
    
    predictions.append(pred)
    
    # scale predicted output and append to window
    pred_scaled_for_window = scaler.transform(pred.reshape(1, -1))
    current_window = np.vstack([current_window[1:], pred_scaled_for_window])

# -----------------------------
# 8. Create output DataFrame
# -----------------------------
result = pd.DataFrame(
    predictions,
    columns=[
        "Pred_MaxTemp",
        "Pred_MinTemp",
        "Pred_Humidity",
        "Pred_WindSpeed",
        "Pred_Pressure"
    ]
)

result.insert(0, "date", future_dates)

print("\n🌤️ LSTM Weather Predictions (Date Range):")
print(result)

# -----------------------------
# 9. Save to CSV
# -----------------------------
result.to_csv("2026_lstm_predictions.csv", index=False)
print("\n📁 Predictions saved as 2026_lstm_predictions.csv")