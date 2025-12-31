import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from joblib import dump

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("new_weather_cleaned_csv.csv")
df = df.dropna()

# Handle date column
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
elif 'datetime' in df.columns:
    df['date'] = pd.to_datetime(df['datetime'])

df = df.sort_values("date")

# -----------------------------
# 2. Select Features (REAL WEATHER FEATURES)
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
# 3. Scaling
# -----------------------------
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
dump(scaler, "lstm_scaler.pkl")

# -----------------------------
# 4. Create Sliding Windows
# -----------------------------
def create_sequences(data, window_size=7):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

WINDOW_SIZE = 7
X, y = create_sequences(features_scaled, WINDOW_SIZE)

# -----------------------------
# 5. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# -----------------------------
# 6. LSTM Model
# -----------------------------
model = Sequential([
    LSTM(64, activation='tanh', input_shape=(X.shape[1], X.shape[2])),
    Dense(32, activation='relu'),
    Dense(y.shape[1])  # multi-output
])

model.compile(
    optimizer='adam',
    loss=MeanSquaredError(),
    metrics=[MeanAbsoluteError()]
)

# -----------------------------
# 7. Training
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 8. Save Model
# -----------------------------
model.save("lstm_weather_model.h5")

print("✅ LSTM model trained and saved as lstm_weather_model.h5")
