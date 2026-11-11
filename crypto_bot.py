import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Downloading historical crypto data (Bitcoin)...")

# 1. Download historical data for Bitcoin (BTC-USD)
# Crypto trades 24/7, so we include recent data up to today
ticker = "BTC-USD"
df = yf.download(ticker, start="2023-01-01")

# Clean up multi-index columns if they exist
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# 2. Feature Engineering (RSI, MACD, Moving Averages)
df['Price_Change'] = df['Close'].pct_change()
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_20'] = df['Close'].rolling(window=20).mean()

# Add RSI
rsi = RSIIndicator(close=df['Close'], window=14)
df['RSI'] = rsi.rsi()

# Add MACD
macd = MACD(close=df['Close'])
df['MACD'] = macd.macd_diff()

# 3. Create Target: 1 if tomorrow's Close > today's Close, else 0
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop missing values
df = df.dropna()

# 4. Select features
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'MA_5', 'MA_20', 'RSI', 'MACD']
X = df[features]
y = df['Target']

# 5. Split into training and testing sets (80% train, 20% test)
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# 6. Train the Model using our best hyperparameter settings found earlier (Trees: 50, Max Depth: 3)
model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate on Test Set
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# 8. Predict Tomorrow's Direction using ALL data
model.fit(X, y)
latest_data = X.iloc[[-1]]
tomorrow_pred = model.predict(latest_data)[0]
probability = model.predict_proba(latest_data)

print(f"\n--- BITCOIN MODEL RESULTS ---")
print(f"Test Accuracy on Historical Data: {accuracy * 100:.2f}%")

print("\n--- TOMORROW'S PREDICTION FOR BTC ---")
if tomorrow_pred == 1:
    print("Prediction: 🟢 BULLISH (Expected to go UP)")
else:
    print("Prediction: 🔴 BEARISH (Expected to go DOWN)")

print(f"Confidence (Down / Up): {probability[0][0]*100:.2f}% / {probability[0][1]*100:.2f}%")