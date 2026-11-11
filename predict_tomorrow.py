import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier

print("Fetching latest market data...")

# 1. Download data up to today
ticker = "AAPL"
df = yf.download(ticker, start="2023-01-01")

# Clean up multi-index columns if they exist
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# 2. Recreate features (same as our training setup)
df['Price_Change'] = df['Close'].pct_change()
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_20'] = df['Close'].rolling(window=20).mean()

rsi = RSIIndicator(close=df['Close'], window=14)
df['RSI'] = rsi.rsi()

macd = MACD(close=df['Close'])
df['MACD'] = macd.macd_diff()

df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.dropna()

# 3. Features for training
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'MA_5', 'MA_20', 'RSI', 'MACD']
X = df[features]
y = df['Target']

# 4. Train the model on ALL available data to make the latest prediction
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. Grab the very last row of data (today's latest stats) to predict tomorrow
latest_data = X.iloc[[-1]]
prediction = model.predict(latest_data)[0]
probability = model.predict_proba(latest_data)

print("\n--- TRADING BOT PREDICTION ---")
if prediction == 1:
    print("Prediction for Tomorrow: 🟢 BULLISH (Expected to go UP)")
else:
    print("Prediction for Tomorrow: 🔴 BEARISH (Expected to go DOWN)")

print(f"Confidence (Down / Up): {probability[0][0]*100:.2f}% / {probability[0][1]*100:.2f}%")