import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier

print("Running Backtest and Generating Chart...")

# 1. Download historical data
ticker = "AAPL"
df = yf.download(ticker, start="2023-01-01", end="2026-01-01")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# 2. Recreate features
df['Price_Change'] = df['Close'].pct_change()
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_20'] = df['Close'].rolling(window=20).mean()

rsi = RSIIndicator(close=df['Close'], window=14)
df['RSI'] = rsi.rsi()

macd = MACD(close=df['Close'])
df['MACD'] = macd.macd_diff()

df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.dropna()

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'MA_5', 'MA_20', 'RSI', 'MACD']
X = df[features]
y = df['Target']

# 3. Walk-Forward Backtest simulation
start_train_size = int(len(df) * 0.6)
predictions = []
actuals = []

model = RandomForestClassifier(n_estimators=100, random_state=42)

for i in range(start_train_size, len(df) - 1):
    X_train = X.iloc[:i]
    y_train = y.iloc[:i]
    
    X_test = X.iloc[[i]]
    y_test = y.iloc[i]
    
    model.fit(X_train, y_train)
    pred = model.predict(X_test)[0]
    
    predictions.append(pred)
    actuals.append(y_test)

# 4. Compile results
df_results = pd.DataFrame({
    'Close': df['Close'].iloc[start_train_size:-1],
    'Prediction': predictions, 
    'Actual': actuals
}, index=df.index[start_train_size:-1])

df_results['Correct'] = df_results['Prediction'] == df_results['Actual']
win_rate = df_results['Correct'].mean() * 100

print(f"\n--- BACKTEST RESULTS ---")
print(f"Strategy Win Rate: {win_rate:.2f}%")

# 5. Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(df_results.index, df_results['Close'], label='Stock Close Price', color='blue', alpha=0.5)

# Highlight correct vs incorrect predictions on the chart
correct_mask = df_results['Correct'] == True
incorrect_mask = df_results['Correct'] == False

plt.scatter(df_results.index[correct_mask], df_results['Close'][correct_mask], 
            color='green', label='Correct Prediction', marker='^', s=30)
plt.scatter(df_results.index[incorrect_mask], df_results['Close'][incorrect_mask], 
            color='red', label='Incorrect Prediction', marker='v', s=30)

plt.title(f'AI Trading Bot Backtest Performance - {ticker} (Win Rate: {win_rate:.2f}%)')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True)
plt.show()