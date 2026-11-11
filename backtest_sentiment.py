import yfinance as yf
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

print("Running Sentiment-Enhanced Backtest Simulation...")

ticker = "AAPL"
stock = yf.Ticker(ticker)

# 1. Fetch live news sentiment
try:
    news_list = stock.news
except Exception:
    news_list = []

compound_scores = []
if news_list:
    for item in news_list[:5]:
        title = item.get('title') or item.get('content', {}).get('title', 'No Title')
        score = sia.polarity_scores(title)['compound']
        compound_scores.append(score)
    avg_sentiment = sum(compound_scores) / len(compound_scores) if compound_scores else 0.0
else:
    avg_sentiment = 0.0

print(f"Current Baseline News Sentiment Score: {avg_sentiment:.2f}")

# 2. Download historical data
df = yf.download(ticker, start="2023-01-01", end="2026-01-01")
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# 3. Feature Engineering
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

# 4. Walk-Forward Backtest with Sentiment Adjustment
start_train_size = int(len(df) * 0.6)
predictions = []
actuals = []
sentiment_biases = []

model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)

for i in range(start_train_size, len(df) - 1):
    X_train = X.iloc[:i]
    y_train = y.iloc[:i]
    
    X_test = X.iloc[[i]]
    y_test = y.iloc[i]
    
    model.fit(X_train, y_train)
    pred = model.predict(X_test)[0]
    
    # Apply sentiment boost/penalty logic during backtest simulation
    # If recent baseline sentiment is strongly positive, bias prediction towards Up (1)
    adjusted_pred = pred
    if avg_sentiment > 0.05 and pred == 0:
        adjusted_pred = 1  # Sentiment override
    
    predictions.append(adjusted_pred)
    actuals.append(y_test)

# 5. Compile Results & Calculate Win Rate
df_results = pd.DataFrame({
    'Close': df['Close'].iloc[start_train_size:-1],
    'Prediction': predictions,
    'Actual': actuals
}, index=df.index[start_train_size:-1])

df_results['Correct'] = df_results['Prediction'] == df_results['Actual']
win_rate = df_results['Correct'].mean() * 100

print("\n--- SENTIMENT-ENHANCED BACKTEST RESULTS ---")
print(f"Total Days Simulated: {len(df_results)}")
print(f"Adjusted Strategy Win Rate: {win_rate:.2f}%")

# 6. Export results to a CSV spreadsheet report
report_filename = "trading_bot_report.csv"
df_results.to_csv(report_filename)
print(f"\n📁 Performance report successfully saved to your folder as '{report_filename}'!")