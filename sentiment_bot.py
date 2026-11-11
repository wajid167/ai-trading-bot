import yfinance as yf
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

print("Analyzing market sentiment...")

ticker = "AAPL"
stock = yf.Ticker(ticker)

# 1. Fetch recent news headlines
try:
    news_list = stock.news
except Exception:
    news_list = []

compound_scores = []

if news_list:
    print(f"\n--- Recent News Headlines for {ticker} ---")
    for item in news_list[:5]:
        title = item.get('title') or item.get('content', {}).get('title', 'No Title')
        score = sia.polarity_scores(title)['compound']
        compound_scores.append(score)
        print(f"• {title} (Sentiment Score: {score:.2f})")
    
    avg_sentiment = sum(compound_scores) / len(compound_scores) if compound_scores else 0.0
else:
    avg_sentiment = 0.0
    print("No recent news found, defaulting sentiment to neutral.")

print(f"\nAverage News Sentiment Score: {avg_sentiment:.2f} (-1 = Very Bearish, +1 = Very Bullish)")

print("\nDownloading historical data...")
df = yf.download(ticker, start="2023-01-01")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# Check if data download was successful
if df.empty or len(df) == 0:
    print("Error: Could not retrieve price data from Yahoo Finance. Please check your internet connection or try again later.")
else:
    # 2. Feature Engineering
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

    if len(X) > 0:
        model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X, y)

        # 3. Predict tomorrow's direction
        latest_data = X.iloc[[-1]]
        prediction = model.predict(latest_data)[0]
        probability = model.predict_proba(latest_data)

        print(f"\n--- COMBINED AI TRADING SIGNAL ---")
        tech_signal = "🟢 BULLISH (Up)" if prediction == 1 else "🔴 BEARISH (Down)"
        print(f"Technical Model Signal: {tech_signal}")
        print(f"Confidence (Down / Up): {probability[0][0]*100:.2f}% / {probability[0][1]*100:.2f}%")
    else:
        print("Not enough data points after cleaning.")