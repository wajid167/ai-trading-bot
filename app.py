import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier

st.title("📈 AI Stock & Crypto Trading Bot Dashboard")
st.write("Select an asset below to train the AI model and predict tomorrow's market movement live!")

# Sidebar user inputs
ticker = st.selectbox("Choose Asset", ["AAPL", "TSLA", "BTC-USD", "ETH-USD", "MSFT"])
start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))

if st.button("Run AI Prediction"):
    with st.spinner(f"Downloading data and training model for {ticker}..."):
        # 1. Download data
        df = yf.download(ticker, start=start_date)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # 2. Features
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

        # 3. Train Model
        model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X, y)

        # 4. Predict
        latest_data = X.iloc[[-1]]
        prediction = model.predict(latest_data)[0]
        probability = model.predict_proba(latest_data)

        st.subheader("Results:")
        if prediction == 1:
            st.success("🟢 BULLISH (Expected to go UP tomorrow)")
        else:
            st.error("🔴 BEARISH (Expected to go DOWN tomorrow)")

        st.write(f"**Confidence Down:** {probability[0][0]*100:.2f}%")
        st.write(f"**Confidence Up:** {probability[0][1]*100:.2f}%")

        # Show price chart
        st.line_chart(df['Close'])