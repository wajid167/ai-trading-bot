import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from execute_trade import place_paper_order, get_account_info, get_open_positions
import time

st.set_page_config(page_title="AI Live ML Trading Dashboard", layout="wide")

st.markdown("# 📈 Real-Time AI & ML Stock & Crypto Trading Dashboard")
st.write("Live market monitoring driven by real-time Scikit-Learn Machine Learning predictions & Alpaca portfolio tracker.")

# Multi-asset dictionary
assets = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Microsoft (MSFT)": "MSFT",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Solana (SOL-USD)": "SOL-USD"
}

# Sidebar control panel
st.sidebar.header("⚙️ Machine Learning & Stream Settings")
selected_label = st.sidebar.selectbox("Choose Asset", list(assets.keys()))
selected_ticker = assets[selected_label]

auto_refresh = st.sidebar.checkbox("Enable Live Refresh (15s)", value=True)
refresh_interval = 15

# Sidebar Alpaca Account Summary Tracker
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Alpaca Paper Account")
account_data = get_account_info()

if account_data:
    st.sidebar.metric("Portfolio Value", f"${account_data['portfolio_value']:,.2f}")
    st.sidebar.metric("Available Cash", f"${account_data['cash']:,.2f}")
else:
    st.sidebar.warning("Check Alpaca API keys in secrets.")

# Fetch real-time intraday price data
def get_live_data(ticker):
    df = yf.download(ticker, period="5d", interval="5m")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# Machine Learning prediction engine
def run_ml_prediction(df):
    data = df.copy()
    
    # Feature Engineering
    data['Returns'] = data['Close'].pct_change()
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_15'] = data['Close'].rolling(window=15).mean()
    data['Volatility'] = data['Returns'].rolling(window=5).std()
    
    # Target variable: 1 if next period price is higher, else 0
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    data.dropna(inplace=True)
    
    if len(data) < 20:
        return "NEUTRAL 🟡", 0.50
    
    features = ['Returns', 'SMA_5', 'SMA_15', 'Volatility']
    X = data[features]
    y = data['Target']
    
    # Train Random Forest Classifier on historical interval data
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X[:-1], y[:-1])
    
    # Predict direction for the latest live bar
    latest_features = X.iloc[[-1]]
    prediction = model.predict(latest_features)[0]
    probabilities = model.predict_proba(latest_features)[0]
    
    confidence = probabilities[prediction]
    
    if prediction == 1:
        return "BUY 🟢", confidence
    else:
        return "SELL 🔴", confidence

# Main execution loop
df_live = get_live_data(selected_ticker)

if not df_live.empty:
    latest_price = float(df_live['Close'].iloc[-1])
    ml_signal, confidence = run_ml_prediction(df_live)
    
    # Top Metrics Banner
    col1, col2, col3 = st.columns(3)
    col1.metric("Live Market Price", f"${latest_price:,.2f}")
    col2.metric("ML Direction Signal", ml_signal)
    col3.metric("Model Confidence", f"{confidence * 100:.1f}%")
    
    # Interactive Plotly Chart
    fig = px.line(df_live, x=df_live.index, y='Close', title=f"{selected_label} 5-Minute Intraday Stream")
    fig.update_layout(xaxis_title="Time", yaxis_title="Price (USD)", template="plotly_dark")
    st.plotly_chart(fig, width='stretch')

    # Trade Execution Panel
    st.markdown("---")
    st.subheader("⚡ Automated Paper Trade Execution")
    
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        trade_qty = st.number_input("Order Quantity", min_value=0.01, value=1.0, step=0.1)
    with t_col2:
        recommended_action = "BUY" if "BUY" in ml_signal else "SELL"
        action = st.selectbox("Action", ["BUY", "SELL"], index=0 if recommended_action == "BUY" else 1)
        
    if st.button(f"Execute Paper {action} Order"):
        try:
            order = place_paper_order(selected_ticker, qty=trade_qty, side=action)
            st.success(f"Placed Paper {action} order for {trade_qty} unit(s) of {selected_ticker}! Order ID: {order.id}")
        except Exception as e:
            st.error(f"Execution Error: {e}")
            
    # Active Portfolio Positions Section
    st.markdown("---")
    st.subheader("📊 Active Paper Portfolio Positions")
    positions = get_open_positions()

    if positions:
        df_pos = pd.DataFrame(positions)
        st.dataframe(df_pos, width='stretch')
    else:
        st.info("No active open positions currently in your paper sandbox.")

else:
    st.error("Market data feed currently unavailable.")

# Auto-refresh mechanism
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()