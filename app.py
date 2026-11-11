import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from execute_trade import place_paper_order
import time

st.set_page_config(page_title="AI Live Trading Dashboard", layout="wide")

st.markdown("# 📈 Real-Time AI Stock & Crypto Trading Dashboard")
st.write("Live market monitoring with automated AI trade signal predictions.")

# Multi-asset configuration
assets = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Microsoft (MSFT)": "MSFT",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Solana (SOL-USD)": "SOL-USD"
}

# Sidebar settings
st.sidebar.header("⚙️ Live Settings")
selected_label = st.sidebar.selectbox("Choose Asset", list(assets.keys()))
selected_ticker = assets[selected_label]

auto_refresh = st.sidebar.checkbox("Enable Live Auto-Refresh (15s)", value=True)
refresh_interval = 15  # seconds

# Helper function to fetch real-time intra-day market data
def get_live_data(ticker):
    # Fetches recent 1-minute intraday data
    df = yf.download(ticker, period="1d", interval="1m")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# Simple AI signal logic based on short-term Moving Average Crossover
def generate_ai_signal(df):
    if len(df) < 5:
        return "NEUTRAL", 0.0
    
    df['SMA_3'] = df['Close'].rolling(window=3).mean()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    
    last_close = float(df['Close'].iloc[-1])
    sma3 = float(df['SMA_3'].iloc[-1])
    sma5 = float(df['SMA_5'].iloc[-1])
    
    if sma3 > sma5:
        return "BUY 🟢", last_close
    elif sma3 < sma5:
        return "SELL 🔴", last_close
    else:
        return "HOLD 🟡", last_close

# Main layout
df_live = get_live_data(selected_ticker)

if not df_live.empty:
    latest_price = float(df_live['Close'].iloc[-1])
    signal, price_target = generate_ai_signal(df_live)
    
    # Live metrics layout
    col1, col2, col3 = st.columns(3)
    col1.metric("Live Price", f"${latest_price:,.2f}")
    col2.metric("AI Market Signal", signal)
    col3.metric("Asset Status", "Active Stream 🟢")
    
    # Real-time Chart
    fig = px.line(df_live, x=df_live.index, y='Close', title=f"{selected_label} Live 1-Minute Price Stream")
    fig.update_layout(xaxis_title="Time", yaxis_title="Price (USD)", template="plotly_dark")
    st.plotly_chart(fig, width='stretch')

    # Trade execution panel
    st.markdown("---")
    st.subheader("⚡ Automated Paper Trade Execution")
    
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        trade_qty = st.number_input("Order Quantity", min_value=0.01, value=1.0, step=0.1)
    with t_col2:
        # Default action auto-selects based on AI signal recommendation
        default_action = "BUY" if "BUY" in signal else ("SELL" if "SELL" in signal else "BUY")
        action = st.selectbox("Action", ["BUY", "SELL"], index=0 if default_action == "BUY" else 1)
        
    if st.button(f"Execute Live Paper {action} Order"):
        try:
            order = place_paper_order(selected_ticker, qty=trade_qty, side=action)
            st.success(f"Placed Paper {action} order for {trade_qty} unit(s) of {selected_ticker}! Order ID: {order.id}")
        except Exception as e:
            st.error(f"Execution Error: {e}")

else:
    st.error("Market data stream currently offline or ticker unavailable.")

# Auto-refresh logic for real-time streaming
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()