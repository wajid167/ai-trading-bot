import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from execute_trade import place_paper_order

st.markdown("# 📈 AI Stock & Crypto Trading Bot Dashboard")
st.write("Select an asset below to train the AI model and predict market movement live!")

assets = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Microsoft (MSFT)": "MSFT",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Solana (SOL-USD)": "SOL-USD"
}

selected_label = st.selectbox("Choose Asset", list(assets.keys()))
selected_ticker = assets[selected_label]

start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))

if st.button("Run AI Prediction"):
    with st.spinner(f"Fetching data for {selected_label}..."):
        df = yf.download(selected_ticker, start=start_date)
        
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            latest_price = float(df['Close'].iloc[-1])
            st.metric(label=f"Latest Close Price ({selected_ticker})", value=f"${latest_price:,.2f}")
            
            fig = px.line(df, x=df.index, y='Close', title=f"{selected_label} Historical Price Trend")
            fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark")
            st.plotly_chart(fig, width='stretch')
            
            # Save state so trade section stays open
            st.session_state['run_success'] = True
            st.session_state['latest_price'] = latest_price
            st.session_state['selected_ticker'] = selected_ticker
        else:
            st.error("Could not fetch data for the selected date range.")

# Always show execution panel if prediction has run successfully in session
if st.session_state.get('run_success', False):
    st.markdown("---")
    st.subheader("⚡ Automated Paper Trade Execution")
    
    col1, col2 = st.columns(2)
    with col1:
        trade_qty = st.number_input("Order Quantity", min_value=0.01, value=1.0, step=0.1)
    with col2:
        action = st.selectbox("Action", ["BUY", "SELL"])
        
    if st.button(f"Execute Paper {action} Order"):
        try:
            order = place_paper_order(st.session_state['selected_ticker'], qty=trade_qty, side=action)
            st.success(f"Successfully placed Paper {action} order for {trade_qty} unit(s) of {st.session_state['selected_ticker']}! Order ID: {order.id}")
        except Exception as e:
            st.error(f"Execution Error: {e}")