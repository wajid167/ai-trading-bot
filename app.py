import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px

st.markdown("# 📈 AI Stock & Crypto Trading Bot Dashboard")
st.write("Select an asset below to train the AI model and predict tomorrow's market movement live!")

# Expanded dictionary of stocks and cryptocurrencies
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
    with st.spinner(f"Fetching data and analyzing market movement for {selected_label}..."):
        # Fetch data using yfinance
        df = yf.download(selected_ticker, start=start_date)
        
        if not df.empty:
            st.success(f"Analysis complete for {selected_label}!")
            
            # Handle MultiIndex columns if returned by yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # Show latest metric
            latest_price = float(df['Close'].iloc[-1])
            st.metric(label=f"Latest Close Price ({selected_ticker})", value=f"${latest_price:,.2f}")
            
            # Display interactive Plotly chart
            fig = px.line(df, x=df.index, y='Close', title=f"{selected_label} Historical Price Trend")
            fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Could not fetch data for the selected date range. Try an earlier start date.")