import streamlit as st

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
    st.info(f"Running AI prediction model for {selected_label} ({selected_ticker}) starting from {start_date}...")
    # Your prediction pipeline call goes here