import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

def place_paper_order(symbol: str, qty: float = 1, side: str = "BUY"):
    """
    Submits a simulated paper trading order to Alpaca.
    Retrieves API keys safely from environment variables.
    """
    api_key = os.getenv("ALPACA_API_KEY", "YOUR_PAPER_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "YOUR_PAPER_SECRET_KEY")
    
    # Initialize trading client in paper mode
    trading_client = TradingClient(api_key, secret_key, paper=True)
    
    order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
    
    # Construct market order
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        time_in_force=TimeInForce.DAY
    )
    
    # Submit order to sandbox
    submitted_order = trading_client.submit_order(order_data=order_data)
    return submitted_order