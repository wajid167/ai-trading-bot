import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

def get_trading_client():
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    return TradingClient(api_key, secret_key, paper=True)

def place_paper_order(symbol: str, qty: float = 1, side: str = "BUY"):
    trading_client = get_trading_client()
    order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
    
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        time_in_force=TimeInForce.DAY
    )
    submitted_order = trading_client.submit_order(order_data=order_data)
    return submitted_order

def get_account_info():
    """Fetches paper trading account cash and portfolio balance."""
    try:
        client = get_trading_client()
        account = client.get_account()
        return {
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
            "status": account.status
        }
    except Exception as e:
        return None

def get_open_positions():
    """Fetches currently open positions in the paper account."""
    try:
        client = get_trading_client()
        positions = client.get_all_positions()
        pos_list = []
        for p in positions:
            pos_list.append({
                "Symbol": p.symbol,
                "Qty": p.qty,
                "Market Value": float(p.market_value),
                "Unrealized P/L": float(p.unrealized_pl)
            })
        return pos_list
    except Exception as e:
        return []