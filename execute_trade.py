import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

def get_trading_client():
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    return TradingClient(api_key, secret_key, paper=True)

def place_paper_order_with_brackets(symbol: str, qty: float = 1, side: str = "BUY", current_price: float = 0.0, tp_pct: float = 2.0, sl_pct: float = 1.0):
    trading_client = get_trading_client()
    order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
    
    # Pehle Bracket Order try karein agar price mojood hai
    if current_price > 0:
        try:
            if order_side == OrderSide.BUY:
                tp_price = round(current_price * (1 + tp_pct / 100), 2)
                sl_price = round(current_price * (1 - sl_pct / 100), 2)
            else:
                tp_price = round(current_price * (1 - tp_pct / 100), 2)
                sl_price = round(current_price * (1 + sl_pct / 100), 2)
                
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=tp_price),
                stop_loss=StopLossRequest(stop_price=sl_price)
            )
            return trading_client.submit_order(order_data=order_data)
        except Exception:
            # Agar bracket order fail ho jaye (jaise position conflict ki wajah se), toh standard order par fallback karein
            pass

    # Fallback: Standard Market Order (Without Brackets)
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        time_in_force=TimeInForce.DAY
    )
    submitted_order = trading_client.submit_order(order_data=order_data)
    return submitted_order

def get_account_info():
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