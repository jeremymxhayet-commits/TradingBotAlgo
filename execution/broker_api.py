import alpaca_trade_api as tradeapi
import logging

class BrokerAPI:
    def __init__(self, api_key, api_secret, base_url):
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        logging.info("BrokerAPI initialized.")

    def get_account_balance(self):
        try:
            account = self.api.get_account()
            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "status": account.status
            }
        except Exception as e:
            logging.error(f"Failed to fetch account balance: {e}")
            return {}

    def get_live_quote(self, symbol):
        try:
            quote = self.api.get_last_quote(symbol)
            return {
                "ask_price": float(quote.askprice),
                "bid_price": float(quote.bidprice),
                "ask_size": int(quote.asksize),
                "bid_size": int(quote.bidsize),
                "timestamp": str(quote.timestamp)
            }
        except Exception as e:
            logging.error(f"Failed to get quote for {symbol}: {e}")
            return {}

    def get_latest_trade(self, symbol):
        try:
            trade = self.api.get_last_trade(symbol)
            return {
                "price": float(trade.price),
                "size": int(trade.size),
                "timestamp": str(trade.timestamp)
            }
        except Exception as e:
            logging.error(f"Failed to get trade for {symbol}: {e}")
            return {}

    def submit_order(self, symbol, qty, side, order_type="market", time_in_force="gtc", price=None):
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force,
                limit_price=price if order_type == "limit" else None
            )
            logging.info(f"Order submitted: {order.id} - {side} {qty} {symbol} ({order_type})")
            return order
        except Exception as e:
            logging.error(f"Failed to submit order: {e}")
            return None

    def cancel_order(self, order_id):
        try:
            self.api.cancel_order(order_id)
            logging.info(f"Order {order_id} cancelled.")
        except Exception as e:
            logging.error(f"Failed to cancel order {order_id}: {e}")

    def cancel_all_orders(self):
        try:
            self.api.cancel_all_orders()
            logging.info("All open orders cancelled.")
        except Exception as e:
            logging.error(f"Failed to cancel all orders: {e}")

    def get_order_status(self, order_id):
        try:
            order = self.api.get_order(order_id)
            return order.status
        except Exception as e:
            logging.error(f"Failed to fetch order status: {e}")
            return None

    def list_open_orders(self):
        try:
            return self.api.list_orders(status="open")
        except Exception as e:
            logging.error(f"Failed to list open orders: {e}")
            return []
