import alpaca_trade_api as tradeapi
import logging
import time

class OrderManager:
    def __init__(self, config):
        """
        Initializes the Order Manager with API credentials and account mode.
        :param config: Dict or config object with keys: API_KEY, API_SECRET, BASE_URL, PAPER_TRADING
        """
        self.api_key = config["API_KEY"]
        self.api_secret = config["API_SECRET"]
        self.base_url = config["BASE_URL"]
        self.paper_trading = config.get("PAPER_TRADING", True)
        self.account = None
        self.api = tradeapi.REST(self.api_key, self.api_secret, self.base_url, api_version='v2')
        self.open_orders = {}  
        self.order_log = []   

        self._initialize_account()
        logging.info(f"OrderManager initialized. Paper trading: {self.paper_trading}")

    def _initialize_account(self):
        try:
            self.account = self.api.get_account()
            logging.info(f"Connected to Alpaca account: {self.account.id}")
        except Exception as e:
            logging.error(f"Failed to connect to Alpaca API: {e}")
            raise

    def get_account_summary(self):
        """Returns basic account info: equity, buying power, status"""
        try:
            self.account = self.api.get_account()
            return {
                "equity": float(self.account.equity),
                "buying_power": float(self.account.buying_power),
                "status": self.account.status
            }
        except Exception as e:
            logging.error(f"Failed to fetch account summary: {e}")
            return {}

    def send_order(self, symbol, qty, side, order_type="market", price=None, time_in_force="gtc"):
        """
        Places a new order with basic safety and duplication checks.
        :param symbol: Ticker symbol (e.g., 'AAPL')
        :param qty: Quantity of shares/contracts
        :param side: 'buy' or 'sell'
        :param order_type: 'market' or 'limit'
        :param price: Required for limit orders
        :param time_in_force: 'gtc', 'day', etc.
        :return: Order object or None
        """
        symbol = symbol.upper()
        side = side.lower()


        if side not in ["buy", "sell"]:
            logging.warning(f"Invalid side: {side}")
            return None
        if order_type == "limit" and price is None:
            logging.warning("Limit order requires a price.")
            return None

        if symbol in self.open_orders:
            logging.info(f"Open order already exists for {symbol}, skipping.")
            return None


        try:
            buying_power = float(self.api.get_account().buying_power)
            if side == "buy" and order_type == "market":
                last_quote = self.api.get_last_quote(symbol)
                estimated_cost = qty * float(last_quote.askprice)
                if estimated_cost > buying_power:
                    logging.warning(f"Insufficient buying power for {symbol}: needed {estimated_cost}, available {buying_power}")
                    return None
        except Exception as e:
            logging.error(f"Error checking buying power or quote for {symbol}: {e}")
            return None

        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force,
                limit_price=price if order_type == "limit" else None
            )
            self.open_orders[symbol] = order.id
            self._log_order(symbol, qty, side, order_type, price, order.id)
            logging.info(f"Order submitted: {side.upper()} {qty} {symbol} ({order_type})")
            return order
        except Exception as e:
            logging.error(f"Failed to send order for {symbol}: {e}")
            return None

    def cancel_order(self, symbol):
        """Cancels the open order for a given symbol"""
        order_id = self.open_orders.get(symbol.upper())
        if not order_id:
            logging.info(f"No open order to cancel for {symbol}")
            return

        try:
            self.api.cancel_order(order_id)
            logging.info(f"Order cancelled for {symbol} (ID: {order_id})")
            del self.open_orders[symbol.upper()]
        except Exception as e:
            logging.error(f"Error cancelling order {order_id} for {symbol}: {e}")

    def cancel_all_orders(self):
        """Cancels all open orders"""
        try:
            self.api.cancel_all_orders()
            self.open_orders.clear()
            logging.info("All open orders cancelled.")
        except Exception as e:
            logging.error(f"Error cancelling all orders: {e}")

    def _log_order(self, symbol, qty, side, order_type, price, order_id):
        """Logs an order for audit trail"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.order_log.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "order_type": order_type,
            "price": price,
            "order_id": order_id
        })

    def get_order_status(self, order_id):
        """Returns status of a given order"""
        try:
            order = self.api.get_order(order_id)
            return order.status
        except Exception as e:
            logging.error(f"Failed to fetch order status for {order_id}: {e}")
            return None

    def list_open_orders(self):
        """Returns a list of currently open orders"""
        try:
            return self.api.list_orders(status="open")
        except Exception as e:
            logging.error(f"Failed to fetch open orders: {e}")
            return []

    def get_order_log(self):
        """Returns the internal log of all submitted orders"""
        return self.order_log
