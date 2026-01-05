import time
import logging
from datetime import datetime, timedelta

class ExecutionAlgorithms:
    def __init__(self, order_manager, market_data_interface):
        """
        :param order_manager: Instance of OrderManager class
        :param market_data_interface: Object that provides real-time or historical price/volume data
        """
        self.om = order_manager
        self.data = market_data_interface

    def twap(self, symbol, qty, side, duration_minutes, interval_seconds=60):
        """
        Time-Weighted Average Price execution
        Executes evenly over time.
        """
        slices = int(duration_minutes * 60 / interval_seconds)
        slice_qty = qty // slices
        for i in range(slices):
            logging.info(f"TWAP slice {i+1}/{slices}: {slice_qty} {symbol}")
            self.om.send_order(symbol, slice_qty, side, order_type="market")
            time.sleep(interval_seconds)

    def vwap(self, symbol, qty, side, duration_minutes, interval_seconds=60):
        """
        Volume-Weighted Average Price execution
        Uses historical volume distribution to determine order sizing.
        """
        hist_volume_profile = self.data.get_intraday_volume_profile(symbol, duration_minutes)
        total_hist_volume = sum(hist_volume_profile)
        executed_qty = 0

        for i, vol in enumerate(hist_volume_profile):
            pct = vol / total_hist_volume
            slice_qty = int(qty * pct)
            if slice_qty == 0:
                continue
            logging.info(f"VWAP slice {i+1}: {slice_qty} {symbol} ({pct:.2%} of total)")
            self.om.send_order(symbol, slice_qty, side, order_type="market")
            executed_qty += slice_qty
            time.sleep(interval_seconds)

        if executed_qty < qty:
            remaining = qty - executed_qty
            logging.info(f"VWAP final adjustment: {remaining} {symbol}")
            self.om.send_order(symbol, remaining, side, order_type="market")

    def pov(self, symbol, qty, side, participation_rate=0.1, duration_minutes=10, interval_seconds=60):
        """
        Participation-of-Volume execution
        Executes a % of live market volume each interval.
        """
        executed = 0
        end_time = datetime.utcnow() + timedelta(minutes=duration_minutes)

        while datetime.utcnow() < end_time and executed < qty:
            market_volume = self.data.get_recent_volume(symbol, interval_seconds)
            slice_qty = int(market_volume * participation_rate)
            slice_qty = min(slice_qty, qty - executed)

            if slice_qty > 0:
                logging.info(f"POV execution: {slice_qty} {symbol} ({participation_rate:.0%} of {market_volume})")
                self.om.send_order(symbol, slice_qty, side, order_type="market")
                executed += slice_qty
            time.sleep(interval_seconds)

    def implementation_shortfall(self, symbol, qty, side, benchmark_price, max_slippage=0.005, duration_minutes=10):
        """
        Implementation Shortfall execution
        Speeds up execution if price moves against, slows if in favor.
        """
        remaining = qty
        interval_seconds = 60
        end_time = datetime.utcnow() + timedelta(minutes=duration_minutes)

        while datetime.utcnow() < end_time and remaining > 0:
            current_price = self.data.get_latest_price(symbol)
            price_diff = (current_price - benchmark_price) / benchmark_price
            
            if (side == "buy" and price_diff > max_slippage) or (side == "sell" and price_diff < -max_slippage):
                logging.warning(f"Price moved too far from benchmark: skipping execution this round.")
                time.sleep(interval_seconds)
                continue

            slice_qty = max(1, int(remaining * (0.25 if abs(price_diff) < 0.001 else 0.5)))
            logging.info(f"IS execution: {slice_qty} {symbol} at {current_price} (vs {benchmark_price})")
            self.om.send_order(symbol, slice_qty, side, order_type="market")
            remaining -= slice_qty
            time.sleep(interval_seconds)

    def iceberg(self, symbol, total_qty, side, visible_size, price):
        """
        Iceberg Order execution
        Sends limit orders in chunks, only visible_size is visible to market.
        """
        remaining = total_qty
        while remaining > 0:
            slice_qty = min(visible_size, remaining)
            logging.info(f"Iceberg execution: showing {slice_qty} of {remaining} {symbol} at {price}")
            self.om.send_order(symbol, slice_qty, side, order_type="limit", price=price)
            remaining -= slice_qty
            time.sleep(10) 

    def rl_execution(self, rl_agent, env):
        """
        Reinforcement Learning-based execution policy
        Executes based on RL agent's learned policy in a simulated/live environment.
        """
        state = env.reset()
        done = False
        while not done:
            action = rl_agent.predict(state)
            state, reward, done, info = env.step(action)
            logging.info(f"RL Agent executed action: {action}, reward: {reward}")
