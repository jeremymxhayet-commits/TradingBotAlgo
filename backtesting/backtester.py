import logging
import time

import pandas as pd

from portfolio.risk_management import RiskManager
from execution.slippage_model import SlippageModel


logger = logging.getLogger(__name__)


class Backtester:
    def __init__(
        self,
        strategy,
        price_data,
        initial_cash=1_000_000,
        start_date=None,
        end_date=None,
        slippage_bps=5,
        max_drawdown_pct=0.10,
        max_leverage=2.0,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        trailing_stop_pct=0.0,
        commission_rate=0.0,
        allow_short=False,
        align_prices=True,
    ):
        self.strategy = strategy
        self.price_data = price_data
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None

        self.slippage_model = SlippageModel(base_spread_bps=slippage_bps)
        self.risk_manager = RiskManager(initial_cash, max_drawdown_pct, max_leverage)
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.commission_rate = commission_rate
        self.allow_short = allow_short
        self.align_prices = align_prices

        self.positions = {}
        self.position_costs = {}
        self.position_highs = {}
        self.position_lows = {}
        self.last_prices = {}
        self.trades = []
        self.equity_curve = []

    def _build_datetime_index(self):
        indices = []
        for df in self.price_data.values():
            if df is None or df.empty:
                continue
            idx = df.index
            if self.start_date is not None:
                idx = idx[idx >= self.start_date]
            if self.end_date is not None:
                idx = idx[idx <= self.end_date]
            indices.append(idx)

        if not indices:
            return []

        combined = sorted(set().union(*indices))
        return combined

    def _current_bar(self, df, current_time):
        if current_time not in df.index:
            return None
        bar = df.loc[current_time]
        if bar is None or bar.isna().all():
            return None
        return bar

    def _align_price_data(self, datetime_index):
        aligned = {}
        for symbol, df in self.price_data.items():
            if df is None or df.empty:
                continue
            df = df.reindex(datetime_index).ffill()
            aligned[symbol] = df
        self.price_data = aligned

    def _execute_orders(self, orders, current_data):
        executed_trades = []
        for order in orders:
            symbol = order["symbol"]
            side = order["side"].lower()
            qty = int(order.get("qty", 0))
            if qty <= 0:
                continue

            bar = current_data.get(symbol)
            if bar is None:
                continue

            price = bar.get("Open", bar.get("Close"))
            volume = bar.get("Volume", 0.0)
            fill_price = self.slippage_model.simulate_fill(price, qty, volume)
            cost = fill_price * qty
            commission = cost * self.commission_rate

            if side == "buy":
                total_cost = cost + commission
                if self.cash < total_cost:
                    continue
                self.cash -= total_cost
                prev_qty = self.positions.get(symbol, 0)
                prev_cost = self.position_costs.get(symbol, 0.0)
                if prev_qty < 0:
                    cover_qty = min(abs(prev_qty), qty)
                    new_qty = prev_qty + qty
                    if new_qty < 0:
                        abs_prev = abs(prev_qty)
                        abs_new = abs_prev - qty
                        avg_cost = prev_cost if abs_new > 0 else 0.0
                        self.positions[symbol] = new_qty
                        if abs_new > 0:
                            self.position_costs[symbol] = avg_cost
                            self.position_lows[symbol] = min(
                                self.position_lows.get(symbol, fill_price), fill_price
                            )
                        else:
                            self.position_costs.pop(symbol, None)
                            self.position_lows.pop(symbol, None)
                    else:
                        self.positions[symbol] = new_qty
                        self.position_costs.pop(symbol, None)
                        self.position_lows.pop(symbol, None)
                    self.strategy.update_position(symbol, qty, fill_price)
                    executed_trades.append(
                        {
                            "timestamp": bar.name,
                            "symbol": symbol,
                            "side": side,
                            "price": fill_price,
                            "quantity": qty,
                            "buy_price": fill_price,
                            "sell_price": prev_cost if prev_cost else None,
                            "pnl": (prev_cost - fill_price) * cover_qty if prev_cost else None,
                            "cash_after": self.cash,
                        }
                    )
                    continue
                new_qty = prev_qty + qty
                avg_cost = ((prev_cost * prev_qty) + (fill_price * qty)) / new_qty
                self.positions[symbol] = new_qty
                self.position_costs[symbol] = avg_cost
                self.position_highs[symbol] = max(self.position_highs.get(symbol, 0.0), fill_price)
                self.strategy.update_position(symbol, qty, fill_price)
            elif side == "sell":
                prev_qty = self.positions.get(symbol, 0)
                prev_cost = self.position_costs.get(symbol, 0.0)
                if prev_qty < qty and not self.allow_short:
                    continue
                if prev_qty > 0 and qty > prev_qty:
                    qty = prev_qty
                self.cash += max(cost - commission, 0.0)
                new_qty = prev_qty - qty
                self.positions[symbol] = new_qty
                if new_qty == 0:
                    self.position_costs.pop(symbol, None)
                    self.position_highs.pop(symbol, None)
                    self.position_lows.pop(symbol, None)
                elif new_qty < 0:
                    abs_prev = abs(prev_qty)
                    abs_new = abs(new_qty)
                    if prev_qty > 0:
                        avg_cost = fill_price
                    else:
                        avg_cost = ((prev_cost * abs_prev) + (fill_price * qty)) / abs_new
                    self.position_costs[symbol] = avg_cost
                    self.position_lows[symbol] = min(self.position_lows.get(symbol, fill_price), fill_price)
                self.strategy.update_position(symbol, -qty, fill_price)
            else:
                continue

            buy_price = None
            sell_price = None
            pnl = None
            if side == "buy":
                buy_price = fill_price
            elif side == "sell":
                buy_price = prev_cost if prev_cost else None
                sell_price = fill_price
                if prev_cost and prev_qty > 0:
                    pnl = (fill_price - prev_cost) * qty
                elif prev_cost and prev_qty < 0:
                    pnl = (prev_cost - fill_price) * qty

            executed_trades.append(
                {
                    "timestamp": bar.name,
                    "symbol": symbol,
                    "side": side,
                    "price": fill_price,
                    "quantity": qty,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "pnl": pnl,
                    "cash_after": self.cash,
                }
            )

        self.trades.extend(executed_trades)
        if executed_trades:
            self.strategy.cash = self.cash
            self.strategy.positions = dict(self.positions)
        return executed_trades

    def _mark_to_market(self, current_data, current_time):
        portfolio_value = self.cash
        for symbol, shares in self.positions.items():
            bar = current_data.get(symbol)
            if bar is not None:
                price = bar.get("Close", bar.get("Open"))
                if price is not None and not pd.isna(price):
                    self.last_prices[symbol] = float(price)
            price = self.last_prices.get(symbol)
            if price is None:
                continue
            portfolio_value += shares * price

        self.equity_curve.append({"timestamp": current_time, "equity": portfolio_value, "cash": self.cash})
        return portfolio_value

    def _stop_loss_orders(self, current_data):
        if (
            (not self.stop_loss_pct or self.stop_loss_pct <= 0)
            and (not self.take_profit_pct or self.take_profit_pct <= 0)
            and (not self.trailing_stop_pct or self.trailing_stop_pct <= 0)
        ):
            return []
        orders = []
        for symbol, qty in self.positions.items():
            avg_cost = self.position_costs.get(symbol)
            if not avg_cost:
                continue
            bar = current_data.get(symbol)
            if bar is None:
                continue
            price = bar.get("Close", bar.get("Open"))
            if qty > 0:
                if self.stop_loss_pct and price <= avg_cost * (1 - self.stop_loss_pct):
                    orders.append({"symbol": symbol, "side": "sell", "qty": qty, "order_type": "market"})
                    continue
                if self.take_profit_pct and price >= avg_cost * (1 + self.take_profit_pct):
                    orders.append({"symbol": symbol, "side": "sell", "qty": qty, "order_type": "market"})
                    continue
                if self.trailing_stop_pct:
                    peak = self.position_highs.get(symbol, price)
                    if price > peak:
                        self.position_highs[symbol] = price
                    elif price <= peak * (1 - self.trailing_stop_pct):
                        orders.append({"symbol": symbol, "side": "sell", "qty": qty, "order_type": "market"})
            elif qty < 0:
                abs_qty = abs(qty)
                if self.stop_loss_pct and price >= avg_cost * (1 + self.stop_loss_pct):
                    orders.append({"symbol": symbol, "side": "buy", "qty": abs_qty, "order_type": "market"})
                    continue
                if self.take_profit_pct and price <= avg_cost * (1 - self.take_profit_pct):
                    orders.append({"symbol": symbol, "side": "buy", "qty": abs_qty, "order_type": "market"})
                    continue
                if self.trailing_stop_pct:
                    trough = self.position_lows.get(symbol, price)
                    if price < trough:
                        self.position_lows[symbol] = price
                    elif price >= trough * (1 + self.trailing_stop_pct):
                        orders.append({"symbol": symbol, "side": "buy", "qty": abs_qty, "order_type": "market"})
        return orders

    def run(self, progress_interval_s=10):
        datetime_index = self._build_datetime_index()
        if not datetime_index:
            raise RuntimeError("No timestamps available for backtest.")
        if self.align_prices:
            self._align_price_data(datetime_index)

        total_steps = len(datetime_index)
        start_time = time.monotonic()
        last_progress = start_time

        for step, current_time in enumerate(datetime_index, start=1):
            current_data = {}
            data_slice = {}

            for symbol, df in self.price_data.items():
                if df is None or df.empty:
                    continue
                df_slice = df.loc[:current_time]
                if df_slice.empty:
                    continue
                bar = self._current_bar(df, current_time)
                if bar is None:
                    continue
                current_data[symbol] = bar
                data_slice[symbol] = df_slice

            if not current_data:
                continue

            stop_orders = self._stop_loss_orders(current_data)
            if stop_orders:
                self._execute_orders(stop_orders, current_data)

            signals = self.strategy.compute_signals(data_slice)
            orders = self.strategy.generate_orders(signals)
            self._execute_orders(orders, current_data)
            portfolio_value = self._mark_to_market(current_data, current_time)
            self.strategy.on_bar(data_slice, current_data, current_time)

            if not self.risk_manager.check_drawdown(portfolio_value):
                logger.warning("Max drawdown exceeded. Halting backtest.")
                break

            now = time.monotonic()
            if progress_interval_s and (now - last_progress) >= progress_interval_s:
                elapsed = int(now - start_time)
                minutes, seconds = divmod(elapsed, 60)
                hours, minutes = divmod(minutes, 60)
                logger.info(
                    "Backtest progress: %.1f%% | elapsed %02d:%02d:%02d | current=%s",
                    (step / total_steps) * 100.0,
                    hours,
                    minutes,
                    seconds,
                    current_time,
                )
                last_progress = now

        equity_df = pd.DataFrame(self.equity_curve)
        if equity_df.empty:
            raise RuntimeError("Equity curve is empty. No backtest results produced.")
        equity_df.set_index("timestamp", inplace=True)

        trades_df = pd.DataFrame(self.trades)
        if trades_df.empty:
            trades_df = pd.DataFrame(
                columns=["timestamp", "symbol", "side", "price", "quantity", "buy_price", "sell_price", "pnl"]
            )

        return {"equity_curve": equity_df, "trades": trades_df}


if __name__ == "__main__":
    from strategies.statistical_arbitrage import StatisticalArbitrageStrategy

    strategy = StatisticalArbitrageStrategy([])
    bt = Backtester(strategy, price_data={}, initial_cash=100000)
    results = bt.run()
    results["equity_curve"].to_csv("backtesting/results/equity_curve_stat.csv")
    results["trades"].to_csv("backtesting/results/trades_stat.csv")
