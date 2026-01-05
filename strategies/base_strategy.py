from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    Provides a standard interface and shared utility methods for strategy logic.
    """

    def __init__(self):
        self.positions: Dict[str, int] = {}
        self.cash: float = 1_000_000.0
        self.portfolio_value: float = self.cash
        self.trade_log: List[Dict[str, Any]] = []

    @abstractmethod
    def compute_signals(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Compute trading signals from input market data.
        """

    @abstractmethod
    def generate_orders(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate order instructions from signals.
        """

    def on_bar(self, data: Dict[str, pd.DataFrame], current_data: Dict[str, pd.Series], current_time: pd.Timestamp):
        """
        Optional hook for strategies to update internal state after each bar.
        """

    def update_position(self, symbol: str, qty: int, price: float):
        current_qty = self.positions.get(symbol, 0)
        self.positions[symbol] = current_qty + qty
        self.cash -= qty * price
        self._log_trade(symbol, qty, price)
        self._update_portfolio_value(price_feed={symbol: price})

    def _log_trade(self, symbol: str, qty: int, price: float):
        self.trade_log.append(
            {
                "symbol": symbol,
                "qty": qty,
                "price": price,
                "value": qty * price,
            }
        )

    def _update_portfolio_value(self, price_feed: Dict[str, float]):
        total_value = self.cash
        for symbol, qty in self.positions.items():
            price = price_feed.get(symbol)
            if price is not None:
                total_value += qty * price
        self.portfolio_value = total_value

    def get_position(self, symbol: str) -> int:
        return self.positions.get(symbol, 0)

    def get_cash(self) -> float:
        return self.cash

    def get_portfolio_value(self) -> float:
        return self.portfolio_value

    def get_trade_log(self) -> List[Dict[str, Any]]:
        return self.trade_log
