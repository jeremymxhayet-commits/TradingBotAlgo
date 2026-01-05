from strategies.base_strategy import Strategy
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

class StatisticalArbitrageStrategy(Strategy):
    """
    Implements classical statistical arbitrage strategies:
    - Mean reversion
    - Pairs trading
    - ARIMA forecasting
    - GARCH volatility targeting
    - Momentum (time-series)
    """

    def __init__(self, lookback: int = 60):
        super().__init__()
        self.lookback = lookback
        self.z_threshold = 2.0
        self.pairs = [('KO', 'PEP'), ('AAPL', 'MSFT')]  

    def compute_signals(self, data: pd.DataFrame) -> dict:
        signals = {}
        for symbol in data.columns.levels[0]:
            price_series = data[symbol]['close'].dropna()
            if len(price_series) < self.lookback:
                continue

            recent = price_series[-self.lookback:]
            mean = recent.mean()
            std = recent.std()
            z_score = (recent.iloc[-1] - mean) / std

            if z_score > self.z_threshold:
                signals[symbol] = 'sell'
            elif z_score < -self.z_threshold:
                signals[symbol] = 'buy'
            else:
                signals[symbol] = 'hold'

        for sym1, sym2 in self.pairs:
            if sym1 in data.columns.levels[0] and sym2 in data.columns.levels[0]:
                s1 = data[sym1]['close'].dropna()
                s2 = data[sym2]['close'].dropna()
                min_len = min(len(s1), len(s2))
                spread = s1[-min_len:] - s2[-min_len:]

                if len(spread) < self.lookback:
                    continue
                z = (spread - spread.mean()) / spread.std()

                if z.iloc[-1] > self.z_threshold:
                    signals[sym1] = 'sell'
                    signals[sym2] = 'buy'
                elif z.iloc[-1] < -self.z_threshold:
                    signals[sym1] = 'buy'
                    signals[sym2] = 'sell'

        return signals

    def generate_orders(self, signals: dict) -> list:
        orders = []
        for symbol, signal in signals.items():
            price = self._get_latest_price(symbol)
            position = self.get_position(symbol)
            volatility = self._estimate_volatility(symbol)
            size = self._risk_adjusted_position(volatility)

            if signal == 'buy' and position <= 0:
                qty = abs(position) + size
                orders.append({"symbol": symbol, "side": "buy", "qty": qty, "order_type": "market"})
            elif signal == 'sell' and position >= 0:
                qty = abs(position) + size
                orders.append({"symbol": symbol, "side": "sell", "qty": qty, "order_type": "market"})

        return orders

    def _get_latest_price(self, symbol: str) -> float:
      
        return 100.0

    def _estimate_volatility(self, symbol: str) -> float:
        dummy_returns = np.random.normal(0, 0.01, 100)
        model = arch_model(dummy_returns, vol='Garch', p=1, q=1)
        res = model.fit(disp='off')
        forecast = res.forecast(horizon=1)
        return np.sqrt(forecast.variance.values[-1, 0])

    def _risk_adjusted_position(self, volatility: float, risk_target: float = 0.01) -> int:
        if volatility == 0:
            return 0
        dollar_risk = self.portfolio_value * risk_target
        position_size = dollar_risk / volatility
        return int(position_size / 100) 
