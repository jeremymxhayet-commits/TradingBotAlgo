import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveFeatureGenerator:
    def __init__(self, window=20):
        self.window = window
        self.price_buffers = {}
        self.volume_buffers = {}
        self.rsi_state = {}

    def _init_symbol(self, symbol):
        self.price_buffers[symbol] = deque(maxlen=self.window)
        self.volume_buffers[symbol] = deque(maxlen=self.window)
        self.rsi_state[symbol] = {
            'gain': deque(maxlen=14),
            'loss': deque(maxlen=14),
            'prev_close': None
        }

    def _calculate_rsi(self, symbol, close):
        state = self.rsi_state[symbol]
        prev = state['prev_close']
        if prev is None:
            state['prev_close'] = close
            return None

        delta = close - prev
        gain = max(delta, 0)
        loss = -min(delta, 0)

        state['gain'].append(gain)
        state['loss'].append(loss)
        state['prev_close'] = close

        if len(state['gain']) < 14:
            return None

        avg_gain = np.mean(state['gain'])
        avg_loss = np.mean(state['loss'])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def update(self, bar: dict) -> dict:
        """
        Accepts a single OHLCV bar dict:
        {
            'symbol': 'AAPL',
            'timestamp': '2025-12-26T14:30:00Z',
            'open': float,
            'high': float,
            'low': float,
            'close': float,
            'volume': float
        }
        Returns a dict of computed features, or None if not enough data yet.
        """
        symbol = bar['symbol']
        ts = pd.to_datetime(bar['timestamp'])
        close = bar['close']
        volume = bar['volume']

        if symbol not in self.price_buffers:
            self._init_symbol(symbol)

        self.price_buffers[symbol].append(close)
        self.volume_buffers[symbol].append(volume)

        if len(self.price_buffers[symbol]) < self.window:
            return None

        prices = np.array(self.price_buffers[symbol])


        sma = np.mean(prices)
        ema = pd.Series(prices).ewm(span=self.window, adjust=False).mean().iloc[-1]
        std = np.std(prices)
        zscore = (close - sma) / std if std > 0 else 0

        momentum = (close / prices[0]) - 1
        ret_1 = (close / prices[-2]) - 1 if len(prices) >= 2 else 0

        rsi = self._calculate_rsi(symbol, close)

        features = {
            'symbol': symbol,
            'timestamp': ts,
            'sma': sma,
            'ema': ema,
            'zscore': zscore,
            'volatility': std,
            'momentum': momentum,
            'return_1': ret_1,
            'rsi': rsi
        }

        return features
