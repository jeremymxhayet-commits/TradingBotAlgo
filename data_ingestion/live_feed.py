import os
import asyncio
import logging
import json
import threading
import queue
from datetime import datetime
from typing import List, Dict

import pandas as pd
from alpaca.data.live import StockDataStream, CryptoDataStream
from alpaca.data.enums import DataFeed, CryptoFeed
from alpaca.data.models import Trade, Quote


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class RealTimeFeedHandler:
    def __init__(self, symbols: List[str], asset_type: str = 'equity', queue_maxsize=1000):
        self.symbols = symbols
        self.asset_type = asset_type.lower()
        self.data_queue = queue.Queue(maxsize=queue_maxsize)

        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.api_secret = os.getenv("APCA_API_SECRET_KEY")

        if not self.api_key or not self.api_secret:
            raise EnvironmentError("Alpaca API credentials not set in environment variables")

        if self.asset_type in ['equity', 'etf']:
            self.stream = StockDataStream(
                self.api_key,
                self.api_secret,
                base_url="https://api.alpaca.markets",
                feed=DataFeed.IEX  
            )
        elif self.asset_type in ['crypto', 'forex']:
            self.stream = CryptoDataStream(
                self.api_key,
                self.api_secret,
                feed=CryptoFeed.US
            )
        else:
            raise ValueError(f"Unsupported asset_type: {self.asset_type}")

    def _normalize_trade(self, trade: Trade) -> Dict:
        return {
            'type': 'trade',
            'symbol': trade.symbol,
            'timestamp': trade.timestamp.isoformat(),
            'price': trade.price,
            'volume': trade.size
        }

    def _normalize_quote(self, quote: Quote) -> Dict:
        return {
            'type': 'quote',
            'symbol': quote.symbol,
            'timestamp': quote.timestamp.isoformat(),
            'bid_price': quote.bid_price,
            'bid_size': quote.bid_size,
            'ask_price': quote.ask_price,
            'ask_size': quote.ask_size
        }

    async def _trade_callback(self, trade: Trade):
        norm = self._normalize_trade(trade)
        try:
            self.data_queue.put_nowait(norm)
        except queue.Full:
            logger.warning("Data queue full. Dropping trade tick.")

    async def _quote_callback(self, quote: Quote):
        norm = self._normalize_quote(quote)
        try:
            self.data_queue.put_nowait(norm)
        except queue.Full:
            logger.warning("Data queue full. Dropping quote tick.")

    async def _subscribe(self):
        for symbol in self.symbols:
            await self.stream.subscribe_trades(self._trade_callback, symbol)
            await self.stream.subscribe_quotes(self._quote_callback, symbol)
        logger.info(f"Subscribed to real-time {self.asset_type} feed for: {', '.join(self.symbols)}")

    async def _start_stream(self):
        await self._subscribe()
        await self.stream._run_forever()

    def start(self):
        """Start feed in a background thread."""
        loop = asyncio.new_event_loop()

        def _run():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._start_stream())

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        logger.info("Live feed handler started in background thread.")

    def get_data_queue(self):
        return self.data_queue


class BarAggregator:
    def __init__(self, interval_seconds=60):
        self.interval = interval_seconds
        self.bars = {}

    def update(self, tick: Dict) -> List[Dict]:
        """
        Aggregate tick into OHLCV bars.
        Returns a list of completed bars (if any).
        """
        completed_bars = []
        if tick['type'] != 'trade':
            return completed_bars

        symbol = tick['symbol']
        ts = pd.to_datetime(tick['timestamp'])
        price = tick['price']
        volume = tick['volume']
        rounded_ts = ts.floor(f'{self.interval}s')

        if symbol not in self.bars:
            self.bars[symbol] = {}

        bar = self.bars[symbol].get(rounded_ts, {
            'symbol': symbol,
            'timestamp': rounded_ts,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': 0
        })

        bar['high'] = max(bar['high'], price)
        bar['low'] = min(bar['low'], price)
        bar['close'] = price
        bar['volume'] += volume

        self.bars[symbol][rounded_ts] = bar

        now = pd.Timestamp.utcnow()
        keys_to_flush = [ts for ts in self.bars[symbol].keys() if ts < now.floor(f'{self.interval}s') - pd.Timedelta(seconds=self.interval)]
        for ts_key in keys_to_flush:
            completed_bars.append(self.bars[symbol].pop(ts_key))

        return completed_bars



if __name__ == "__main__":
    import time

    symbols = ['AAPL', 'MSFT']
    feed = RealTimeFeedHandler(symbols, asset_type='equity')
    feed.start()

    queue_ref = feed.get_data_queue()
    aggregator = BarAggregator(interval_seconds=60)

    print("Listening to live data... Press CTRL+C to stop.")
    try:
        while True:
            while not queue_ref.empty():
                tick = queue_ref.get()
                bars = aggregator.update(tick)
                for bar in bars:
                    print(f"[BAR] {bar}")
                print(f"[TICK] {tick}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopped.")
