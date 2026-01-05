#this is data_fetcher.py
import os
import time
import logging
from datetime import datetime

import pandas as pd
from tqdm import tqdm

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Please install yfinance with `pip install yfinance`")

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    raise ImportError("Please install alpaca-trade-api with `pip install alpaca-trade-api`")



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




class DataFetcher:
    def __init__(self, provider='yahoo', raw_data_dir='data/raw/', outlier_clip=(0.001, 0.999), fill_method='ffill'):
        self.provider = provider.lower()
        self.raw_data_dir = os.path.abspath(raw_data_dir)
        os.makedirs(self.raw_data_dir, exist_ok=True)

        self.outlier_clip = outlier_clip
        self.fill_method = fill_method

        if self.provider == 'alpaca':
            self.alpaca_api = self._init_alpaca_api()

    def _init_alpaca_api(self):
        key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
        secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
        base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
        if not key or not secret:
            raise ValueError("Set Alpaca API credentials in environment variables")
        return tradeapi.REST(key, secret, base_url, api_version='v2')

    def fetch(self, symbol, start, end, interval='1d', asset_type='equity'):
        logger.info(f"Fetching {symbol} from {start} to {end} via {self.provider} [{interval}]")

        if self.provider == 'yahoo':
            df = self._fetch_yahoo(symbol, start, end, interval)
        elif self.provider == 'alpaca':
            df = self._fetch_alpaca(symbol, start, end, interval)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        if df is not None and not df.empty:
            df = self._clean_and_preprocess(df, symbol, interval, start, end)
            self._save_to_versioned_csv(symbol, df, interval)
            return df
        else:
            logger.warning(f"No data fetched for {symbol}")
            return pd.DataFrame()

    def _fetch_yahoo(self, symbol, start, end, interval):
        yf_interval = {
            '1m': '1m', '5m': '5m', '15m': '15m', '1d': '1d'
        }.get(interval, '1d')
        try:
            df = yf.download(symbol, start=start, end=end, interval=yf_interval, progress=False, auto_adjust=True)
            df.dropna(how='all', inplace=True)
            df.index = pd.to_datetime(df.index)
            df.reset_index(inplace=True)
            df['symbol'] = symbol
            return df
        except Exception as e:
            logger.error(f"Yahoo fetch error for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_alpaca(self, symbol, start, end, interval):
        timeframe = {
            '1m': '1Min', '5m': '5Min', '15m': '15Min', '1d': '1D'
        }.get(interval, '1D')
        try:
            bars = self.alpaca_api.get_bars(
                symbol,
                timeframe,
                start=pd.to_datetime(start).isoformat(),
                end=pd.to_datetime(end).isoformat(),
                adjustment='all'
            ).df
            bars = bars[bars['symbol'] == symbol]
            bars.reset_index(inplace=True)
            return bars
        except Exception as e:
            logger.error(f"Alpaca fetch error for {symbol}: {e}")
            return pd.DataFrame()

    def _clean_and_preprocess(self, df, symbol, interval, start, end):
        logger.info(f"Cleaning & preprocessing {symbol} [{interval}]")

        df.columns = [col.strip().capitalize() for col in df.columns]
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Timestamp'}, inplace=True)
        elif 'Date' in df.columns:
            df.rename(columns={'Date': 'Timestamp'}, inplace=True)

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        df.sort_index(inplace=True)

        if self.fill_method == 'drop':
            df.dropna(inplace=True)
        elif self.fill_method == 'interpolate':
            df.interpolate(inplace=True)
        else:
            df.fillna(method=self.fill_method, inplace=True)

        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                lower, upper = df[col].quantile(self.outlier_clip[0]), df[col].quantile(self.outlier_clip[1])
                df = df[(df[col] >= lower) & (df[col] <= upper)]

        df['symbol'] = symbol
        return df

    def _save_to_versioned_csv(self, symbol, df, interval):
        filename = f"{symbol}_{interval}_latest.csv"
        filepath = os.path.join(self.raw_data_dir, filename)
        df.reset_index().to_csv(filepath, index=False)
        logger.info(f"Saved {symbol} [{interval}] data to {filepath}")

    def batch_fetch(self, symbols, start, end, interval='1d', asset_type='equity', sleep=1):
        for sym in tqdm(symbols, desc=f"Fetching batch via {self.provider}"):
            try:
                self.fetch(sym, start, end, interval, asset_type)
                time.sleep(sleep)
            except Exception as e:
                logger.error(f"Error fetching {sym}: {e}")


def load_historical_data(symbol, start, end, interval='1d'):
    raw_dir = os.path.abspath('data/raw/')
    filename = f"{symbol}_{interval}_latest.csv"
    filepath = os.path.join(raw_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath, parse_dates=['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df = df[start:end]
    return df


if __name__ == "__main__":
    fetcher = DataFetcher(provider='yahoo')

    symbols = ['AAPL', 'MSFT', 'GOOG']
    fetcher.batch_fetch(
        symbols=symbols,
        start='2020-01-01',
        end='2025-12-31',
        interval='1d',
        asset_type='equity'
    )
    print("Done fetching")