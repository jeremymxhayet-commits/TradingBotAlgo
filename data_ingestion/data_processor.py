import os
import logging
from typing import List
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import BollingerBands
from ta.utils import dropna

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class DataProcessor:
    def __init__(self, output_dir='data/processed'):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def process_file(self, filepath: str, label_horizon: int = 5) -> pd.DataFrame:
        logger.info(f"Processing file: {filepath}")
        df = pd.read_csv(filepath, parse_dates=['Timestamp'])
        df = dropna(df)
        df.set_index('Timestamp', inplace=True)
        df.sort_index(inplace=True)

        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            logger.error("Missing OHLCV columns in dataset")
            return pd.DataFrame()

        df = self._add_technical_indicators(df)
        df = self._add_alpha_factors(df)
        df = self._add_labels(df, horizon=label_horizon)
        df.dropna(inplace=True)

        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'unknown'
        filename = f"{symbol}_processed.csv"
        outpath = os.path.join(self.output_dir, filename)
        df.to_csv(outpath)
        logger.info(f"Saved processed data to {outpath}")
        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['sma_10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
        df['ema_10'] = EMAIndicator(close=df['Close'], window=10).ema_indicator()
        df['rsi_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
        df['macd'] = MACD(close=df['Close']).macd()
        df['macd_signal'] = MACD(close=df['Close']).macd_signal()
        df['macd_diff'] = MACD(close=df['Close']).macd_diff()

        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = df['bb_upper'] - df['bb_lower']

        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        df['volatility_10'] = df['Close'].rolling(window=10).std()
        return df

    def _add_alpha_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        df['return_1'] = df['Close'].pct_change(1)
        df['return_5'] = df['Close'].pct_change(5)
        df['zscore_10'] = (df['Close'] - df['Close'].rolling(10).mean()) / df['Close'].rolling(10).std()
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        return df

    def _add_labels(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        df['future_return'] = df['Close'].shift(-horizon) / df['Close'] - 1
        df['target_class'] = np.where(df['future_return'] > 0.002, 1,
                              np.where(df['future_return'] < -0.002, -1, 0))
        return df

    def batch_process(self, filepaths: List[str], label_horizon: int = 5):
        for fp in filepaths:
            try:
                self.process_file(fp, label_horizon=label_horizon)
            except Exception as e:
                logger.error(f"Error processing {fp}: {e}")


if __name__ == "__main__":
    import glob

    processor = DataProcessor(output_dir='data/processed')
    files = glob.glob('data/raw/*.csv')
    processor.batch_process(files, label_horizon=5)
