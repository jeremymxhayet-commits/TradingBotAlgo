import os
import time
import logging
import re

import pandas as pd
from tqdm import tqdm

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Please install yfinance with `pip install yfinance`")

tradeapi = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def normalize_interval(interval):
    if not interval:
        return "1d"
    key = str(interval).strip().lower()
    mapping = {
        "1min": "1m",
        "1m": "1m",
        "5min": "5m",
        "5m": "5m",
        "15min": "15m",
        "15m": "15m",
        "1d": "1d",
        "1day": "1d",
    }
    if key in mapping:
        return mapping[key]
    raise ValueError(f"Unsupported interval: {interval}")


def safe_symbol(symbol):
    return str(symbol).replace("/", "-")


def standardize_ohlcv_columns(df):
    if df is None or df.empty:
        return df

    df = df.copy()

    def _flatten_col(col):
        if isinstance(col, tuple):
            return "_".join([str(c).strip() for c in col if c is not None and str(c).strip() != ""])
        return str(col).strip()

    df.columns = [_flatten_col(c) for c in df.columns]

    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        tokens = [t for t in re.split(r"[^a-z0-9]+", key) if t]
        if key in ("date", "datetime", "timestamp") or any(t in ("date", "datetime", "timestamp") for t in tokens):
            rename_map[col] = "Timestamp"
        elif key == "open" or "open" in tokens:
            rename_map[col] = "Open"
        elif key == "high" or "high" in tokens:
            rename_map[col] = "High"
        elif key == "low" or "low" in tokens:
            rename_map[col] = "Low"
        elif key == "close" or ("close" in tokens and "adj" not in tokens):
            rename_map[col] = "Close"
        elif key in ("adj close", "adj_close", "adjclose") or ("adj" in tokens and "close" in tokens):
            if "Close" not in df.columns:
                rename_map[col] = "Close"
        elif key in ("volume", "vol") or "volume" in tokens or "vol" in tokens:
            rename_map[col] = "Volume"

    df.rename(columns=rename_map, inplace=True)

    if "Timestamp" not in df.columns:
        if df.index.name in ("Date", "Datetime", "Timestamp", "date", "datetime", "timestamp"):
            df["Timestamp"] = df.index
        else:
            df.insert(0, "Timestamp", df.index)

    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            raise ValueError(f"Missing required OHLC column: {col}")

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df.set_index("Timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df


class DataFetcher:
    def __init__(self, provider="yahoo", raw_data_dir="data/raw/", outlier_clip=(0.001, 0.999), fill_method="ffill"):
        self.provider = provider.lower()
        self.raw_data_dir = os.path.abspath(raw_data_dir)
        os.makedirs(self.raw_data_dir, exist_ok=True)

        self.outlier_clip = outlier_clip
        self.fill_method = fill_method

        if self.provider == "alpaca":
            self.alpaca_api = self._init_alpaca_api()

    def _init_alpaca_api(self):
        global tradeapi
        if tradeapi is None:
            try:
                import alpaca_trade_api as tradeapi  # type: ignore
            except ImportError as exc:
                raise ImportError("Please install alpaca-trade-api with `pip install alpaca-trade-api`") from exc

        key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
        secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
        base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
        if not key or not secret:
            raise ValueError("Set Alpaca API credentials in environment variables")
        return tradeapi.REST(key, secret, base_url, api_version="v2")

    def fetch(self, symbol, start, end, interval="1d", asset_type="equity"):
        interval = normalize_interval(interval)
        logger.info("Fetching %s from %s to %s via %s [%s]", symbol, start, end, self.provider, interval)

        if self.provider == "multi":
            if asset_type == "options":
                df = self._fetch_polygon(symbol, start, end, interval, asset_type=asset_type)
            elif asset_type == "forex":
                df = self._fetch_oanda(symbol, start, end, interval)
            else:
                df = self._fetch_yahoo(symbol, start, end, interval)
        elif self.provider == "yahoo":
            df = self._fetch_yahoo(symbol, start, end, interval)
        elif self.provider == "alpaca":
            df = self._fetch_alpaca(symbol, start, end, interval)
        elif self.provider == "polygon":
            df = self._fetch_polygon(symbol, start, end, interval, asset_type=asset_type)
        elif self.provider == "oanda":
            df = self._fetch_oanda(symbol, start, end, interval)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        if df is not None and not df.empty:
            df = self._clean_and_preprocess(df, symbol, interval, start, end)
            self._save_to_versioned_csv(symbol, df, interval)
            return df

        logger.warning("No data fetched for %s", symbol)
        return pd.DataFrame()

    def _fetch_yahoo(self, symbol, start, end, interval):
        yf_interval = normalize_interval(interval)
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval=yf_interval,
                progress=False,
                auto_adjust=True,
            )
            if df is None or df.empty:
                return pd.DataFrame()
            df.index = pd.to_datetime(df.index)
            df.reset_index(inplace=True)
            df["symbol"] = symbol
            return df
        except Exception as e:
            logger.error("Yahoo fetch error for %s: %s", symbol, e)
            return pd.DataFrame()

    def _fetch_alpaca(self, symbol, start, end, interval):
        timeframe = {
            "1m": "1Min",
            "5m": "5Min",
            "15m": "15Min",
            "1d": "1D",
        }.get(normalize_interval(interval), "1D")
        try:
            bars = self.alpaca_api.get_bars(
                symbol,
                timeframe,
                start=pd.to_datetime(start).isoformat(),
                end=pd.to_datetime(end).isoformat(),
                adjustment="all",
            ).df
            bars = bars[bars["symbol"] == symbol]
            bars.reset_index(inplace=True)
            return bars
        except Exception as e:
            logger.error("Alpaca fetch error for %s: %s", symbol, e)
            return pd.DataFrame()

    @staticmethod
    def _polygon_option_symbol(symbol: str) -> str:
        if symbol.startswith("O:"):
            return symbol
        return f"O:{symbol}"

    def _fetch_polygon(self, symbol, start, end, interval, asset_type="equity"):
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise ImportError("Please install requests with `pip install requests`") from exc
        from config import POLYGON_API_KEY

        if not POLYGON_API_KEY:
            raise ValueError("Set POLYGON_API_KEY in environment variables.")

        if asset_type == "options":
            ticker = self._polygon_option_symbol(symbol)
        else:
            ticker = symbol

        interval = normalize_interval(interval)
        multiplier = 1
        span = "day" if interval == "1d" else "minute"
        if interval == "5m":
            multiplier = 5
        elif interval == "15m":
            multiplier = 15

        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/"
            f"{multiplier}/{span}/{start}/{end}"
        )
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}
        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            logger.error("Polygon fetch error for %s: %s", symbol, response.text)
            return pd.DataFrame()
        payload = response.json()
        results = payload.get("results") or []
        if not results:
            return pd.DataFrame()
        rows = []
        for item in results:
            rows.append(
                {
                    "Timestamp": pd.to_datetime(item.get("t"), unit="ms"),
                    "Open": item.get("o"),
                    "High": item.get("h"),
                    "Low": item.get("l"),
                    "Close": item.get("c"),
                    "Volume": item.get("v", 0.0),
                }
            )
        return pd.DataFrame(rows)

    def _fetch_oanda(self, symbol, start, end, interval):
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise ImportError("Please install requests with `pip install requests`") from exc
        from config import OANDA_API_KEY, OANDA_BASE_URL

        if not OANDA_API_KEY:
            raise ValueError("Set OANDA_API_KEY in environment variables.")

        granularity_map = {
            "1m": "M1",
            "5m": "M5",
            "15m": "M15",
            "1d": "D",
        }
        granularity = granularity_map.get(normalize_interval(interval), "D")
        instrument = symbol.replace("/", "_")
        url = f"{OANDA_BASE_URL}/v3/instruments/{instrument}/candles"
        headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
        params = {"from": start, "to": end, "granularity": granularity, "price": "M"}
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
            logger.error("OANDA fetch error for %s: %s", symbol, response.text)
            return pd.DataFrame()
        payload = response.json()
        candles = payload.get("candles") or []
        rows = []
        for candle in candles:
            if not candle.get("complete", True):
                continue
            mid = candle.get("mid", {})
            rows.append(
                {
                    "Timestamp": pd.to_datetime(candle.get("time")),
                    "Open": float(mid.get("o", 0.0)),
                    "High": float(mid.get("h", 0.0)),
                    "Low": float(mid.get("l", 0.0)),
                    "Close": float(mid.get("c", 0.0)),
                    "Volume": float(candle.get("volume", 0.0)),
                }
            )
        return pd.DataFrame(rows)

    def _clean_and_preprocess(self, df, symbol, interval, start, end):
        logger.info("Cleaning & preprocessing %s [%s]", symbol, interval)
        try:
            df = standardize_ohlcv_columns(df)
        except ValueError as exc:
            logger.warning("Skipping %s due to data format issue: %s", symbol, exc)
            return pd.DataFrame()

        if self.fill_method == "drop":
            df.dropna(inplace=True)
        elif self.fill_method == "interpolate":
            df.interpolate(inplace=True)
        else:
            if self.fill_method == "ffill":
                df.ffill(inplace=True)
            elif self.fill_method == "bfill":
                df.bfill(inplace=True)
            else:
                df.fillna(method=self.fill_method, inplace=True)

        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                lower, upper = df[col].quantile(self.outlier_clip[0]), df[col].quantile(self.outlier_clip[1])
                df = df[(df[col] >= lower) & (df[col] <= upper)]

        df["symbol"] = symbol
        return df

    def _save_to_versioned_csv(self, symbol, df, interval):
        interval = normalize_interval(interval)
        filename = f"{safe_symbol(symbol)}_{interval}_latest.csv"
        filepath = os.path.join(self.raw_data_dir, filename)
        df.reset_index().to_csv(filepath, index=False)
        logger.info("Saved %s [%s] data to %s", symbol, interval, filepath)

    def batch_fetch(self, symbols, start, end, interval="1d", asset_type="equity", sleep=1):
        interval = normalize_interval(interval)
        for sym in tqdm(symbols, desc=f"Fetching batch via {self.provider}"):
            try:
                self.fetch(sym, start, end, interval, asset_type)
                time.sleep(sleep)
            except Exception as e:
                logger.error("Error fetching %s: %s", sym, e)


def load_historical_data(symbol, start, end, interval="1d"):
    raw_dir = os.path.abspath("data/raw/")
    interval = normalize_interval(interval)
    filename = f"{safe_symbol(symbol)}_{interval}_latest.csv"
    filepath = os.path.join(raw_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    df = standardize_ohlcv_columns(df)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[start:end]
    return df


if __name__ == "__main__":
    fetcher = DataFetcher(provider="yahoo")
    symbols = ["AAPL", "MSFT", "GOOG"]
    fetcher.batch_fetch(
        symbols=symbols,
        start="2020-01-01",
        end="2025-12-31",
        interval="1d",
        asset_type="equity",
    )
    print("Done fetching")
