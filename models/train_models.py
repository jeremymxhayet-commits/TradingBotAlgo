import os
import logging
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import (
    DATA_PATH,
    MODEL_PATH,
    XGB_MODEL_PATH,
    LSTM_MODEL_PATH,
    XGB_PARAMS,
    LSTM_PARAMS,
    BACKTEST_START_DATE,
    BACKTEST_END_DATE,
    TRADE_UNIVERSE,
    YAHOO_SYMBOL_MAP,
    XGB_CLASS_MODEL_PATHS,
)
from data_ingestion.data_fetcher import standardize_ohlcv_columns
from data_ingestion.data_fetcher import safe_symbol


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=series.index).rolling(window=period).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    return 100 - (100 / (1 + rs))


def _generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["returns"] = df["Close"].pct_change()
    df["sma"] = df["Close"].rolling(window=10).mean()
    df["rsi"] = _compute_rsi(df["Close"])
    df["macd"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    df["vol"] = df["returns"].rolling(window=10).std()
    df = df.dropna()
    return df[["returns", "sma", "rsi", "macd", "vol"]]


def _filter_by_date(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[start_date:end_date]


def _load_raw_csvs(start_date: str, end_date: str) -> List[pd.DataFrame]:
    raw_dir = os.path.join(DATA_PATH, "raw")
    if not os.path.isdir(raw_dir):
        return []
    dataframes = []
    for name in os.listdir(raw_dir):
        if not name.endswith(".csv"):
            continue
        path = os.path.join(raw_dir, name)
        df = pd.read_csv(path)
        df = standardize_ohlcv_columns(df)
        df = _filter_by_date(df, start_date, end_date)
        if not df.empty and "Close" in df.columns:
            dataframes.append(df)
    return dataframes


def _load_symbol_csv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    yahoo_symbol = YAHOO_SYMBOL_MAP.get(symbol, symbol)
    filename = f"{safe_symbol(yahoo_symbol)}_1d_latest.csv"
    raw_dir = os.path.join(DATA_PATH, "raw")
    path = os.path.join(raw_dir, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = standardize_ohlcv_columns(df)
    df = _filter_by_date(df, start_date, end_date)
    return df


def _load_raw_csvs_for_symbols(symbols: List[str], start_date: str, end_date: str) -> List[pd.DataFrame]:
    dfs = []
    for symbol in symbols:
        df = _load_symbol_csv(symbol, start_date, end_date)
        if not df.empty and "Close" in df.columns:
            dfs.append(df)
    return dfs


def _build_xgb_dataset(dfs: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    features_list = []
    target_list = []
    for df in dfs:
        features = _generate_features(df)
        if features.empty:
            continue
        close = df.loc[features.index, "Close"]
        target = (close.pct_change().shift(-1) > 0).astype(int)
        aligned = features.iloc[:-1]
        target = target.iloc[:-1]
        if aligned.empty or target.empty:
            continue
        features_list.append(aligned.values)
        target_list.append(target.values)
    if not features_list:
        return np.empty((0, 5)), np.empty((0,), dtype=int)
    X = np.vstack(features_list)
    y = np.concatenate(target_list)
    return X, y


def _build_lstm_dataset(dfs: List[pd.DataFrame], seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    sequences = []
    targets = []
    for df in dfs:
        features = _generate_features(df)
        if len(features) <= seq_len:
            continue
        close = df.loc[features.index, "Close"].values
        feats = features.values
        for i in range(len(features) - seq_len):
            sequences.append(feats[i : i + seq_len])
            targets.append(close[i + seq_len])
    if not sequences:
        return np.empty((0, seq_len, 5)), np.empty((0,), dtype=float)
    return np.array(sequences), np.array(targets)


class SimpleProbModel:
    def __init__(self, prob: float):
        self.prob = float(prob)

    def predict_proba(self, X):
        prob = np.clip(self.prob, 1e-3, 1 - 1e-3)
        return np.column_stack([1 - prob, np.full(len(X), prob)])


def train_xgb_model(dfs: List[pd.DataFrame]):
    X, y = _build_xgb_dataset(dfs)
    if X.size == 0:
        logger.warning("No data available to train XGB model.")
        return None
    try:
        import xgboost as xgb  # type: ignore

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X, y)
        return model
    except Exception as exc:
        logger.warning("XGBoost unavailable, falling back to sklearn or simple model: %s", exc)
    try:
        from sklearn.ensemble import GradientBoostingClassifier  # type: ignore

        model = GradientBoostingClassifier()
        model.fit(X, y)
        return model
    except Exception as exc:
        logger.warning("Sklearn unavailable, using simple probability model: %s", exc)
    return SimpleProbModel(prob=float(np.mean(y)))


def train_lstm_model(dfs: List[pd.DataFrame]):
    seq_len = int(LSTM_PARAMS.get("sequence_length", 30))
    seq_len = max(10, min(seq_len, 60))
    X, y = _build_lstm_dataset(dfs, seq_len)
    if X.size == 0:
        logger.warning("No data available to train LSTM model.")
        return None

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(seq_len, X.shape[2])),
            tf.keras.layers.LSTM(LSTM_PARAMS.get("units", 64)),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    epochs = int(LSTM_PARAMS.get("epochs", 5))
    batch_size = int(LSTM_PARAMS.get("batch_size", 32))
    model.fit(X, y, epochs=max(1, min(epochs, 5)), batch_size=batch_size, verbose=1)
    return model


def train_models_for_range(start_date: str, end_date: str, symbols_by_asset=None):
    os.makedirs(MODEL_PATH, exist_ok=True)
    dfs = _load_raw_csvs(start_date, end_date)
    if not dfs:
        logger.error("No CSVs found under data/raw for %s -> %s.", start_date, end_date)
        return None

    logger.info("Training XGB model...")
    xgb_model = train_xgb_model(dfs)
    if xgb_model is not None:
        joblib.dump(xgb_model, XGB_MODEL_PATH)
        logger.info("Saved XGB model to %s", XGB_MODEL_PATH)

    symbols_by_asset = symbols_by_asset or TRADE_UNIVERSE
    for asset_class, symbols in symbols_by_asset.items():
        class_path = XGB_CLASS_MODEL_PATHS.get(asset_class)
        if not class_path:
            continue
        class_dfs = _load_raw_csvs_for_symbols(symbols, start_date, end_date)
        if not class_dfs:
            continue
        logger.info("Training XGB model for %s...", asset_class)
        class_model = train_xgb_model(class_dfs)
        if class_model is not None:
            joblib.dump(class_model, class_path)
            logger.info("Saved XGB %s model to %s", asset_class, class_path)

    logger.info("Training LSTM model...")
    lstm_model = train_lstm_model(dfs)
    if lstm_model is not None:
        lstm_model.save(LSTM_MODEL_PATH)
        logger.info("Saved LSTM model to %s", LSTM_MODEL_PATH)
    return {"xgb": xgb_model is not None, "lstm": lstm_model is not None}


def main():
    os.makedirs(MODEL_PATH, exist_ok=True)
    dfs = _load_raw_csvs(BACKTEST_START_DATE, BACKTEST_END_DATE)
    if not dfs:
        logger.error("No CSVs found under data/raw. Run a backtest fetch first.")
        return

    logger.info("Training XGB model...")
    xgb_model = train_xgb_model(dfs)
    if xgb_model is not None:
        joblib.dump(xgb_model, XGB_MODEL_PATH)
        logger.info("Saved XGB model to %s", XGB_MODEL_PATH)

    for asset_class, symbols in TRADE_UNIVERSE.items():
        class_path = XGB_CLASS_MODEL_PATHS.get(asset_class)
        if not class_path:
            continue
        class_dfs = _load_raw_csvs_for_symbols(symbols, BACKTEST_START_DATE, BACKTEST_END_DATE)
        if not class_dfs:
            continue
        logger.info("Training XGB model for %s...", asset_class)
        class_model = train_xgb_model(class_dfs)
        if class_model is not None:
            joblib.dump(class_model, class_path)
            logger.info("Saved XGB %s model to %s", asset_class, class_path)

    logger.info("Training LSTM model...")
    lstm_model = train_lstm_model(dfs)
    if lstm_model is not None:
        lstm_model.save(LSTM_MODEL_PATH)
        logger.info("Saved LSTM model to %s", LSTM_MODEL_PATH)


if __name__ == "__main__":
    main()
