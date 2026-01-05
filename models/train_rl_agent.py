import argparse
import logging
import os
from typing import List

import pandas as pd

from config import (
    DATA_PATH,
    MODEL_PATH,
    RL_MODEL_PATH,
    RL_PARAMS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TRADE_UNIVERSE,
    YAHOO_SYMBOL_MAP,
)
from data_ingestion.data_fetcher import standardize_ohlcv_columns
from data_ingestion.data_fetcher import safe_symbol
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _filter_by_date(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[start_date:end_date]


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


def _load_training_data(start_date: str, end_date: str) -> pd.DataFrame:
    symbols = []
    for _, sym_list in TRADE_UNIVERSE.items():
        symbols.extend(sym_list)
    dfs = _load_raw_csvs_for_symbols(symbols, start_date, end_date)
    if not dfs:
        raise RuntimeError("No CSVs found under data/raw for RL training.")
    combined = pd.concat(dfs, axis=0, ignore_index=True)
    if "Date" in combined.columns:
        combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
        combined = combined.dropna(subset=["Date"]).sort_values("Date")
        combined = combined.set_index("Date")
    return combined


def train_rl_agent(start_date: str, end_date: str):
    try:
        import gymnasium  
    except Exception:
        try:
            import gym  
        except Exception as exc:
            logger.error("gymnasium (or gym) is required for RL training: %s", exc)
            logger.error("Install with: pip install gymnasium")
            return None
    try:
        from stable_baselines3 import PPO 
        from stable_baselines3.common.vec_env import DummyVecEnv 
    except Exception as exc:
        logger.error("stable_baselines3 is required for RL training: %s", exc)
        return None
    from reinforcement_learning.trading_env import TradingEnv

    os.makedirs(MODEL_PATH, exist_ok=True)
    df = _load_training_data(start_date, end_date)
    env = DummyVecEnv([lambda: TradingEnv(df)])

    policy_kwargs = RL_PARAMS.get("policy_kwargs", {})
    total_timesteps = int(RL_PARAMS.get("total_timesteps", 100_000))
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(RL_MODEL_PATH)
    logger.info("Saved RL agent to %s", RL_MODEL_PATH)
    return RL_MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Train RL trading agent")
    parser.add_argument("--start-date", default=TRAIN_START_DATE)
    parser.add_argument("--end-date", default=TRAIN_END_DATE)
    args = parser.parse_args()
    train_rl_agent(start_date=args.start_date, end_date=args.end_date)


if __name__ == "__main__":
    main()
