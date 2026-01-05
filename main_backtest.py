print(">>> Backtest script starting...")
import os
import logging
import time
from datetime import datetime

from config import (
    BACKTEST_START_DATE,
    BACKTEST_END_DATE,
    INITIAL_CAPITAL,
    TRADE_UNIVERSE,
    DEFAULT_BAR_INTERVAL,
    RESULTS_PATH,
    MAX_DRAWDOWN_PCT,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TRAILING_STOP_PCT,
    ML_THRESHOLD,
    XGB_MODEL_PATH,
    LSTM_MODEL_PATH,
    RL_MODEL_PATH,
    XGB_PARAMS,
    ML_ONLINE_LEARNING,
    ML_ONLINE_WINDOW,
    ML_ONLINE_INTERVAL,
    ML_ONLINE_MIN_SAMPLES,
    ENABLED_ASSET_CLASSES,
    YAHOO_SYMBOL_MAP,
    ASSET_CLASS_THRESHOLDS,
    XGB_CLASS_MODEL_PATHS,
    RISK_PER_TRADE,
    MAX_POSITION_PER_ASSET,
    VOLATILITY_TARGET,
    VOL_LOOKBACK,
    ATR_LOOKBACK,
    TREND_LOOKBACK,
    MIN_TREND_STRENGTH,
    MIN_VOLATILITY,
    NO_TRADE_PROB_BAND,
    MIN_EDGE_TO_COST,
    MAX_VOLUME_PARTICIPATION,
    MIN_TRADE_QTY,
    ATR_STOP_MULT,
    ATR_TAKE_PROFIT_MULT,
    ATR_TRAIL_MULT,
    ENSEMBLE_LOOKBACK,
    ENSEMBLE_MIN_WEIGHT,
    ENSEMBLE_MIN_SCORE,
    SLIPPAGE,
    COMMISSION,
    ALLOW_SHORT,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    VALIDATION_START_DATE,
    VALIDATION_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
)

from data_ingestion.data_fetcher import (
    DataFetcher,
    load_historical_data,
    normalize_interval,
)
from strategies.ml_strategy import MLStrategy
from strategies.statistical_arbitrage import StatisticalArbitrageStrategy
from strategies.ensemble_strategy import EnsembleStrategy
from backtesting.backtester import Backtester
from backtesting.metrics import evaluate_performance
from models.train_models import train_models_for_range


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

os.makedirs(RESULTS_PATH, exist_ok=True)


def _load_or_fetch_symbol(fetcher, symbol, start, end, interval, asset_type=None):
    yahoo_symbol = YAHOO_SYMBOL_MAP.get(symbol, symbol)
    try:
        df = load_historical_data(yahoo_symbol, start, end, interval)
        if df.empty:
            logger.warning("Loaded empty data for %s [%s]. Skipping.", symbol, interval)
            return None
        return df
    except FileNotFoundError:
        logger.info("Missing CSV for %s [%s]. Fetching...", symbol, interval)
        df = fetcher.fetch(yahoo_symbol, start, end, interval, asset_type=asset_type or "equity")
        if df is None or df.empty:
            logger.warning("No data fetched for %s [%s]. Skipping.", symbol, interval)
            return None
        return df


def run_backtest(strategy_name="ml", start_date=None, end_date=None, tag=None):
    start_time = time.monotonic()
    logger.info("=== Starting Backtest: %s Strategy ===", strategy_name.upper())

    symbol_asset_map = {}
    if isinstance(TRADE_UNIVERSE, dict):
        symbol_list = []
        for asset_class, symbols in TRADE_UNIVERSE.items():
            if ENABLED_ASSET_CLASSES and asset_class not in ENABLED_ASSET_CLASSES:
                continue
            for sym in symbols:
                symbol_list.append(sym)
                symbol_asset_map[sym] = asset_class
        seen = set()
        symbol_list = [s for s in symbol_list if not (s in seen or seen.add(s))]
    else:
        symbol_list = list(TRADE_UNIVERSE)
    interval = normalize_interval(DEFAULT_BAR_INTERVAL)
    start = start_date or BACKTEST_START_DATE
    end = end_date or BACKTEST_END_DATE
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    if interval in ("1m", "5m", "15m"):
        delta_days = (end_dt - start_dt).days
        if delta_days > 8:
            logger.warning("Intraday interval %s over %d days exceeds Yahoo limit; switching to 1d.", interval, delta_days)
            interval = "1d"

    fetcher = DataFetcher(provider="multi")
    price_data = {}

    logger.info("Loading historical data from data/raw...")
    for symbol in symbol_list:
        asset_type = symbol_asset_map.get(symbol, "equity")
        df = _load_or_fetch_symbol(
            fetcher=fetcher,
            symbol=symbol,
            start=start,
            end=end,
            interval=interval,
            asset_type=asset_type,
        )
        if df is None or df.empty:
            continue
        price_data[symbol] = df

        date_min = df.index.min()
        date_max = df.index.max()
        logger.info(
            "Loaded %s | rows=%d | range=%s -> %s",
            symbol,
            len(df),
            date_min.strftime("%Y-%m-%d %H:%M:%S"),
            date_max.strftime("%Y-%m-%d %H:%M:%S"),
        )

    if not price_data:
        raise RuntimeError("No historical data loaded for any symbols.")

    logger.info("Symbols loaded: %d", len(price_data))

    if strategy_name == "ml":
        strategy = MLStrategy(
            symbol_list,
            xgb_model_path=XGB_MODEL_PATH,
            lstm_model_path=LSTM_MODEL_PATH,
            rl_model_path=RL_MODEL_PATH,
            threshold=ML_THRESHOLD,
            xgb_params=XGB_PARAMS,
            online_learning=ML_ONLINE_LEARNING,
            online_window=ML_ONLINE_WINDOW,
            online_interval=ML_ONLINE_INTERVAL,
            online_min_samples=ML_ONLINE_MIN_SAMPLES,
            risk_per_trade=RISK_PER_TRADE,
            max_position_per_asset=MAX_POSITION_PER_ASSET,
            volatility_target=VOLATILITY_TARGET,
            vol_lookback=VOL_LOOKBACK,
            atr_lookback=ATR_LOOKBACK,
            trend_lookback=TREND_LOOKBACK,
            min_trend_strength=MIN_TREND_STRENGTH,
            min_volatility=MIN_VOLATILITY,
            no_trade_prob_band=NO_TRADE_PROB_BAND,
            min_edge_to_cost=MIN_EDGE_TO_COST,
            max_volume_participation=MAX_VOLUME_PARTICIPATION,
            min_trade_qty=MIN_TRADE_QTY,
            atr_stop_mult=ATR_STOP_MULT,
            atr_take_profit_mult=ATR_TAKE_PROFIT_MULT,
            atr_trail_mult=ATR_TRAIL_MULT,
            ensemble_lookback=ENSEMBLE_LOOKBACK,
            ensemble_min_weight=ENSEMBLE_MIN_WEIGHT,
            ensemble_min_score=ENSEMBLE_MIN_SCORE,
            cost_rate=float(COMMISSION) + float(SLIPPAGE),
            allow_short=ALLOW_SHORT,
            symbol_asset_map=symbol_asset_map,
            class_thresholds=ASSET_CLASS_THRESHOLDS,
            xgb_class_model_paths=XGB_CLASS_MODEL_PATHS,
        )
    elif strategy_name == "stat":
        strategy = StatisticalArbitrageStrategy(symbol_list)
    elif strategy_name == "ensemble":
        strategy = EnsembleStrategy(symbol_list)
    else:
        raise ValueError(f"Unsupported strategy: {strategy_name}")

    strategy.cash = INITIAL_CAPITAL
    strategy.positions = {}
    if hasattr(strategy, "model_status"):
        status = strategy.model_status()
        logger.info(
            "Model status | xgb=%s lstm=%s rl=%s class_models=%s",
            status.get("xgb"),
            status.get("lstm"),
            status.get("rl"),
            status.get("xgb_class_models"),
        )

    bt = Backtester(
        strategy=strategy,
        price_data=price_data,
        initial_cash=INITIAL_CAPITAL,
        start_date=start,
        end_date=end,
        slippage_bps=int(SLIPPAGE * 10000),
        max_drawdown_pct=MAX_DRAWDOWN_PCT,
        stop_loss_pct=STOP_LOSS_PCT,
        take_profit_pct=TAKE_PROFIT_PCT,
        trailing_stop_pct=TRAILING_STOP_PCT,
        commission_rate=COMMISSION,
        allow_short=ALLOW_SHORT,
    )

    logger.info("Running simulation...")
    results = bt.run()

    equity_curve = results["equity_curve"]
    trades = results["trades"]

    tag_suffix = f"_{tag}" if tag else ""
    equity_path = os.path.join(RESULTS_PATH, f"equity_curve_{strategy_name}{tag_suffix}.csv")
    trades_path = os.path.join(RESULTS_PATH, f"trades_{strategy_name}{tag_suffix}.csv")
    equity_curve.to_csv(equity_path)
    trades.to_csv(trades_path)

    logger.info("Trades executed: %d", len(trades))
    logger.info("Final equity: %.2f", equity_curve["equity"].iloc[-1])

    logger.info("Performance Summary:")
    evaluate_performance(
        equity_curve=equity_curve,
        trades=trades,
        benchmark=None,
        output_path=RESULTS_PATH,
        tag=f"{strategy_name}{tag_suffix}",
    )

    elapsed = int(time.monotonic() - start_time)
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    logger.info("Backtest complete in %02d:%02d:%02d.", hours, minutes, seconds)


def run_train_validate_test(
    strategy_name="ml",
    train_start=TRAIN_START_DATE,
    train_end=TRAIN_END_DATE,
    validation_start=VALIDATION_START_DATE,
    validation_end=VALIDATION_END_DATE,
    test_start=TEST_START_DATE,
    test_end=TEST_END_DATE,
):
    logger.info(
        "Training models for %s -> %s", train_start, train_end
    )
    train_models_for_range(start_date=train_start, end_date=train_end)
    logger.info("Running validation backtest...")
    run_backtest(strategy_name=strategy_name, start_date=validation_start, end_date=validation_end, tag="val")
    logger.info("Running test backtest...")
    run_backtest(strategy_name=strategy_name, start_date=test_start, end_date=test_end, tag="test")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run strategy backtest")
    parser.add_argument(
        "--strategy", choices=["ml", "stat", "ensemble"],
        default="ml", help="Which strategy to backtest"
    )
    parser.add_argument("--start-date", default=None, help="Override backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Override backtest end date (YYYY-MM-DD)")
    parser.add_argument("--tag", default=None, help="Tag for output filenames")
    parser.add_argument("--split", action="store_true", help="Run train/validation/test split backtest")
    args = parser.parse_args()
    if args.split:
        run_train_validate_test(strategy_name=args.strategy)
    else:
        run_backtest(
            strategy_name=args.strategy,
            start_date=args.start_date,
            end_date=args.end_date,
            tag=args.tag,
        )
