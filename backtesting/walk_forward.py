import logging
from typing import List, Tuple

import pandas as pd

from config import (
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    WALK_FORWARD_TEST_MONTHS,
    WALK_FORWARD_RETRAIN_MONTHS,
    WALK_FORWARD_EXPANDING,
)
from models.train_models import train_models_for_range
from main_backtest import run_backtest


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return max(1, (end.year - start.year) * 12 + (end.month - start.month))


def _add_months(dt: pd.Timestamp, months: int) -> pd.Timestamp:
    return (dt + pd.DateOffset(months=months)).normalize()


def _window_tag(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"wf_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"


def generate_walk_forward_windows(
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    test_months: int,
    retrain_months: int,
    expanding: bool = True,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)
    test_end_dt = pd.to_datetime(test_end)

    windows = []
    current_train_start = train_start_dt
    current_train_end = train_end_dt
    current_test_start = test_start_dt
    train_window_months = _months_between(train_start_dt, train_end_dt)

    while current_test_start <= test_end_dt:
        current_test_end = _add_months(current_test_start, test_months) - pd.Timedelta(days=1)
        if current_test_end > test_end_dt:
            current_test_end = test_end_dt
        windows.append((current_train_start, current_train_end, current_test_start, current_test_end))

        current_test_start = current_test_end + pd.Timedelta(days=1)
        if expanding:
            current_train_end = current_test_end
        else:
            current_train_start = _add_months(current_train_start, retrain_months)
            current_train_end = _add_months(current_train_start, train_window_months) - pd.Timedelta(days=1)
    return windows


def run_walk_forward(
    strategy_name: str = "ml",
    train_start: str = TRAIN_START_DATE,
    train_end: str = TRAIN_END_DATE,
    test_start: str = TEST_START_DATE,
    test_end: str = TEST_END_DATE,
    test_months: int = WALK_FORWARD_TEST_MONTHS,
    retrain_months: int = WALK_FORWARD_RETRAIN_MONTHS,
    expanding: bool = WALK_FORWARD_EXPANDING,
):
    windows = generate_walk_forward_windows(
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        test_months=test_months,
        retrain_months=retrain_months,
        expanding=expanding,
    )
    if not windows:
        raise RuntimeError("No walk-forward windows generated.")

    for train_start_dt, train_end_dt, test_start_dt, test_end_dt in windows:
        logger.info(
            "Walk-forward window | train=%s -> %s | test=%s -> %s",
            train_start_dt.date(),
            train_end_dt.date(),
            test_start_dt.date(),
            test_end_dt.date(),
        )
        train_models_for_range(
            start_date=train_start_dt.strftime("%Y-%m-%d"),
            end_date=train_end_dt.strftime("%Y-%m-%d"),
        )
        run_backtest(
            strategy_name=strategy_name,
            start_date=test_start_dt.strftime("%Y-%m-%d"),
            end_date=test_end_dt.strftime("%Y-%m-%d"),
            tag=_window_tag(test_start_dt, test_end_dt),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run walk-forward backtest")
    parser.add_argument("--strategy", choices=["ml", "stat", "ensemble"], default="ml")
    parser.add_argument("--train-start", default=TRAIN_START_DATE)
    parser.add_argument("--train-end", default=TRAIN_END_DATE)
    parser.add_argument("--test-start", default=TEST_START_DATE)
    parser.add_argument("--test-end", default=TEST_END_DATE)
    parser.add_argument("--test-months", type=int, default=WALK_FORWARD_TEST_MONTHS)
    parser.add_argument("--retrain-months", type=int, default=WALK_FORWARD_RETRAIN_MONTHS)
    parser.add_argument("--expanding", action="store_true", default=WALK_FORWARD_EXPANDING)
    args = parser.parse_args()

    run_walk_forward(
        strategy_name=args.strategy,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        test_months=args.test_months,
        retrain_months=args.retrain_months,
        expanding=args.expanding,
    )
