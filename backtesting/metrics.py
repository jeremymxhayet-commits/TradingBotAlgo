import os
import logging
import pandas as pd


logger = logging.getLogger(__name__)


def evaluate_performance(equity_curve, trades, benchmark=None, output_path=None, tag=""):
    if equity_curve is None or equity_curve.empty:
        raise ValueError("Equity curve is empty; cannot evaluate performance.")

    equity = equity_curve["equity"]
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0
    drawdown = (equity / equity.cummax() - 1.0).min()
    num_trades = len(trades) if trades is not None else 0

    summary = pd.DataFrame(
        [
            {
                "total_return": total_return,
                "max_drawdown": drawdown,
                "num_trades": num_trades,
                "final_equity": equity.iloc[-1],
            }
        ]
    )

    logger.info("Total return: %.2f%%", total_return * 100)
    logger.info("Max drawdown: %.2f%%", drawdown * 100)
    logger.info("Number of trades: %d", num_trades)
    logger.info("Final equity: %.2f", equity.iloc[-1])

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        filename = f"performance_summary_{tag}.csv" if tag else "performance_summary.csv"
        summary.to_csv(os.path.join(output_path, filename), index=False)

    return summary
