import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm

class RiskManager:
    def __init__(self, capital: float, max_drawdown_pct: float, max_leverage: float):
        """
        :param capital: Total capital available.
        :param max_drawdown_pct: Maximum drawdown percentage allowed before halting.
        :param max_leverage: Maximum leverage allowed (e.g., 2.0 = 200%).
        """
        self.initial_capital = capital
        self.max_drawdown = max_drawdown_pct
        self.max_leverage = max_leverage
        self.peak_value = capital

    def check_drawdown(self, portfolio_value: float) -> bool:
        self.peak_value = max(self.peak_value, portfolio_value)
        drawdown = (self.peak_value - portfolio_value) / self.peak_value
        return drawdown <= self.max_drawdown

    def calculate_parametric_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Parametric VaR using mean-variance method."""
        mean = returns.mean()
        std = returns.std()
        var = norm.ppf(1 - confidence) * std - mean
        return var

    def calculate_historical_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        return -np.percentile(returns, (1 - confidence) * 100)

    def calculate_garch_volatility(self, returns: pd.Series) -> float:
        model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='normal')
        fitted_model = model.fit(disp='off')
        forecast = fitted_model.forecast(horizon=1)
        sigma_t = np.sqrt(forecast.variance.values[-1, :][0]) / 100
        return sigma_t

    def calculate_volatility_target_position(self, target_risk: float, sigma_t: float) -> float:
        return target_risk / sigma_t if sigma_t > 0 else 0.0

    def check_leverage(self, positions: pd.Series, prices: pd.Series) -> bool:
        notional = np.sum(np.abs(positions * prices))
        leverage = notional / self.initial_capital
        return leverage <= self.max_leverage

    def apply_stop_loss_take_profit(self, entry_price: float, current_price: float,
                                    stop_loss_pct: float, take_profit_pct: float) -> str:
        change_pct = (current_price - entry_price) / entry_price
        if change_pct <= -stop_loss_pct:
            return "stop_loss"
        elif change_pct >= take_profit_pct:
            return "take_profit"
        return "hold"

    def determine_position_size_forex(self, capital: float, risk_pct: float, stop_distance: float) -> float:
        dollar_risk = capital * risk_pct
        position_size = dollar_risk / stop_distance if stop_distance > 0 else 0
        return position_size

    def hedge_position(self, asset_corr: float, hedge_threshold: float) -> bool:
        return abs(asset_corr) >= hedge_threshold

    def adjust_positions(self, proposed_positions: pd.Series, prices: pd.Series,
                         volatility_dict: dict, target_risk: float) -> pd.Series:
        adjusted = proposed_positions.copy()
        for asset in proposed_positions.index:
            vol = volatility_dict.get(asset, None)
            if vol:
                adjusted[asset] = self.calculate_volatility_target_position(target_risk, vol)
        notional = np.sum(adjusted * prices)
        leverage = notional / self.initial_capital
        if leverage > self.max_leverage:
            adjusted *= self.max_leverage / leverage
        return adjusted
