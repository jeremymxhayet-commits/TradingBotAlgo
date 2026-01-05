import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Tuple, Literal
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from arch import arch_model



def fit_arima(series: pd.Series, order: Tuple[int, int, int]) -> ARIMA:
    """Fit ARIMA(p,d,q) model to a univariate series."""
    model = ARIMA(series, order=order)
    result = model.fit()
    return result

def forecast_arima(model: ARIMA, steps: int = 1) -> np.ndarray:
    """Forecast future values using a fitted ARIMA model."""
    return model.forecast(steps=steps)

def fit_var(df: pd.DataFrame, lags: int = 1):
    """Fit Vector AutoRegression (VAR) model to multiple time series."""
    model = VAR(df)
    result = model.fit(lags)
    return result

def forecast_var(model, steps: int = 1) -> pd.DataFrame:
    """Forecast future values using a fitted VAR model."""
    return model.forecast(model.y, steps=steps)

def fit_garch(series: pd.Series, p: int = 1, q: int = 1):
    """Fit GARCH(p,q) model to estimate volatility."""
    model = arch_model(series, vol='Garch', p=p, q=q)
    result = model.fit(disp='off')
    return result

def forecast_garch(model, steps: int = 1) -> np.ndarray:
    """Forecast volatility using a fitted GARCH model."""
    return np.sqrt(model.forecast(horizon=steps).variance.values[-1])



def compute_ff_regression(returns: pd.Series, ff_factors: pd.DataFrame) -> pd.Series:
    """
    Run regression of asset returns on Fama-French factors.
    Returns estimated betas.
    """
    import statsmodels.api as sm
    X = sm.add_constant(ff_factors)
    model = sm.OLS(returns, X).fit()
    return model.params



def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: Literal['call', 'put']) -> float:
    """Calculate Black-Scholes price for European call or put option."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price

def black_scholes_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: Literal['call', 'put']) -> float:
    """Calculate Black-Scholes Delta."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    elif option_type == 'put':
        return -norm.cdf(-d1)



def markowitz_weights(expected_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float = 0.0) -> np.ndarray:
    """
    Compute portfolio weights maximizing Sharpe ratio.
    """
    from scipy.optimize import minimize

    n = len(expected_returns)

    def neg_sharpe(weights):
        port_return = np.dot(weights, expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return - (port_return - risk_free_rate) / port_vol

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = np.ones(n) / n

    result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x
