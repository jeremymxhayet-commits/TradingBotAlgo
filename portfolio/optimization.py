import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.0):
        """
        Initialize the optimizer with historical returns.
        :param returns: A DataFrame where each column is an asset and each row is a return (percentage or log).
        :param risk_free_rate: The risk-free rate for Sharpe Ratio calculations (annualized).
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()

    def _portfolio_performance(self, weights):
        returns = np.dot(weights, self.mean_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (returns - self.risk_free_rate) / volatility if volatility > 0 else 0.0
        return returns, volatility, sharpe_ratio

    def maximize_sharpe_ratio(self):
        num_assets = len(self.mean_returns)
        args = (self.mean_returns, self.cov_matrix, self.risk_free_rate)

        def neg_sharpe(weights):
            r, vol, sharpe = self._portfolio_performance(weights)
            return -sharpe

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1. / num_assets, ]

        result = minimize(neg_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    def minimize_volatility(self, target_return: float):
        num_assets = len(self.mean_returns)

        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, self.mean_returns) - target_return}
        )
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1. / num_assets, ]

        result = minimize(portfolio_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    def risk_parity_weights(self):
        inv_vol = 1 / np.sqrt(np.diag(self.cov_matrix))
        weights = inv_vol / np.sum(inv_vol)
        return weights

    def kelly_criterion_weights(self):
        inv_cov = np.linalg.pinv(self.cov_matrix)
        kelly_weights = np.dot(inv_cov, self.mean_returns)
        kelly_weights /= np.sum(np.abs(kelly_weights))
        return kelly_weights

    def equal_weighted(self):
        num_assets = len(self.mean_returns)
        return np.ones(num_assets) / num_assets

    def summary(self, weights):
        r, vol, sharpe = self._portfolio_performance(weights)
        return {
            'expected_return': r,
            'volatility': vol,
            'sharpe_ratio': sharpe
        }
