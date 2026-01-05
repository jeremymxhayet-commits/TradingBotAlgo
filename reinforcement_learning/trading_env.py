import numpy as np
import pandas as pd

try:
    import gymnasium as gym
except Exception:  # pragma: no cover - fallback when gymnasium is missing
    import gym  # type: ignore


class TradingEnv(gym.Env):
    """
    Minimal trading environment with discrete actions:
    0 = sell/short, 1 = hold/flat, 2 = buy/long.
    Observation = [returns, sma, rsi, macd, vol].
    Reward = position * next_return - transaction_cost * turnover.
    """

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, transaction_cost: float = 0.0005):
        super().__init__()
        self.transaction_cost = float(transaction_cost)
        self.features, self.next_returns = self._build_features(df)
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.features.shape[1],), dtype=np.float32
        )
        self._idx = 0
        self._position = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = 0
        self._position = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = int(action)
        new_position = self._action_to_position(action)
        reward = (new_position * self.next_returns[self._idx]) - (
            self.transaction_cost * abs(new_position - self._position)
        )
        self._position = new_position
        self._idx += 1
        terminated = self._idx >= (len(self.features) - 1)
        obs = self._get_obs()
        return obs, float(reward), terminated, False, {}

    def _get_obs(self):
        return self.features[self._idx].astype(np.float32)

    @staticmethod
    def _action_to_position(action: int) -> int:
        if action == 0:
            return -1
        if action == 2:
            return 1
        return 0

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain, index=series.index).rolling(window=period).mean()
        avg_loss = pd.Series(loss, index=series.index).rolling(window=period).mean()
        rs = avg_gain / (avg_loss + 1e-6)
        return 100 - (100 / (1 + rs))

    def _build_features(self, df: pd.DataFrame):
        if df is None or df.empty or "Close" not in df.columns:
            raise ValueError("TradingEnv requires a non-empty DataFrame with Close prices.")
        data = df.copy()
        data["returns"] = data["Close"].pct_change()
        data["sma"] = data["Close"].rolling(window=10).mean()
        data["rsi"] = self._compute_rsi(data["Close"])
        data["macd"] = data["Close"].ewm(span=12).mean() - data["Close"].ewm(span=26).mean()
        data["vol"] = data["returns"].rolling(window=10).std()
        data = data.dropna()
        features = data[["returns", "sma", "rsi", "macd", "vol"]].values
        next_returns = data["returns"].shift(-1).fillna(0).values
        if len(features) < 2:
            raise ValueError("Not enough data to build RL features.")
        return features, next_returns
