import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class RLTrader:
    """
    Lightweight wrapper for an RL agent with a heuristic fallback.
    Expects state = [returns, sma, rsi, macd, vol].
    """

    def __init__(self, model_path: str, action_map: Optional[dict] = None):
        self.model_path = model_path
        self.action_map = action_map or {0: "sell", 1: "hold", 2: "buy"}
        self.model = self._load_model(model_path)

    def predict_action(self, state) -> str:
        if state is None:
            return "hold"
        state_arr = np.asarray(state, dtype=float).reshape(1, -1)
        if self.model is None:
            return self._heuristic_action(state_arr[0])
        try:
            action, _ = self.model.predict(state_arr, deterministic=True)
            if isinstance(action, (list, tuple, np.ndarray)):
                action = int(np.asarray(action).flatten()[0])
            return self.action_map.get(int(action), "hold")
        except Exception as exc:
            logger.warning("RL prediction failed, using heuristic: %s", exc)
            return self._heuristic_action(state_arr[0])

    def _load_model(self, path: str):
        try:
            from stable_baselines3 import PPO  # type: ignore
        except Exception as exc:
            logger.warning("stable_baselines3 unavailable: %s", exc)
            return None
        try:
            return PPO.load(path)
        except Exception as exc:
            logger.warning("Failed to load RL model at %s: %s", path, exc)
            return None

    @staticmethod
    def _heuristic_action(state: np.ndarray) -> str:
        if state.size < 5:
            return "hold"
        returns, sma, rsi, macd, _vol = state[:5]
        if returns > 0 and macd > 0 and rsi < 70:
            return "buy"
        if returns < 0 and macd < 0 and rsi > 30:
            return "sell"
        return "hold"
