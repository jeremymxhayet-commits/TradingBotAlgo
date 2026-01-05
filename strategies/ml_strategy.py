from strategies.base_strategy import Strategy
import logging
import os
from collections import deque
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

try:
    from models.rl_trader import RLTrader
except ImportError:
    RLTrader = None

load_model = tf.keras.models.load_model

logger = logging.getLogger(__name__)


class MLStrategy(Strategy):
    """
    Machine Learning Strategy using:
    - XGBoost for classification
    - LSTM for sequence prediction
    - RL agent for policy-based decision making
    """

    def __init__(
        self,
        symbols,
        xgb_model_path=None,
        lstm_model_path=None,
        rl_model_path=None,
        threshold=0.55,
        xgb_params=None,
        online_learning=False,
        online_window=252,
        online_interval=5,
        online_min_samples=200,
        max_hold_bars=60,
        momentum_lookback=5,
        risk_per_trade=0.01,
        max_position_per_asset=0.2,
        volatility_target=0.015,
        vol_lookback=20,
        atr_lookback=14,
        trend_lookback=20,
        min_trend_strength=0.0005,
        min_volatility=0.003,
        no_trade_prob_band=0.05,
        min_edge_to_cost=1.5,
        max_volume_participation=0.05,
        min_trade_qty=1,
        atr_stop_mult=2.0,
        atr_take_profit_mult=3.5,
        atr_trail_mult=2.5,
        ensemble_lookback=50,
        ensemble_min_weight=0.1,
        ensemble_min_score=0.15,
        cost_rate=0.0,
        allow_short=False,
        symbol_asset_map=None,
        class_thresholds=None,
        xgb_class_model_paths=None,
    ):
        super().__init__()
        self.symbols = symbols
        self.threshold = threshold
        self.xgb_params = xgb_params or {}
        self.online_learning = online_learning
        self.online_window = max(50, int(online_window))
        self.online_interval = max(1, int(online_interval))
        self.online_min_samples = max(50, int(online_min_samples))
        self._online_step = 0
        self.max_hold_bars = max(1, int(max_hold_bars))
        self.momentum_lookback = max(2, int(momentum_lookback))
        self.hold_bars = {symbol: 0 for symbol in symbols}
        self.symbol_asset_map = symbol_asset_map or {}
        self.class_thresholds = class_thresholds or {}
        self.risk_per_trade = float(risk_per_trade)
        self.max_position_per_asset = float(max_position_per_asset)
        self.volatility_target = float(volatility_target)
        self.vol_lookback = max(5, int(vol_lookback))
        self.atr_lookback = max(5, int(atr_lookback))
        self.trend_lookback = max(5, int(trend_lookback))
        self.min_trend_strength = float(min_trend_strength)
        self.min_volatility = float(min_volatility)
        self.no_trade_prob_band = float(no_trade_prob_band)
        self.min_edge_to_cost = float(min_edge_to_cost)
        self.max_volume_participation = float(max_volume_participation)
        self.min_trade_qty = max(1, int(min_trade_qty))
        self.atr_stop_mult = float(atr_stop_mult)
        self.atr_take_profit_mult = float(atr_take_profit_mult)
        self.atr_trail_mult = float(atr_trail_mult)
        self.ensemble_lookback = max(10, int(ensemble_lookback))
        self.ensemble_min_weight = float(ensemble_min_weight)
        self.ensemble_min_score = float(ensemble_min_score)
        self.cost_rate = float(cost_rate)
        self.allow_short = bool(allow_short)
        self.entry_prices = {}
        self.peak_prices = {}
        self.trough_prices = {}
        self.model_performance = {
            "xgb": deque(maxlen=self.ensemble_lookback),
            "lstm": deque(maxlen=self.ensemble_lookback),
            "rl": deque(maxlen=self.ensemble_lookback),
        }
        self.last_model_predictions = {}

        self.xgb_model = self._load_xgb_model(xgb_model_path)
        self.lstm_model = self._load_lstm_model(lstm_model_path)
        self.rl_agent = self._load_rl_agent(rl_model_path)
        self.xgb_class_models = self._load_xgb_class_models(xgb_class_model_paths or {})

    def compute_signals(self, data: dict) -> list:
        signals = []
        for symbol, df in data.items():
            df = df.dropna()
            if len(df) < 50 or "Close" not in df.columns:
                continue

            features = self._generate_features(df)
            aux = self._compute_aux_features(df)
            if features.empty:
                continue
            asset_class = self.symbol_asset_map.get(symbol)
            threshold = self.class_thresholds.get(asset_class, self.threshold)
            xgb_model = self.xgb_class_models.get(asset_class) or self.xgb_model
            heuristic_decision = self._heuristic_decision(df, features)
            position = self.get_position(symbol)
            if position > 0 and self._should_exit_atr(df, aux, symbol):
                signals.append(
                    {
                        "symbol": symbol,
                        "action": "sell",
                        "timestamp": df.index[-1],
                        "price": df["Close"].iloc[-1],
                        "volume": df["Volume"].iloc[-1] if "Volume" in df.columns else None,
                        "confidence": 1.0,
                        "expected_edge": 0.0,
                        "atr": aux.get("atr"),
                        "volatility": aux.get("volatility"),
                    }
                )
                continue
            if position == 0 and not self._regime_allows_trade(aux):
                signals.append(
                    {
                        "symbol": symbol,
                        "action": "hold",
                        "timestamp": df.index[-1],
                        "price": df["Close"].iloc[-1],
                        "volume": df["Volume"].iloc[-1] if "Volume" in df.columns else None,
                        "confidence": 0.0,
                        "expected_edge": 0.0,
                        "atr": aux.get("atr"),
                        "volatility": aux.get("volatility"),
                    }
                )
                continue
            if not self._models_available():
                decision = heuristic_decision
                if position != 0 and self._should_exit_momentum(df, symbol):
                    decision = "sell" if position > 0 else "buy"
                signals.append(
                    {
                        "symbol": symbol,
                        "action": decision,
                        "timestamp": df.index[-1],
                        "price": df["Close"].iloc[-1],
                        "volume": df["Volume"].iloc[-1] if "Volume" in df.columns else None,
                        "confidence": 0.0,
                        "expected_edge": 0.0,
                        "atr": aux.get("atr"),
                        "volatility": aux.get("volatility"),
                    }
                )
                continue

            latest_features = features.iloc[-1:].values

            xgb_prob = xgb_model.predict_proba(latest_features)[0, 1] if xgb_model else 0.5

            if self.lstm_model and len(features) >= 30:
                sequence = features.iloc[-30:].values.reshape((1, 30, features.shape[1]))
                lstm_pred = self.lstm_model.predict(sequence, verbose=0)[0, 0]
            else:
                lstm_pred = df["Close"].iloc[-1]

            state = features.iloc[-1:].values[0]
            action = self.rl_agent.predict_action(state) if self.rl_agent else "hold"

            xgb_vote = self._xgb_vote(xgb_prob, threshold)
            lstm_vote = self._lstm_vote(lstm_pred, df["Close"].iloc[-1], aux.get("volatility", 0.0))
            rl_vote = self._rl_vote(action)
            weights = self._ensemble_weights()
            score = (
                weights.get("xgb", 0.0) * xgb_vote
                + weights.get("lstm", 0.0) * lstm_vote
                + weights.get("rl", 0.0) * rl_vote
            )
            decision = self._score_to_action(score)

            if decision == "hold":
                decision = heuristic_decision
                if position != 0 and self._should_exit_momentum(df, symbol):
                    decision = "sell" if position > 0 else "buy"
            else:
                if position > 0:
                    exit_signal = (
                        xgb_prob < 1 - threshold
                        or lstm_pred < df["Close"].iloc[-1]
                        or heuristic_decision == "sell"
                        or self._should_exit_momentum(df, symbol)
                    )
                    if exit_signal:
                        decision = "sell"
                elif position < 0:
                    exit_signal = (
                        xgb_prob > threshold
                        or lstm_pred > df["Close"].iloc[-1]
                        or heuristic_decision == "buy"
                        or self._should_exit_momentum(df, symbol)
                    )
                    if exit_signal:
                        decision = "buy"

            self.last_model_predictions[symbol] = {
                "xgb": xgb_vote,
                "lstm": lstm_vote,
                "rl": rl_vote,
            }
            confidence = min(1.0, abs(score))
            expected_edge = confidence * float(aux.get("volatility", 0.0))

            signals.append(
                {
                    "symbol": symbol,
                    "action": decision,
                    "timestamp": df.index[-1],
                    "price": df["Close"].iloc[-1],
                    "volume": df["Volume"].iloc[-1] if "Volume" in df.columns else None,
                    "confidence": confidence,
                    "expected_edge": expected_edge,
                    "atr": aux.get("atr"),
                    "volatility": aux.get("volatility"),
                }
            )

        return signals

    def generate_orders(self, signals: list) -> list:
        orders = []
        for signal in signals:
            symbol = signal["symbol"]
            action = signal["action"]
            position = self.get_position(symbol)
            size = self._position_size(signal)
            if size <= 0:
                continue
            if self._skip_due_to_costs(signal):
                continue

            if action == "buy" and position <= 0:
                qty = abs(position) + size
                orders.append({"symbol": symbol, "side": "buy", "qty": qty, "order_type": "market"})
            elif action == "buy" and position > 0:
                orders.append({"symbol": symbol, "side": "buy", "qty": size, "order_type": "market"})
            elif action == "sell" and position >= 0:
                if not self.allow_short and position == 0:
                    continue
                qty = abs(position) + size
                orders.append({"symbol": symbol, "side": "sell", "qty": qty, "order_type": "market"})
            elif action == "sell" and position < 0:
                orders.append({"symbol": symbol, "side": "sell", "qty": size, "order_type": "market"})

        return orders

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["returns"] = df["Close"].pct_change()
        df["sma"] = df["Close"].rolling(window=10).mean()
        df["rsi"] = self._compute_rsi(df["Close"])
        df["macd"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
        df["vol"] = df["returns"].rolling(window=10).std()
        df = df.dropna()
        return df[["returns", "sma", "rsi", "macd", "vol"]]

    def on_bar(self, data: dict, current_data: dict, current_time: pd.Timestamp):
        for symbol in self.symbols:
            if self.get_position(symbol) != 0:
                self.hold_bars[symbol] = self.hold_bars.get(symbol, 0) + 1
            else:
                self.hold_bars[symbol] = 0
        self._update_model_performance(data)
        if not self.online_learning:
            return
        self._online_step += 1
        if self._online_step % self.online_interval != 0:
            return
        try:
            import xgboost as xgb  # type: ignore
        except Exception:
            logger.warning("Online learning skipped: xgboost not available.")
            return
        if self.xgb_class_models:
            class_datasets = self._build_online_datasets_by_class(data)
            for asset_class, (X, y) in class_datasets.items():
                if X.size == 0 or len(y) < self.online_min_samples:
                    continue
                model = xgb.XGBClassifier(**self.xgb_params)
                model.fit(X, y)
                self.xgb_class_models[asset_class] = model
        else:
            X, y = self._build_online_dataset(data)
            if X.size == 0 or len(y) < self.online_min_samples:
                return
            model = xgb.XGBClassifier(**self.xgb_params)
            model.fit(X, y)
            self.xgb_model = model

    def _build_online_dataset(self, data: dict):
        features_list = []
        target_list = []
        for _, df in data.items():
            df = df.dropna()
            if df.empty or "Close" not in df.columns:
                continue
            features = self._generate_features(df)
            if len(features) < 2:
                continue
            close = df.loc[features.index, "Close"]
            target = (close.pct_change().shift(-1) > 0).astype(int)
            features = features.iloc[:-1]
            target = target.iloc[:-1]
            if features.empty or target.empty:
                continue
            window = min(len(features), self.online_window)
            features_list.append(features.iloc[-window:].values)
            target_list.append(target.iloc[-window:].values)
        if not features_list:
            return np.empty((0, 5)), np.empty((0,), dtype=int)
        X = np.vstack(features_list)
        y = np.concatenate(target_list)
        return X, y

    def _build_online_datasets_by_class(self, data: dict):
        datasets = {}
        for symbol, df in data.items():
            asset_class = self.symbol_asset_map.get(symbol)
            if asset_class is None:
                continue
            df = df.dropna()
            if df.empty or "Close" not in df.columns:
                continue
            features = self._generate_features(df)
            if len(features) < 2:
                continue
            close = df.loc[features.index, "Close"]
            target = (close.pct_change().shift(-1) > 0).astype(int)
            features = features.iloc[:-1]
            target = target.iloc[:-1]
            if features.empty or target.empty:
                continue
            window = min(len(features), self.online_window)
            X = features.iloc[-window:].values
            y = target.iloc[-window:].values
            if asset_class not in datasets:
                datasets[asset_class] = ([], [])
            datasets[asset_class][0].append(X)
            datasets[asset_class][1].append(y)
        stacked = {}
        for asset_class, (X_list, y_list) in datasets.items():
            if not X_list:
                continue
            X = np.vstack(X_list)
            y = np.concatenate(y_list)
            stacked[asset_class] = (X, y)
        return stacked

    def _models_available(self) -> bool:
        return any(model is not None for model in (self.xgb_model, self.lstm_model, self.rl_agent))

    def model_status(self) -> dict:
        return {
            "xgb": self.xgb_model is not None,
            "lstm": self.lstm_model is not None,
            "rl": self.rl_agent is not None,
            "xgb_class_models": sorted(self.xgb_class_models.keys()),
        }

    def _heuristic_decision(self, df: pd.DataFrame, features: pd.DataFrame) -> str:
        latest = features.iloc[-1]
        price = df["Close"].iloc[-1]
        sma = latest["sma"]
        rsi = latest["rsi"]
        ret = latest["returns"]
        if price > sma and ret > 0 and rsi < 70:
            return "buy"
        if price < sma and ret < 0 and rsi > 30:
            return "sell"
        return "hold"

    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain, index=series.index).rolling(window=period).mean()
        avg_loss = pd.Series(loss, index=series.index).rolling(window=period).mean()
        rs = avg_gain / (avg_loss + 1e-6)
        return 100 - (100 / (1 + rs))

    def _fixed_size(self) -> int:
        return 100

    def _should_exit_momentum(self, df: pd.DataFrame, symbol: str) -> bool:
        hold_bars = self.hold_bars.get(symbol, 0)
        if hold_bars < self.max_hold_bars:
            return False
        if len(df) < self.momentum_lookback + 1:
            return False
        returns = df["Close"].pct_change().dropna()
        if len(returns) < self.momentum_lookback:
            return False
        recent_momentum = returns.iloc[-self.momentum_lookback:].mean()
        position = self.get_position(symbol)
        if position > 0:
            return recent_momentum < 0
        if position < 0:
            return recent_momentum > 0
        return False

    def _should_exit_atr(self, df: pd.DataFrame, aux: dict, symbol: str) -> bool:
        if df.empty or "Close" not in df.columns:
            return False
        price = df["Close"].iloc[-1]
        entry = self.entry_prices.get(symbol)
        if entry is None:
            return False
        atr = aux.get("atr")
        if not atr or atr <= 0:
            return False
        position = self.get_position(symbol)
        if position > 0:
            peak = self.peak_prices.get(symbol, entry)
            if price > peak:
                self.peak_prices[symbol] = price
                peak = price
            if price <= entry - (self.atr_stop_mult * atr):
                return True
            if price >= entry + (self.atr_take_profit_mult * atr):
                return True
            if price <= peak - (self.atr_trail_mult * atr):
                return True
        elif position < 0 and self.allow_short:
            trough = self.trough_prices.get(symbol, entry)
            if price < trough:
                self.trough_prices[symbol] = price
                trough = price
            if price >= entry + (self.atr_stop_mult * atr):
                return True
            if price <= entry - (self.atr_take_profit_mult * atr):
                return True
            if price >= trough + (self.atr_trail_mult * atr):
                return True
        return False

    def _load_xgb_model(self, path):
        if not path:
            return None
        if not os.path.exists(path):
            logger.warning("XGB model path not found: %s", path)
            return None
        return joblib.load(path)

    def _load_lstm_model(self, path):
        if not path:
            return None
        if not os.path.exists(path):
            logger.warning("LSTM model path not found: %s", path)
            return None
        return load_model(path)

    def _load_rl_agent(self, path):
        if not path or not RLTrader:
            return None
        if not os.path.exists(path):
            logger.warning("RL model path not found: %s", path)
            return None
        return RLTrader(path)

    def _load_xgb_class_models(self, paths):
        models = {}
        for asset_class, path in paths.items():
            if not path:
                continue
            if not os.path.exists(path):
                continue
            try:
                models[asset_class] = joblib.load(path)
            except Exception as exc:
                logger.warning("Failed to load XGB model for %s: %s", asset_class, exc)
        return models

    def update_position(self, symbol: str, qty: int, price: float):
        super().update_position(symbol, qty, price)
        position = self.get_position(symbol)
        if position > 0:
            self.entry_prices[symbol] = self.entry_prices.get(symbol, price)
            self.peak_prices[symbol] = max(self.peak_prices.get(symbol, price), price)
            self.trough_prices.pop(symbol, None)
        elif position < 0:
            self.entry_prices[symbol] = self.entry_prices.get(symbol, price)
            self.trough_prices[symbol] = min(self.trough_prices.get(symbol, price), price)
            self.peak_prices.pop(symbol, None)
        else:
            self.entry_prices.pop(symbol, None)
            self.peak_prices.pop(symbol, None)
            self.trough_prices.pop(symbol, None)

    def _compute_aux_features(self, df: pd.DataFrame) -> dict:
        if df is None or df.empty or "Close" not in df.columns:
            return {}
        closes = df["Close"]
        highs = df["High"] if "High" in df.columns else closes
        lows = df["Low"] if "Low" in df.columns else closes
        atr = self._compute_atr(highs, lows, closes, self.atr_lookback)
        vol = closes.pct_change().rolling(window=self.vol_lookback).std().iloc[-1]
        trend = self._trend_strength(closes, self.trend_lookback)
        return {"atr": atr, "volatility": float(vol) if not np.isnan(vol) else 0.0, "trend": trend}

    def _compute_atr(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, lookback: int) -> float:
        if len(closes) < lookback + 1:
            return 0.0
        prev_close = closes.shift(1)
        tr = pd.concat(
            [
                (highs - lows).abs(),
                (highs - prev_close).abs(),
                (lows - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=lookback).mean().iloc[-1]
        return float(atr) if not np.isnan(atr) else 0.0

    def _trend_strength(self, closes: pd.Series, lookback: int) -> float:
        if len(closes) < lookback:
            return 0.0
        window = closes.iloc[-lookback:].values
        x = np.arange(len(window))
        slope = np.polyfit(x, window, 1)[0]
        last_price = window[-1] if window[-1] != 0 else 1.0
        return float(slope / last_price)

    def _regime_allows_trade(self, aux: dict) -> bool:
        vol = float(aux.get("volatility", 0.0))
        trend = float(aux.get("trend", 0.0))
        if vol < self.min_volatility:
            return False
        if abs(trend) < self.min_trend_strength:
            return False
        return True

    def _xgb_vote(self, xgb_prob: float, threshold: float) -> int:
        if abs(xgb_prob - 0.5) < self.no_trade_prob_band:
            return 0
        if xgb_prob > threshold:
            return 1
        if xgb_prob < 1 - threshold:
            return -1
        return 0

    def _lstm_vote(self, lstm_pred: float, price: float, volatility: float) -> int:
        if price <= 0:
            return 0
        move = (lstm_pred - price) / price
        if abs(move) < max(volatility, 1e-6):
            return 0
        return 1 if move > 0 else -1

    @staticmethod
    def _rl_vote(action: str) -> int:
        if action == "buy":
            return 1
        if action == "sell":
            return -1
        return 0

    def _ensemble_weights(self) -> dict:
        weights = {}
        for key, perf in self.model_performance.items():
            if len(perf) < 10:
                weights[key] = 1.0
            else:
                acc = float(np.mean(perf))
                weights[key] = max(self.ensemble_min_weight, acc)
        total = sum(weights.values())
        if total <= 0:
            return {"xgb": 1 / 3, "lstm": 1 / 3, "rl": 1 / 3}
        return {k: v / total for k, v in weights.items()}

    def _score_to_action(self, score: float) -> str:
        if abs(score) < self.ensemble_min_score:
            return "hold"
        return "buy" if score > 0 else "sell"

    def _position_size(self, signal: dict) -> int:
        price = float(signal.get("price", 0.0) or 0.0)
        if price <= 0:
            return 0
        atr = float(signal.get("atr") or 0.0)
        confidence = float(signal.get("confidence") or 0.0)
        volatility = float(signal.get("volatility") or 0.0)
        risk_budget = self.get_cash() * self.risk_per_trade
        stop_distance = atr * self.atr_stop_mult if atr > 0 else price * 0.02
        qty_risk = risk_budget / stop_distance if stop_distance > 0 else 0.0
        if volatility > 0:
            target_fraction = min(self.max_position_per_asset, self.volatility_target / volatility)
        else:
            target_fraction = self.max_position_per_asset
        max_qty = (self.get_cash() * target_fraction) / price
        qty = min(qty_risk, max_qty)
        qty *= max(0.2, confidence)
        volume = signal.get("volume")
        if volume:
            qty = min(qty, float(volume) * self.max_volume_participation)
        qty = int(qty)
        if qty < self.min_trade_qty:
            return 0
        return max(qty, 0)

    def _skip_due_to_costs(self, signal: dict) -> bool:
        expected_edge = float(signal.get("expected_edge") or 0.0)
        if expected_edge <= 0:
            return False
        return expected_edge < (self.cost_rate * self.min_edge_to_cost)

    def _update_model_performance(self, data: dict):
        for symbol, df in data.items():
            if df is None or df.empty or "Close" not in df.columns:
                continue
            if len(df) < 2:
                continue
            last_pred = self.last_model_predictions.get(symbol)
            if not last_pred:
                continue
            ret = df["Close"].pct_change().iloc[-1]
            actual = 1 if ret > 0 else -1 if ret < 0 else 0
            for model_key, pred in last_pred.items():
                if pred == 0 or actual == 0:
                    continue
                self.model_performance[model_key].append(1 if pred == actual else 0)
