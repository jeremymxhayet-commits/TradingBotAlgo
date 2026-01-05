import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple

from xgboost import XGBClassifier
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit



XGB_MODEL_PATH = "models/trained/xgb_model.pkl"
LSTM_MODEL_PATH = "models/trained/lstm_model.keras"
SCALER_PATH = "models/trained/scaler.pkl"



def train_xgboost(X: pd.DataFrame, y: pd.Series) -> None:
    """Train and save an XGBoost classifier."""
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric="logloss",
    )
    model.fit(X, y)
    os.makedirs(os.path.dirname(XGB_MODEL_PATH), exist_ok=True)
    joblib.dump(model, XGB_MODEL_PATH)
    print("[INFO] XGBoost model trained and saved.")

def load_xgboost() -> XGBClassifier:
    """Load a saved XGBoost model."""
    return joblib.load(XGB_MODEL_PATH)

def predict_xgboost(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Predict using a trained XGBoost model."""
    return model.predict_proba(X)[:, 1]


def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """Build and compile an LSTM model."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_lstm(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
) -> None:
    """Train and save an LSTM model."""
    model = build_lstm_model((X.shape[1], X.shape[2]))

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1,
    )

    os.makedirs(os.path.dirname(LSTM_MODEL_PATH), exist_ok=True)
    model.save(LSTM_MODEL_PATH)
    print("[INFO] LSTM model trained and saved.")

def load_lstm() -> tf.keras.Model:
    """Load a saved LSTM model."""
    return tf.keras.models.load_model(LSTM_MODEL_PATH)

def predict_lstm(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    """Predict using a trained LSTM model."""
    return model.predict(X)



def scale_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize features and save scaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    return (
        pd.DataFrame(X_scaled, index=X.index, columns=X.columns),
        scaler,
    )

def load_scaler() -> StandardScaler:
    """Load a saved StandardScaler."""
    return joblib.load(SCALER_PATH)

def prepare_lstm_input(
    X: pd.DataFrame,
    window_size: int = 60,
) -> np.ndarray:
    """Convert flat features to 3D array for LSTM."""
    sequences = []
    for i in range(window_size, len(X)):
        sequences.append(X.iloc[i - window_size : i].values)
    return np.array(sequences)


def timeseries_cv_split(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
):
    """Generator for time-series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(X):
        yield (
            X.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx],
        )
