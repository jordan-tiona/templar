"""Ensemble: combine TFT and XGBoost predictions into a single signal score."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config.settings import BUY_THRESHOLD, SELL_THRESHOLD


def combine(
    tft_pred: pd.Series,
    xgb_pred: pd.Series,
    tft_weight: float = 0.6,
    xgb_weight: float = 0.4,
) -> pd.Series:
    """
    Blend predictions into a single score in [0, 1].
    Higher score = stronger buy signal; lower = stronger sell signal.
    """
    df = pd.DataFrame({"tft": tft_pred, "xgb": xgb_pred}).dropna()

    # Normalize each model's raw return predictions to [0, 1]
    scaler = MinMaxScaler()
    normed = scaler.fit_transform(df.values)
    df_normed = pd.DataFrame(normed, index=df.index, columns=["tft_n", "xgb_n"])

    score = tft_weight * df_normed["tft_n"] + xgb_weight * df_normed["xgb_n"]
    score.name = "ensemble_score"
    return score


def to_signals(score: pd.Series) -> pd.Series:
    """Map ensemble score to BUY / SELL / HOLD."""
    signals = pd.Series("HOLD", index=score.index, name="signal")
    signals[score >= BUY_THRESHOLD] = "BUY"
    signals[score <= SELL_THRESHOLD] = "SELL"
    return signals
