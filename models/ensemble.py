"""Ensemble: combine TFT and XGBoost predictions into a single signal score."""

import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config.settings import BUY_THRESHOLD, SELL_THRESHOLD, TFT_ENSEMBLE_WEIGHT, XGB_ENSEMBLE_WEIGHT

log = logging.getLogger(__name__)


def combine(
    tft_pred: pd.Series,
    xgb_pred: pd.Series,
    tft_weight: float = TFT_ENSEMBLE_WEIGHT,
    xgb_weight: float = XGB_ENSEMBLE_WEIGHT,
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

    raw = tft_weight * df_normed["tft_n"] + xgb_weight * df_normed["xgb_n"]

    # Re-normalize blended score to [0, 1] so thresholds work as intended.
    # The weighted sum doesn't span [0,1] when the two models disagree on ranking.
    lo, hi = raw.min(), raw.max()
    score = (raw - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=raw.index)
    score.name = "ensemble_score"
    return score


def to_signals(score: pd.Series, max_side_frac: float = 0.2) -> pd.Series:
    """
    Map ensemble score to BUY / SHORT / HOLD.
    Shorts the bottom N symbols to match long count (market-neutral).

    Rails (added after 2026-07-08, when a degenerate model with massive score
    ties marked 144/148 symbols BUY and then overwrote 140 of them to SHORT):
    - degenerate score distributions (mostly ties) produce all-HOLD, no trades;
    - longs are capped at max_side_frac of the universe, highest scores first;
    - the short pool excludes the BUYs, so shorts can never overwrite longs.
    """
    signals = pd.Series("HOLD", index=score.index, name="signal")

    if score.nunique() < max(5, len(score) // 2):
        log.warning(
            "Degenerate ensemble scores (%d distinct across %d symbols) -- holding everything",
            score.nunique(), len(score),
        )
        return signals

    max_side = max(1, int(len(score) * max_side_frac))
    buys = score[score >= BUY_THRESHOLD].nlargest(max_side).index
    signals[buys] = "BUY"
    if len(buys) > 0:
        shorts = score.drop(buys).nsmallest(len(buys)).index
        signals[shorts] = "SHORT"
    return signals
