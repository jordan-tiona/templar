"""Shared dataset preparation for TFT and XGBoost."""

import pandas as pd
import numpy as np
from config.settings import PREDICTION_HORIZON, SEQUENCE_LENGTH


# Features used by both models
STATIC_CATEGORICALS = ["symbol"]

TIME_VARYING_UNKNOWN = [
    # Price/volume
    "open", "high", "low", "close", "volume", "log_volume",
    # Returns
    "return_1d", "return_5d", "return_20d",
    # Trend
    "EMA_9", "EMA_21", "EMA_50",
    "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
    # Momentum
    "RSI_14", "STOCHk_14_3_3", "STOCHd_14_3_3",
    # Volatility
    "BBL_20_2_0_2_0", "BBM_20_2_0_2_0", "BBU_20_2_0_2_0", "BBB_20_2_0_2_0", "BBP_20_2_0_2_0",
    "ATRr_14",
    # Volume
    "OBV", "VWAP_D",
    # Sentiment
    "sentiment_mean", "sentiment_count",
]

TARGET = "target_return"


def add_target(df: pd.DataFrame, horizon: int = PREDICTION_HORIZON) -> pd.DataFrame:
    """Add forward return as prediction target, drop rows where target is unknown."""
    df = df.copy()
    df[TARGET] = df["close"].shift(-horizon) / df["close"] - 1
    df = df.dropna(subset=[TARGET])
    return df


def prepare_combined(
    feature_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge all symbols into a single long-format DataFrame suitable for TFT.
    Adds target, time_idx (integer), and symbol column.
    """
    frames = []
    for sym, df in feature_dfs.items():
        df = add_target(df)
        df = df.copy()
        df["symbol"] = sym
        df["time_idx"] = np.arange(len(df))
        # Keep only columns both models need
        cols = [TARGET, "symbol", "time_idx"] + [
            c for c in TIME_VARYING_UNKNOWN if c in df.columns
        ]
        frames.append(df[cols])

    return pd.concat(frames, ignore_index=False)


def train_val_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time_idx within each symbol (no lookahead leakage)."""
    splits = []
    for sym, grp in df.groupby("symbol"):
        n = len(grp)
        t1 = int(n * train_frac)
        t2 = int(n * (train_frac + val_frac))
        train = grp.iloc[:t1].copy()
        train["split"] = "train"
        val = grp.iloc[t1:t2].copy()
        val["split"] = "val"
        test = grp.iloc[t2:].copy()
        test["split"] = "test"
        splits.extend([train, val, test])

    full = pd.concat(splits)
    return (
        full[full["split"] == "train"].drop(columns="split"),
        full[full["split"] == "val"].drop(columns="split"),
        full[full["split"] == "test"].drop(columns="split"),
    )
