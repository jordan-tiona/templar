"""Feature engineering: technical indicators + sentiment merge."""

import pandas as pd
import numpy as np
import pandas_ta as ta

from config.settings import DATA_PROCESSED, SEQUENCE_LENGTH


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a standard set of TA features to an OHLCV DataFrame."""
    df = df.copy()

    # Trend
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=21, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.macd(append=True)

    # Momentum
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(append=True)

    # Volatility
    df.ta.bbands(length=20, append=True)
    df.ta.atr(length=14, append=True)

    # Volume
    df.ta.obv(append=True)
    df.ta.vwap(append=True)

    # Returns
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)

    # Log volume
    df["log_volume"] = np.log1p(df["volume"])

    return df


def merge_sentiment(
    bars: pd.DataFrame,
    sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join sentiment onto bars, forward-fill gaps (weekends/holidays)."""
    if sentiment.empty:
        bars["sentiment_mean"] = 0.0
        bars["sentiment_count"] = 0
        return bars

    merged = bars.join(sentiment, how="left")
    merged["sentiment_mean"] = merged["sentiment_mean"].ffill().fillna(0.0)
    merged["sentiment_count"] = merged["sentiment_count"].fillna(0).astype(int)
    return merged


def build_features(
    symbol: str,
    bars: pd.DataFrame,
    sentiment: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """Full pipeline: TA → sentiment merge → dropna → save."""
    df = add_technical_indicators(bars)
    df = merge_sentiment(df, sentiment)
    df = df.dropna()

    if save:
        out = DATA_PROCESSED / f"{symbol}.parquet"
        df.to_parquet(out)

    return df
