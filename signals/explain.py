"""
Explain why the model generated a BUY/SELL signal for a given symbol.

Usage:
    python -m signals.explain AAPL
    python -m signals.explain AAPL MSFT

Or from Python:
    from signals.explain import explain_signal
    explain_signal("AAPL")
"""

import sys
import logging
import textwrap
import xgboost as xgb
import numpy as np
import pandas as pd

from config.settings import WATCHLIST, SEQUENCE_LENGTH, PREDICTION_HORIZON, BUY_THRESHOLD, SELL_THRESHOLD
from data import fetch_bars, fetch_news_sentiment, build_features
from models import tft as tft_model, xgb as xgb_model, ensemble
from models.dataset import prepare_combined, add_target, TIME_VARYING_UNKNOWN

log = logging.getLogger(__name__)


def _signal_label(score: float) -> str:
    if score >= BUY_THRESHOLD:
        return "BUY"
    if score <= SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def _ema_trend(row: pd.Series) -> str:
    e9 = row.get("EMA_9", float("nan"))
    e21 = row.get("EMA_21", float("nan"))
    e50 = row.get("EMA_50", float("nan"))
    if any(np.isnan(v) for v in [e9, e21, e50]):
        return "n/a"
    if e9 > e21 > e50:
        return "bullish (9 > 21 > 50)"
    if e9 < e21 < e50:
        return "bearish (9 < 21 < 50)"
    return "mixed"


def _rsi_label(rsi: float) -> str:
    if np.isnan(rsi):
        return "n/a"
    if rsi >= 70:
        return "overbought"
    if rsi <= 30:
        return "oversold"
    if rsi >= 55:
        return "bullish"
    if rsi <= 45:
        return "bearish"
    return "neutral"


def _fmt(val, fmt=".2f", suffix=""):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    return f"{val:{fmt}}{suffix}"


def explain_signal(symbol: str, lookback_days: int = SEQUENCE_LENGTH + PREDICTION_HORIZON + 100) -> None:
    """Print a human-readable explanation of the latest signal for symbol."""

    # --- Data ---
    bars = fetch_bars([symbol])
    if symbol not in bars or bars[symbol].empty:
        print(f"No bar data available for {symbol}.")
        return

    full_bars = bars[symbol]
    if len(full_bars) < 400:
        print(f"{symbol}: insufficient history ({len(full_bars)} bars, need 400+).")
        return

    df_raw = full_bars.tail(lookback_days)
    sentiment = fetch_news_sentiment(symbol)
    df = build_features(symbol, df_raw, sentiment, save=False)
    df_with_target = add_target(df)

    if len(df_with_target) < SEQUENCE_LENGTH + PREDICTION_HORIZON + 20:
        print(f"{symbol}: too few rows after target construction ({len(df_with_target)}).")
        return

    latest_row = df_with_target.iloc[-1]
    signal_date = df_with_target.index[-1]

    # --- Model predictions ---
    tft = tft_model.load()
    xgb_m, scaler = xgb_model.load()

    df_with_target["symbol"] = symbol
    df_with_target["time_idx"] = range(len(df_with_target))

    try:
        tft_pred_val = tft_model.predict(tft, df_with_target, df_with_target).iloc[-1]
    except Exception as e:
        log.warning("TFT prediction failed: %s", e)
        tft_pred_val = float("nan")

    feature_cols = [c for c in TIME_VARYING_UNKNOWN if c in df_with_target.columns]
    X_raw = df_with_target.iloc[[-1]][feature_cols].values
    X_scaled = scaler.transform(X_raw)
    xgb_pred_val = float(xgb_m.predict(X_scaled)[0])

    tft_s = pd.Series({symbol: tft_pred_val})
    xgb_s = pd.Series({symbol: xgb_pred_val})
    score_val = float(ensemble.combine(tft_s, xgb_s).iloc[0])
    signal = _signal_label(score_val)

    # --- XGBoost SHAP contributions ---
    booster = xgb_m.get_booster()
    dmatrix = xgb.DMatrix(X_scaled, feature_names=feature_cols)
    contribs = booster.predict(dmatrix, pred_contribs=True)[0]  # shape: n_features + 1
    bias = contribs[-1]
    shap_vals = pd.Series(contribs[:-1], index=feature_cols)
    top_shap = shap_vals.reindex(shap_vals.abs().sort_values(ascending=False).index)

    # --- Print report ---
    w = 60
    bar = "=" * w

    print(f"\n{bar}")
    print(f"  Signal Explanation: {symbol}  —  {signal_date.date()}")
    print(bar)
    print(f"  Signal: {signal:4s}  |  Ensemble: {score_val:.3f}  |  "
          f"TFT pred: {tft_pred_val:+.2%}  |  XGB pred: {xgb_pred_val:+.2%}")
    print(f"  Thresholds: BUY >= {BUY_THRESHOLD}  |  SELL <= {SELL_THRESHOLD}")
    print()

    # Snapshot
    r = latest_row
    print("  Market Snapshot")
    print("  " + "-" * (w - 2))
    fields = [
        ("Close",       f"${r.get('close', float('nan')):.2f}"),
        ("Volume",      f"{r.get('volume', float('nan')):,.0f}"),
        ("Return 1d",   _fmt(r.get("return_1d"), ".2%")),
        ("Return 5d",   _fmt(r.get("return_5d"), ".2%")),
        ("Return 20d",  _fmt(r.get("return_20d"), ".2%")),
        ("EMA 9",       _fmt(r.get("EMA_9"), ".2f", "")),
        ("EMA 21",      _fmt(r.get("EMA_21"), ".2f", "")),
        ("EMA 50",      _fmt(r.get("EMA_50"), ".2f", "")),
        ("EMA trend",   _ema_trend(r)),
        ("RSI-14",      f"{_fmt(r.get('RSI_14'), '.1f')}  ({_rsi_label(r.get('RSI_14', float('nan')))})"),
        ("Stoch %K",    _fmt(r.get("STOCHk_14_3_3"), ".1f")),
        ("Stoch %D",    _fmt(r.get("STOCHd_14_3_3"), ".1f")),
        ("MACD",        _fmt(r.get("MACD_12_26_9"), ".3f")),
        ("MACD hist",   _fmt(r.get("MACDh_12_26_9"), ".3f")),
        ("BB %B",       _fmt(r.get("BBP_20_2_0_2_0"), ".2f") + "  (0=lower, 1=upper band)"),
        ("BB width",    _fmt(r.get("BBB_20_2_0_2_0"), ".2f")),
        ("ATR-14",      _fmt(r.get("ATRr_14"), ".3f")),
        ("Sentiment",   f"{_fmt(r.get('sentiment_mean'), '.3f')}  "
                        f"({int(r.get('sentiment_count', 0))} articles)"),
    ]
    for label, val in fields:
        print(f"  {label:<14} {val}")

    print()
    print("  XGBoost Feature Contributions (SHAP)")
    print("  " + "-" * (w - 2))
    print(f"  {'Feature':<28} {'Contribution':>12}  Direction")
    print(f"  {'-'*28} {'-'*12}  ---------")
    for feat, val in top_shap.items():
        direction = "▲ bullish" if val > 0 else "▼ bearish"
        print(f"  {feat:<28} {val:>+12.4f}  {direction}")
    print(f"  {'[bias / model base]':<28} {bias:>+12.4f}")
    print(f"  {'[XGB raw prediction]':<28} {xgb_pred_val:>+12.4f}")
    print(bar + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    symbols = sys.argv[1:] or ["AAPL"]
    for sym in symbols:
        explain_signal(sym.upper())
