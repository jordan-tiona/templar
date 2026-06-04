"""Daily signal generation pipeline."""

import json
import logging
import numpy as np
import pandas as pd
import xgboost as xgb_lib
from datetime import datetime

from config.settings import WATCHLIST, SEQUENCE_LENGTH, PREDICTION_HORIZON, DATA_CACHE
from data import fetch_bars, fetch_news_sentiment, build_features, load_stocktwits_today
from models import tft as tft_model, xgb as xgb_model, ensemble
from models.dataset import add_target, TIME_VARYING_UNKNOWN
from .explain_text import add_summaries

log = logging.getLogger(__name__)


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _compute_explain(
    sym: str,
    df: pd.DataFrame,
    xgb_m,
    feature_cols: list[str],
    X_scaled: np.ndarray,
    tft_pred: float,
    xgb_pred: float,
    ensemble_score: float,
    signal: str,
    tft_interp: dict | None = None,
) -> dict:
    latest = df.iloc[-1]
    signal_date = df.index[-1]

    booster = xgb_m.get_booster()
    dm = xgb_lib.DMatrix(X_scaled, feature_names=feature_cols)
    contribs = booster.predict(dm, pred_contribs=True)[0]
    shap_vals = sorted(
        [{"feature": f, "value": float(v)} for f, v in zip(feature_cols, contribs[:-1])],
        key=lambda x: abs(x["value"]),
        reverse=True,
    )

    snapshot_keys = [
        "close", "volume", "return_1d", "return_5d", "return_20d",
        "EMA_9", "EMA_21", "EMA_50",
        "RSI_14", "STOCHk_14_3_3", "STOCHd_14_3_3",
        "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
        "BBP_20_2_0_2_0", "BBB_20_2_0_2_0", "ATRr_14",
        "OBV", "VWAP_D", "sentiment_mean", "sentiment_count",
    ]
    snapshot = {k: _safe_float(latest.get(k)) for k in snapshot_keys}

    return {
        "symbol": sym,
        "date": str(signal_date.date()) if hasattr(signal_date, "date") else str(signal_date),
        "signal": signal,
        "ensemble_score": _safe_float(ensemble_score),
        "tft_pred": _safe_float(tft_pred),
        "xgb_pred": _safe_float(xgb_pred),
        "shap": shap_vals,
        "xgb_bias": float(contribs[-1]),
        "snapshot": snapshot,
        "tft_interpretation": tft_interp,
    }


def _save(signals_df: pd.DataFrame, explain: dict) -> None:
    generated_at = datetime.utcnow().isoformat()

    signals_payload = {
        "generated_at": generated_at,
        "signals": [
            {
                "symbol": sym,
                "signal": row["signal"],
                "ensemble_score": _safe_float(row["ensemble_score"]),
                "tft_pred": _safe_float(row["tft_pred"]),
                "xgb_pred": _safe_float(row["xgb_pred"]),
            }
            for sym, row in signals_df.iterrows()
        ],
    }

    (DATA_CACHE / "latest_signals.json").write_text(json.dumps(signals_payload, indent=2))
    (DATA_CACHE / "latest_explain.json").write_text(json.dumps(explain, indent=2))

    history_path = DATA_CACHE / "signals_history.jsonl"
    with open(history_path, "a") as f:
        f.write(json.dumps(signals_payload) + "\n")

    log.info("Signals and explain data saved to %s", DATA_CACHE)


def generate_signals(
    symbols: list[str] | None = None,
    lookback_days: int = SEQUENCE_LENGTH + PREDICTION_HORIZON + 100,
) -> pd.DataFrame:
    """
    Run the full signal pipeline for today.
    Returns a DataFrame with columns [symbol, ensemble_score, signal, tft_pred, xgb_pred].
    Also saves signals + explain data to data/cache/ for the dashboard.
    """
    symbols = symbols or WATCHLIST

    log.info("Fetching bars for %d symbols", len(symbols))
    bars = fetch_bars(symbols)

    feature_dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        if sym not in bars or bars[sym].empty:
            log.warning("No bar data for %s, skipping", sym)
            continue
        full_bars = bars[sym]
        if len(full_bars) < 400:
            log.warning("%s has insufficient total history (%d bars), skipping", sym, len(full_bars))
            continue
        df = full_bars.tail(lookback_days)
        sentiment = fetch_news_sentiment(sym)
        feature_dfs[sym] = build_features(sym, df, sentiment, save=False)

    if not feature_dfs:
        raise RuntimeError("No feature data available for any symbol")

    log.info("Loading models")
    tft = tft_model.load()
    xgb_m, scaler = xgb_model.load()

    tft_preds: dict[str, float] = {}
    xgb_preds: dict[str, float] = {}
    explain_inputs: dict[str, dict] = {}

    for sym, df in feature_dfs.items():
        df_t = add_target(df)
        if len(df_t) < SEQUENCE_LENGTH + PREDICTION_HORIZON + 20:
            log.warning("%s: too few rows after target drop (%d), skipping", sym, len(df_t))
            continue

        feature_cols = [c for c in TIME_VARYING_UNKNOWN if c in df_t.columns]
        X_scaled = scaler.transform(df_t.iloc[[-1]][feature_cols].values)

        df_t["symbol"] = sym
        df_t["time_idx"] = range(len(df_t))

        try:
            tft_preds[sym] = float(tft_model.predict(tft, df_t, df_t).iloc[-1])
        except Exception as e:
            log.error("TFT prediction failed for %s: %s", sym, e)
            tft_preds[sym] = float("nan")

        try:
            xgb_preds[sym] = float(xgb_m.predict(X_scaled)[0])
        except Exception as e:
            log.error("XGBoost prediction failed for %s: %s", sym, e)
            xgb_preds[sym] = float("nan")

        explain_inputs[sym] = {"df": df_t, "X_scaled": X_scaled, "feature_cols": feature_cols}

    tft_series = pd.Series(tft_preds, name="tft_pred")
    xgb_series = pd.Series(xgb_preds, name="xgb_pred")
    score = ensemble.combine(tft_series, xgb_series)
    signals = ensemble.to_signals(score)

    result = pd.DataFrame({
        "tft_pred": tft_series,
        "xgb_pred": xgb_series,
        "ensemble_score": score,
        "signal": signals,
    })
    result.index.name = "symbol"
    result["generated_at"] = datetime.utcnow().isoformat()

    log.info("Signal summary:\n%s", result[["ensemble_score", "signal"]].to_string())

    explain: dict[str, dict] = {}
    for sym in result.index:
        if sym not in explain_inputs:
            continue
        inp = explain_inputs[sym]
        row = result.loc[sym]

        tft_interp = None
        try:
            tft_interp = tft_model.interpret(tft, inp["df"])
        except Exception as e:
            log.warning("TFT interpretation failed for %s: %s", sym, e)

        try:
            explain[sym] = _compute_explain(
                sym=sym,
                df=inp["df"],
                xgb_m=xgb_m,
                feature_cols=inp["feature_cols"],
                X_scaled=inp["X_scaled"],
                tft_pred=float(row["tft_pred"]),
                xgb_pred=float(row["xgb_pred"]),
                ensemble_score=float(row["ensemble_score"]),
                signal=row["signal"],
                tft_interp=tft_interp,
            )
        except Exception as e:
            log.error("Failed to compute explain data for %s: %s", sym, e)

    result = _apply_stocktwits_filter(result)

    add_summaries(explain)
    _save(result, explain)

    return result


# ---------------------------------------------------------------------------
# StockTwits inference filter (Option B)
# ---------------------------------------------------------------------------
# Suppresses signals when fresh social sentiment strongly contradicts the model.
# Thresholds are intentionally conservative: only act on clear contrary signals
# with enough volume to be meaningful.
_ST_BULLISH_SUPPRESS_THRESHOLD = 0.30   # ratio below which BUY is suppressed
_ST_BEARISH_SUPPRESS_THRESHOLD = 0.70   # ratio above which SHORT is suppressed
_ST_MIN_MENTIONS = 5                    # ignore if fewer messages than this today


def _apply_stocktwits_filter(result: pd.DataFrame) -> pd.DataFrame:
    """
    Suppress BUY/SHORT signals where fresh StockTwits sentiment strongly disagrees.
    Modifies 'signal' in-place; does nothing if no recent cache exists for a symbol.
    """
    result = result.copy()
    suppressed: list[str] = []

    for sym in result.index:
        signal = result.at[sym, "signal"]
        if signal not in ("BUY", "SHORT"):
            continue

        bullish_ratio, mention_count = load_stocktwits_today(sym)
        if mention_count < _ST_MIN_MENTIONS:
            continue  # not enough data to act on

        if signal == "BUY" and bullish_ratio < _ST_BULLISH_SUPPRESS_THRESHOLD:
            result.at[sym, "signal"] = "HOLD"
            suppressed.append(f"{sym} BUY→HOLD (ratio={bullish_ratio:.2f}, n={mention_count})")
        elif signal == "SHORT" and bullish_ratio > _ST_BEARISH_SUPPRESS_THRESHOLD:
            result.at[sym, "signal"] = "HOLD"
            suppressed.append(f"{sym} SHORT→HOLD (ratio={bullish_ratio:.2f}, n={mention_count})")

    if suppressed:
        log.info("StockTwits filter suppressed %d signal(s):", len(suppressed))
        for s in suppressed:
            log.info("  -- %s", s)

    return result
