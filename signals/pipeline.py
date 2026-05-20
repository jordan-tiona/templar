"""Daily signal generation pipeline."""

import logging
import pandas as pd
from datetime import datetime, timedelta

from config.settings import WATCHLIST, SEQUENCE_LENGTH, PREDICTION_HORIZON
from data import fetch_bars, fetch_news_sentiment, build_features
from models import tft as tft_model, xgb as xgb_model, ensemble
from models.dataset import prepare_combined, add_target, TIME_VARYING_UNKNOWN

log = logging.getLogger(__name__)


def generate_signals(
    symbols: list[str] | None = None,
    lookback_days: int = SEQUENCE_LENGTH + PREDICTION_HORIZON + 100,
) -> pd.DataFrame:
    """
    Run the full signal pipeline for today.
    Returns a DataFrame with columns [symbol, ensemble_score, signal, tft_pred, xgb_pred].
    """
    symbols = symbols or WATCHLIST

    # 1. Fetch recent bars (only need recent window for inference)
    log.info("Fetching bars for %d symbols", len(symbols))
    bars = fetch_bars(symbols)

    # 2. Build features per symbol
    feature_dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        if sym not in bars or bars[sym].empty:
            log.warning("No bar data for %s, skipping", sym)
            continue
        df = bars[sym].tail(lookback_days)
        sentiment = fetch_news_sentiment(sym)
        feature_dfs[sym] = build_features(sym, df, sentiment, save=False)

    if not feature_dfs:
        raise RuntimeError("No feature data available for any symbol")

    # 3. Load models
    log.info("Loading models")
    tft = tft_model.load()
    xgb, scaler = xgb_model.load()

    # 4. Generate predictions per symbol
    tft_preds, xgb_preds = {}, {}
    for sym, df in feature_dfs.items():
        df_with_target = add_target(df)
        min_rows = max(SEQUENCE_LENGTH + PREDICTION_HORIZON + 20, 400)
        if len(df_with_target) < min_rows:
            log.warning("%s has insufficient history (%d rows, need %d), skipping", sym, len(df_with_target), min_rows)
            continue

        # Use only the latest row for the signal
        latest = df_with_target.iloc[[-1]]

        try:
            # TFT needs context window; pass full df as encoder context
            df_with_target["symbol"] = sym
            df_with_target["time_idx"] = range(len(df_with_target))
            tft_preds[sym] = tft_model.predict(tft, df_with_target, df_with_target).iloc[-1]
        except Exception as e:
            log.error("TFT prediction failed for %s: %s", sym, e)
            tft_preds[sym] = float("nan")

        try:
            xgb_preds[sym] = xgb_model.predict(xgb, scaler, latest).iloc[0]
        except Exception as e:
            log.error("XGBoost prediction failed for %s: %s", sym, e)
            xgb_preds[sym] = float("nan")

    # 5. Ensemble
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
    return result
