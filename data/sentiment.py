"""News sentiment scoring via GDELT DOC 2.0 timeline API + FinBERT (kept for future use)."""

import logging
import time
from datetime import datetime, timedelta
from functools import lru_cache

import pandas as pd
import requests
from transformers import pipeline

from config.settings import DATA_CACHE

log = logging.getLogger(__name__)

_GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

# Ticker → company name for GDELT queries
_COMPANY_NAMES: dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "META": "Meta",
    "JPM": "JPMorgan",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "XOM": "ExxonMobil",
    "CVX": "Chevron",
    "JNJ": "Johnson Johnson",
    "UNH": "UnitedHealth",
    "SPY": "S&P 500",
}


# ---------------------------------------------------------------------------
# FinBERT helpers — kept for potential future use, not called by GDELT path
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _finbert():
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        top_k=None,
        device=-1,  # CPU; set to 0 for GPU
    )


def _score(texts: list[str]) -> list[float]:
    """Return a list of sentiment scores in [-1, 1] (neg→pos)."""
    if not texts:
        return []
    results = _finbert()(texts, batch_size=16, truncation=True, max_length=512)
    scores = []
    for item in results:
        label_score = {r["label"]: r["score"] for r in item}
        score = label_score.get("positive", 0) - label_score.get("negative", 0)
        scores.append(score)
    return scores


# ---------------------------------------------------------------------------
# GDELT helpers
# ---------------------------------------------------------------------------

def _gdelt_chunk(query: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch one chunk of GDELT timeline tone data.
    Returns a DataFrame with columns [date, sentiment_mean, sentiment_count].
    """
    params = {
        "query": query,
        "mode": "timelinetone",
        "STARTDATETIME": start.strftime("%Y%m%d%H%M%S"),
        "ENDDATETIME": end.strftime("%Y%m%d%H%M%S"),
        "format": "json",
    }
    try:
        resp = requests.get(_GDELT_BASE, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("GDELT request failed: %s", exc)
        return pd.DataFrame(columns=["date", "sentiment_mean", "sentiment_count"])

    try:
        data = resp.json()
    except ValueError as exc:
        log.warning("GDELT JSON parse error: %s", exc)
        return pd.DataFrame(columns=["date", "sentiment_mean", "sentiment_count"])

    timeline = data.get("timeline", [])
    if not timeline:
        return pd.DataFrame(columns=["date", "sentiment_mean", "sentiment_count"])

    # Each item in timeline has a "series" key with a list of {date, value} dicts.
    # Flatten all series entries.
    rows = []
    for series_item in timeline:
        for point in series_item.get("data", []):
            raw_date = point.get("date", "")
            value = point.get("value")
            if value is None:
                continue
            # date format: "YYYYMMDDTHHMMSS"
            try:
                dt = datetime.strptime(raw_date[:8], "%Y%m%d").date()
            except ValueError:
                continue
            rows.append({"date": dt, "sentiment_mean": float(value) / 100.0, "sentiment_count": 1})

    if not rows:
        return pd.DataFrame(columns=["date", "sentiment_mean", "sentiment_count"])

    return pd.DataFrame(rows)


def fetch_news_sentiment(
    symbol: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame indexed by date with columns [sentiment_mean, sentiment_count].
    Pulls 5 years of history from GDELT DOC 2.0 in 6-month chunks, caches to disk.
    """
    cache_path = DATA_CACHE / f"sentiment_{symbol}.parquet"
    if not force_refresh and cache_path.exists():
        return pd.read_parquet(cache_path)

    company = _COMPANY_NAMES.get(symbol, symbol)
    query = f'"{company} stock"'

    end = datetime.utcnow()
    start_total = end - timedelta(days=5 * 365)

    chunk_size = timedelta(days=180)
    all_chunks: list[pd.DataFrame] = []

    chunk_start = start_total
    while chunk_start < end:
        chunk_end = min(chunk_start + chunk_size, end)
        log.debug("GDELT fetch %s: %s → %s", symbol, chunk_start.date(), chunk_end.date())
        chunk_df = _gdelt_chunk(query, chunk_start, chunk_end)
        if not chunk_df.empty:
            all_chunks.append(chunk_df)
        chunk_start = chunk_end
        time.sleep(1)  # avoid GDELT rate limiting across chunks

    if not all_chunks:
        log.warning("No GDELT data returned for %s", symbol)
        return pd.DataFrame(columns=["sentiment_mean", "sentiment_count"])

    combined = pd.concat(all_chunks, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])

    # Aggregate by date (multiple chunks can overlap at boundaries)
    daily = combined.groupby("date").agg(
        sentiment_mean=("sentiment_mean", "mean"),
        sentiment_count=("sentiment_count", "sum"),
    )
    daily.index.name = "date"
    daily.sort_index(inplace=True)

    daily.to_parquet(cache_path)
    log.info("GDELT sentiment for %s: %d days cached", symbol, len(daily))
    return daily


def fetch_all_sentiment(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch GDELT sentiment for all symbols. No rate limiting needed (public API)."""
    results = {}
    for sym in symbols:
        results[sym] = fetch_news_sentiment(sym)
    return results
