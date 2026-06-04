"""News sentiment scoring via GDELT DOC 2.0 timeline API + FinBERT (kept for future use).

StockTwits social sentiment is collected daily and cached as a separate signal.
It is not used in model training until sufficient history has accumulated (~3 months).
At inference it acts as a filter: strong contrary social sentiment can suppress a signal.
"""

import json
import logging
import time
from datetime import datetime, date, timedelta
from functools import lru_cache

import pandas as pd
import requests
import yfinance as yf
from transformers import pipeline

from config.settings import DATA_CACHE

log = logging.getLogger(__name__)

_GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
_COMPANY_NAME_CACHE = DATA_CACHE / "company_names.json"


def _load_company_name_cache() -> dict[str, str]:
    if _COMPANY_NAME_CACHE.exists():
        with open(_COMPANY_NAME_CACHE) as f:
            return json.load(f)
    return {}


def _save_company_name_cache(names: dict[str, str]) -> None:
    with open(_COMPANY_NAME_CACHE, "w") as f:
        json.dump(names, f, indent=2)


def _company_name(symbol: str) -> str:
    """Return a human-readable company name for GDELT queries, cached to disk."""
    cache = _load_company_name_cache()
    if symbol in cache:
        return cache[symbol]
    try:
        info = yf.Ticker(symbol).info
        name = info.get("longName") or info.get("shortName") or symbol
        # Strip common suffixes that add noise to news queries
        for suffix in (", Inc.", " Inc.", ", Corp.", " Corp.", " Corporation", ", Ltd.", " Ltd.", " Group"):
            name = name.replace(suffix, "")
        name = name.strip()
    except Exception:
        name = symbol
    cache[symbol] = name
    _save_company_name_cache(cache)
    log.debug("Resolved company name: %s -> %s", symbol, name)
    return name


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

def _gdelt_chunk(query: str, start: datetime, end: datetime, retries: int = 3) -> pd.DataFrame:
    """
    Fetch one chunk of GDELT timeline tone data with exponential backoff.
    Returns a DataFrame with columns [date, sentiment_mean, sentiment_count].
    """
    params = {
        "query": query,
        "mode": "timelinetone",
        "STARTDATETIME": start.strftime("%Y%m%d%H%M%S"),
        "ENDDATETIME": end.strftime("%Y%m%d%H%M%S"),
        "format": "json",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(_GDELT_BASE, params=params, timeout=15)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            wait = 2 * (2 ** attempt)  # 2s, 4s, 8s
            if attempt < retries - 1:
                log.debug("GDELT retry %d/%d after %ds: %s", attempt + 1, retries, wait, exc)
                time.sleep(wait)
            else:
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

    company = _company_name(symbol)
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
    """Fetch GDELT sentiment for all symbols."""
    results = {}
    for i, sym in enumerate(symbols):
        results[sym] = fetch_news_sentiment(sym)
        if i < len(symbols) - 1:
            time.sleep(3)  # pause between symbols to avoid sustained rate limiting
    return results


# ---------------------------------------------------------------------------
# StockTwits social sentiment
# ---------------------------------------------------------------------------

_STOCKTWITS_BASE = "https://api.stocktwits.com/api/2/streams/symbol"
_ST_BACKFILL_PAGES = 10  # ~300 messages max per symbol on initial run
# Minimum labeled-message count before bullish_ratio is considered reliable.
ST_MIN_LABELED = 5


def _stocktwits_page(
    symbol: str,
    max_id: int | None = None,
    since_id: int | None = None,
) -> dict:
    url = f"{_STOCKTWITS_BASE}/{symbol}.json"
    params: dict = {"limit": 30}
    if max_id is not None:
        params["max"] = max_id
    if since_id is not None:
        params["since"] = since_id
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        log.warning("StockTwits request failed for %s: %s", symbol, exc)
        return {}


def _parse_stocktwits_messages(messages: list[dict]) -> pd.DataFrame:
    rows = []
    for msg in messages:
        try:
            dt = datetime.strptime(msg["created_at"], "%Y-%m-%dT%H:%M:%SZ").date()
        except (KeyError, ValueError):
            continue
        sentiment_raw = (msg.get("entities") or {}).get("sentiment") or {}
        label = sentiment_raw.get("basic", "") if isinstance(sentiment_raw, dict) else ""
        rows.append({
            "date": dt,
            "msg_id": int(msg["id"]),
            "bullish": 1 if label == "Bullish" else 0,
            "bearish": 1 if label == "Bearish" else 0,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["date", "msg_id", "bullish", "bearish"]
    )


def _aggregate_stocktwits(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw message rows into daily [stocktwits_bullish_ratio, stocktwits_mention_count]."""
    raw = raw.drop_duplicates("msg_id").copy()
    raw["date"] = pd.to_datetime(raw["date"])
    daily = raw.groupby("date").agg(
        stocktwits_mention_count=("msg_id", "count"),
        _bullish=("bullish", "sum"),
        _bearish=("bearish", "sum"),
    )
    labeled = daily["_bullish"] + daily["_bearish"]
    # Where too few messages are labeled, default ratio to neutral 0.5
    daily["stocktwits_bullish_ratio"] = (
        daily["_bullish"] / labeled.where(labeled >= ST_MIN_LABELED)
    ).fillna(0.5)
    return daily[["stocktwits_bullish_ratio", "stocktwits_mention_count"]].sort_index()


def fetch_stocktwits_sentiment(
    symbol: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return DataFrame indexed by date with [stocktwits_bullish_ratio, stocktwits_mention_count].

    First call: paginates back ~300 messages for initial history.
    Subsequent calls: incremental — only fetches messages since last run.
    Results are cached to data/cache/sentiment_stocktwits_{symbol}.parquet.
    """
    cache_path = DATA_CACHE / f"sentiment_stocktwits_{symbol}.parquet"
    cursor_path = DATA_CACHE / f"stocktwits_cursor_{symbol}.json"

    existing_raw: pd.DataFrame | None = None
    since_id: int | None = None

    if not force_refresh and cache_path.exists() and cursor_path.exists():
        # Load raw message-level cache (stored alongside aggregated parquet)
        raw_cache_path = DATA_CACHE / f"stocktwits_raw_{symbol}.parquet"
        if raw_cache_path.exists():
            existing_raw = pd.read_parquet(raw_cache_path)
        with open(cursor_path) as f:
            since_id = json.load(f).get("since_id")

    new_message_rows: list[pd.DataFrame] = []
    new_since_id: int | None = None

    if since_id is not None:
        # Incremental update: pull messages newer than last seen
        data = _stocktwits_page(symbol, since_id=since_id)
        messages = data.get("messages", [])
        if messages:
            new_message_rows.append(_parse_stocktwits_messages(messages))
            new_since_id = data.get("cursor", {}).get("since")
        else:
            new_since_id = since_id  # nothing new
    else:
        # Initial backfill: paginate backward up to _ST_BACKFILL_PAGES pages
        max_id: int | None = None
        for page in range(_ST_BACKFILL_PAGES):
            data = _stocktwits_page(symbol, max_id=max_id)
            messages = data.get("messages", [])
            if not messages:
                break
            parsed = _parse_stocktwits_messages(messages)
            new_message_rows.append(parsed)
            if page == 0:
                new_since_id = data.get("cursor", {}).get("since")
            max_id = data.get("cursor", {}).get("max")
            if not max_id:
                break
            time.sleep(0.5)

    # Merge with existing raw rows and re-aggregate
    all_raw_parts = ([existing_raw] if existing_raw is not None else []) + new_message_rows
    if not all_raw_parts:
        log.warning("No StockTwits data returned for %s", symbol)
        return pd.DataFrame(columns=["stocktwits_bullish_ratio", "stocktwits_mention_count"])

    all_raw = pd.concat(all_raw_parts, ignore_index=True)

    # Persist raw rows for future incremental merges
    raw_cache_path = DATA_CACHE / f"stocktwits_raw_{symbol}.parquet"
    all_raw.to_parquet(raw_cache_path)

    # Persist cursor for next incremental run
    if new_since_id is not None:
        with open(cursor_path, "w") as f:
            json.dump({"since_id": new_since_id}, f)

    daily = _aggregate_stocktwits(all_raw)
    daily.to_parquet(cache_path)
    log.info("StockTwits for %s: %d days cached (%d messages)", symbol, len(daily), len(all_raw))
    return daily


def collect_all_stocktwits(
    symbols: list[str],
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Collect StockTwits sentiment for all symbols, 1s gap between requests."""
    results: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        results[sym] = fetch_stocktwits_sentiment(sym, force_refresh=force_refresh)
        if i < len(symbols) - 1:
            time.sleep(1)
    return results


def load_stocktwits_today(symbol: str) -> tuple[float, int]:
    """
    Return (bullish_ratio, mention_count) for the most recent cached date.
    Returns (0.5, 0) if no recent data (within 3 calendar days).
    """
    cache_path = DATA_CACHE / f"sentiment_stocktwits_{symbol}.parquet"
    if not cache_path.exists():
        return 0.5, 0
    try:
        df = pd.read_parquet(cache_path)
        if df.empty:
            return 0.5, 0
        latest_date = df.index.max().date()
        if (date.today() - latest_date).days > 3:
            return 0.5, 0
        row = df.loc[df.index.max()]
        return float(row["stocktwits_bullish_ratio"]), int(row["stocktwits_mention_count"])
    except Exception as exc:
        log.debug("StockTwits load failed for %s: %s", symbol, exc)
        return 0.5, 0
