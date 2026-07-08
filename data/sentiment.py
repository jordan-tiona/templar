"""News sentiment: Alpaca News API (Benzinga, history to ~2015) scored with FinBERT.

Daily per-symbol aggregates are cached to data/cache/sentiment_news_{symbol}.parquet
and updated incrementally. Articles are bucketed by their US/Eastern calendar date.

StockTwits social sentiment is collected daily and cached as a separate signal.
It is not used in model training until sufficient history has accumulated (~3 months).
At inference it acts as a filter: strong contrary social sentiment can suppress a signal.
"""

import json
import logging
import time
from datetime import datetime, date, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from transformers import pipeline

from config.settings import DATA_CACHE, ALPACA_API_KEY, ALPACA_SECRET_KEY

log = logging.getLogger(__name__)

_EASTERN = ZoneInfo("America/New_York")
_NEWS_LOOKBACK_DAYS = 5 * 365


# ---------------------------------------------------------------------------
# FinBERT helpers
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
# Alpaca News API (Benzinga) helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _news_client():
    from alpaca.data.historical.news import NewsClient
    return NewsClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def _fetch_headlines(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch Benzinga headlines for symbol between start and end (UTC).
    Returns a DataFrame with columns [date, headline]; date is the US/Eastern
    calendar date of publication. alpaca-py paginates internally (50 articles
    per request), so the range is fetched in 180-day chunks with pauses to
    stay under the 200 req/min rate limit on long backfills.
    """
    from alpaca.data.requests import NewsRequest

    chunk = timedelta(days=180)
    rows = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + chunk, end)
        req = NewsRequest(
            symbols=symbol, start=chunk_start, end=chunk_end, include_content=False,
        )
        news = _news_client().get_news(req).data.get("news", [])
        rows.extend(
            {"date": n.created_at.astimezone(_EASTERN).date(), "headline": n.headline}
            for n in news
            if n.headline
        )
        chunk_start = chunk_end
        if chunk_start < end:
            time.sleep(1)
    return pd.DataFrame(rows, columns=["date", "headline"])


def _aggregate_daily(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate [date, score] rows into daily [sentiment_mean, sentiment_count]."""
    daily = scored.groupby("date").agg(
        sentiment_mean=("score", "mean"),
        sentiment_count=("score", "count"),
    )
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    return daily.sort_index()


def fetch_news_sentiment(
    symbol: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame indexed by date with columns [sentiment_mean, sentiment_count].

    First call backfills ~5 years of Benzinga headlines via the Alpaca News API
    and scores them with FinBERT; subsequent calls only fetch and score days
    after the last cached date. The most recent cached day is re-fetched so
    late-arriving articles are included.
    """
    cache_path = DATA_CACHE / f"sentiment_news_{symbol}.parquet"
    now = datetime.now(timezone.utc)

    existing: pd.DataFrame | None = None
    start = now - timedelta(days=_NEWS_LOOKBACK_DAYS)
    if not force_refresh and cache_path.exists():
        existing = pd.read_parquet(cache_path)
        if not existing.empty:
            # Re-fetch the last cached day to pick up late-arriving articles
            last = existing.index.max()
            existing = existing[existing.index < last]
            start = last.tz_localize(_EASTERN).astimezone(timezone.utc)

    try:
        headlines = _fetch_headlines(symbol, start, now)
    except Exception as exc:
        log.warning("Alpaca news fetch failed for %s: %s", symbol, exc)
        return existing if existing is not None else pd.DataFrame(
            columns=["sentiment_mean", "sentiment_count"]
        )

    if headlines.empty:
        daily_new = pd.DataFrame(columns=["sentiment_mean", "sentiment_count"])
    else:
        headlines = headlines.drop_duplicates()  # chunk boundaries can overlap
        headlines["score"] = _score(headlines["headline"].tolist())
        daily_new = _aggregate_daily(headlines[["date", "score"]])

    parts = [p for p in (existing, daily_new) if p is not None and not p.empty]
    if not parts:
        log.info("No news sentiment for %s", symbol)
        return pd.DataFrame(columns=["sentiment_mean", "sentiment_count"])

    daily = pd.concat(parts).sort_index()
    daily.to_parquet(cache_path)
    log.debug(
        "News sentiment for %s: %d days cached (%d new headlines)",
        symbol, len(daily), len(headlines),
    )
    return daily


def fetch_all_sentiment(
    symbols: list[str],
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch news sentiment for all symbols with progress logging."""
    results = {}
    for i, sym in enumerate(symbols, 1):
        results[sym] = fetch_news_sentiment(sym, force_refresh=force_refresh)
        if i % 10 == 0 or i == len(symbols):
            log.info("News sentiment: %d/%d symbols done", i, len(symbols))
        time.sleep(0.3)  # stay well under the 200 req/min rate limit
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
