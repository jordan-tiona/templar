"""News sentiment scoring via Alpha Vantage + FinBERT."""

import os
import time
import requests
import pandas as pd
from transformers import pipeline
from functools import lru_cache

from config.settings import ALPHA_VANTAGE_API_KEY, DATA_CACHE

_AV_BASE = "https://www.alphavantage.co/query"
_TOPICS = "financial_markets,finance,earnings"


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


def fetch_news_sentiment(
    symbol: str,
    limit: int = 200,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame indexed by date with columns [sentiment_mean, sentiment_count].
    Pulls from Alpha Vantage NEWS_SENTIMENT, scores with FinBERT, caches to disk.
    """
    cache_path = DATA_CACHE / f"sentiment_{symbol}.parquet"
    if not force_refresh and cache_path.exists():
        return pd.read_parquet(cache_path)

    if not ALPHA_VANTAGE_API_KEY:
        return pd.DataFrame(columns=["sentiment_mean", "sentiment_count"])

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "topics": _TOPICS,
        "limit": limit,
        "apikey": ALPHA_VANTAGE_API_KEY,
    }
    resp = requests.get(_AV_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    feed = data.get("feed", [])
    if not feed:
        return pd.DataFrame(columns=["sentiment_mean", "sentiment_count"])

    texts = [item.get("title", "") + " " + item.get("summary", "") for item in feed]
    dates = [item["time_published"][:8] for item in feed]  # YYYYMMDD

    scores = _score(texts)

    df = pd.DataFrame({"date": dates, "score": scores})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    daily = df.groupby("date")["score"].agg(
        sentiment_mean="mean",
        sentiment_count="count",
    )
    daily.to_parquet(cache_path)
    return daily


def fetch_all_sentiment(symbols: list[str], delay: float = 12.0) -> dict[str, pd.DataFrame]:
    """Fetch sentiment for all symbols. Throttles to respect AV free-tier rate limits."""
    results = {}
    for i, sym in enumerate(symbols):
        results[sym] = fetch_news_sentiment(sym)
        if i < len(symbols) - 1:
            time.sleep(delay)  # AV free tier: 5 req/min
    return results
