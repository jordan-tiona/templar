"""Fetch and cache historical OHLCV bars from Alpaca."""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config.settings import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY,
    DATA_RAW, HISTORY_YEARS, WATCHLIST,
)


def _client() -> StockHistoricalDataClient:
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def fetch_bars(
    symbols: list[str] | None = None,
    years: int = HISTORY_YEARS,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Return a dict of symbol -> OHLCV DataFrame, using disk cache when available."""
    symbols = symbols or WATCHLIST
    end = datetime.utcnow().date()
    start = end - timedelta(days=365 * years)

    results: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    for sym in symbols:
        cache_path = DATA_RAW / f"{sym}.parquet"
        if not force_refresh and cache_path.exists():
            df = pd.read_parquet(cache_path)
            # top-up if cache is stale (missing recent trading days)
            last = df.index.max().date()
            if (end - last).days > 1:
                to_fetch.append(sym)
                results[sym] = df  # will be merged below
            else:
                results[sym] = df
        else:
            to_fetch.append(sym)

    if to_fetch:
        client = _client()
        req = StockBarsRequest(
            symbol_or_symbols=to_fetch,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        raw = client.get_stock_bars(req).df

        # alpaca returns a MultiIndex (symbol, timestamp); flatten per symbol
        for sym in to_fetch:
            df_new = raw.xs(sym, level="symbol") if sym in raw.index.get_level_values("symbol") else pd.DataFrame()
            df_new.index = pd.to_datetime(df_new.index).tz_localize(None)
            df_new.index.name = "date"

            cache_path = DATA_RAW / f"{sym}.parquet"
            if sym in results and not results[sym].empty:
                # merge top-up with cached data
                df_new = pd.concat([results[sym], df_new])
                df_new = df_new[~df_new.index.duplicated(keep="last")].sort_index()

            df_new.to_parquet(cache_path)
            results[sym] = df_new

    return results
