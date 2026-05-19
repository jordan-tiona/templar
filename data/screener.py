"""Symbol screener — fetch S&P 500 universe and filter by volume."""

import io
import json
import logging

import pandas as pd
import requests
import yfinance as yf

from config.settings import WATCHLIST, ROOT

log = logging.getLogger(__name__)

WATCHLIST_PATH = ROOT / "watchlist.json"


_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Templar/1.0; +https://github.com/templar)"}


def fetch_sp500() -> list[str]:
    """Fetch current S&P 500 tickers from Wikipedia."""
    resp = requests.get(_WIKI_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    symbols = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    return symbols


def screen_by_volume(
    symbols: list[str],
    min_avg_volume: int = 5_000_000,
    top_n: int = 50,
    period: str = "3mo",
) -> list[str]:
    """
    Filter symbols by average daily volume using yfinance.
    Downloads volume data for all symbols in one batch call.
    Returns top_n symbols sorted by avg volume descending.
    """
    log.info("Downloading volume data for %d symbols...", len(symbols))
    data = yf.download(
        symbols,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # Extract Volume; handle single-symbol case (flat columns) vs multi-symbol (MultiIndex)
    if isinstance(data.columns, pd.MultiIndex):
        volume = data["Volume"]
    else:
        volume = data[["Volume"]]
        volume.columns = symbols[:1]

    avg_vol = volume.mean(axis=0).dropna()
    filtered = avg_vol[avg_vol >= min_avg_volume]
    ranked = filtered.sort_values(ascending=False)
    result = ranked.head(top_n).index.tolist()
    log.info(
        "Screened %d → %d symbols (min_avg_vol=%d, top_n=%d)",
        len(symbols), len(result), min_avg_volume, top_n,
    )
    return result


def get_watchlist() -> list[str]:
    """Return current watchlist from watchlist.json if it exists, else from settings."""
    if WATCHLIST_PATH.exists():
        with open(WATCHLIST_PATH) as f:
            return json.load(f)
    return list(WATCHLIST)


def save_watchlist(symbols: list[str]) -> None:
    """Save watchlist to ROOT/watchlist.json."""
    with open(WATCHLIST_PATH, "w") as f:
        json.dump(symbols, f, indent=2)
    log.info("Watchlist saved to %s (%d symbols)", WATCHLIST_PATH, len(symbols))
