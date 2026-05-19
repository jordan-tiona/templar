"""Portfolio-level risk checks before order submission."""

import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.models import Position

from config.settings import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY,
    MAX_POSITION_PCT, MAX_DRAWDOWN_PCT,
)

log = logging.getLogger(__name__)


def get_client() -> TradingClient:
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)


def portfolio_equity(client: TradingClient) -> float:
    account = client.get_account()
    return float(account.equity)


def peak_equity(client: TradingClient) -> float:
    """Use last_equity as a proxy for previous-day peak (good enough for daily trading)."""
    account = client.get_account()
    return float(account.last_equity)


def drawdown_breached(client: TradingClient) -> bool:
    equity = portfolio_equity(client)
    peak = peak_equity(client)
    if peak <= 0:
        return False
    drawdown = (peak - equity) / peak
    if drawdown >= MAX_DRAWDOWN_PCT:
        log.warning("Max drawdown breached: %.2f%% >= %.2f%%", drawdown * 100, MAX_DRAWDOWN_PCT * 100)
        return True
    return False


def max_position_value(client: TradingClient) -> float:
    return portfolio_equity(client) * MAX_POSITION_PCT


def current_position(client: TradingClient, symbol: str) -> Position | None:
    try:
        return client.get_open_position(symbol)
    except Exception:
        return None


def position_value(client: TradingClient, symbol: str) -> float:
    pos = current_position(client, symbol)
    if pos is None:
        return 0.0
    return abs(float(pos.market_value))
