"""Portfolio-level risk checks before order submission."""

import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.models import Position
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderType, QueryOrderStatus

from config.settings import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY,
    MAX_POSITION_PCT, MAX_DRAWDOWN_PCT,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    TARGET_GROSS_LEVERAGE,
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


def max_position_value(client: TradingClient, n_targets: int = 0) -> float:
    """
    Per-position budget. When the target book size is known, split the gross
    exposure target evenly across it (capped by MAX_POSITION_PCT) so total
    exposure stays inside overnight margin regardless of position count.
    """
    equity = portfolio_equity(client)
    cap = equity * MAX_POSITION_PCT
    if n_targets <= 0:
        return cap
    return min(cap, equity * TARGET_GROSS_LEVERAGE / n_targets)


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


def take_profit_triggered(pos: Position) -> bool:
    """
    Return True if position has hit the take-profit target.
    Stop losses are handled by Alpaca native GTC stop orders (not checked here).
    """
    try:
        plpc = float(pos.unrealized_plpc)
    except (TypeError, ValueError):
        return False

    is_long = float(pos.qty) > 0
    if is_long and plpc >= TAKE_PROFIT_PCT:
        log.info("%s long take profit: +%.2f%%", pos.symbol, plpc * 100)
        return True
    # unrealized_plpc is profit-signed for both sides (positive = winning),
    # so shorts also take profit at plpc >= +TAKE_PROFIT_PCT.
    if not is_long and plpc >= TAKE_PROFIT_PCT:
        log.info("%s short take profit: +%.2f%%", pos.symbol, plpc * 100)
        return True
    return False


def stop_loss_breached(pos: Position) -> bool:
    """
    Safety net behind the native GTC stop orders: if a stop order is lost
    (cancelled, expired, rejected), the daily run still closes any position
    past the stop-loss threshold.
    """
    try:
        plpc = float(pos.unrealized_plpc)
    except (TypeError, ValueError):
        return False

    if plpc <= -STOP_LOSS_PCT:
        log.warning("%s past stop-loss threshold: %.2f%% (native stop missing or not filled)",
                    pos.symbol, plpc * 100)
        return True
    return False


def positions_missing_stop(client: TradingClient) -> list[str]:
    """
    Symbols with an open position but no live GTC stop order.

    A native stop protects intraday; stop_loss_breached() above is only a
    once-per-daily-run backstop against plpc, so a stop that gets cancelled
    (e.g. mid-resize) and never replaced can leave a position unprotected
    for a full trading day or more before that backstop would even see it.
    """
    positions = client.get_all_positions()
    # Explicit limit: get_orders defaults to 50, which silently truncates once
    # the book has more open orders than that (this cost a real bug once).
    open_orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
    protected = {o.symbol for o in open_orders if o.order_type == OrderType.STOP}
    missing = [p.symbol for p in positions if p.symbol not in protected]
    for symbol in missing:
        log.error("%s: open position has no live stop-loss order", symbol)
    return missing
