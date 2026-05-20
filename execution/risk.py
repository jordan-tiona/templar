"""Portfolio-level risk checks before order submission."""

import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.models import Position

from config.settings import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY,
    MAX_POSITION_PCT, MAX_DRAWDOWN_PCT,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
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


def stop_loss_triggered(pos: Position) -> bool:
    """
    Return True if position should be force-closed due to stop loss or take profit.
    Works for both long (qty > 0) and short (qty < 0) positions.
    Alpaca provides unrealized_plpc as a decimal (e.g. -0.07 = -7%).
    """
    try:
        plpc = float(pos.unrealized_plpc)  # positive = gain, negative = loss
    except (TypeError, ValueError):
        return False

    qty = float(pos.qty)
    is_long = qty > 0

    if is_long:
        if plpc <= -STOP_LOSS_PCT:
            log.warning("%s long stop loss: %.2f%%", pos.symbol, plpc * 100)
            return True
        if plpc >= TAKE_PROFIT_PCT:
            log.info("%s long take profit: %.2f%%", pos.symbol, plpc * 100)
            return True
    else:
        # Short: profit when price falls (plpc > 0 means we're losing on a short)
        if plpc >= STOP_LOSS_PCT:
            log.warning("%s short stop loss: %.2f%%", pos.symbol, plpc * 100)
            return True
        if plpc <= -TAKE_PROFIT_PCT:
            log.info("%s short take profit: %.2f%%", pos.symbol, plpc * 100)
            return True

    return False
