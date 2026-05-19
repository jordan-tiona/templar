"""Order submission via Alpaca."""

import logging
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from .risk import (
    get_client,
    drawdown_breached,
    max_position_value,
    position_value,
    current_position,
)

log = logging.getLogger(__name__)


def _shares_to_buy(client: TradingClient, symbol: str, latest_price: float) -> int:
    max_val = max_position_value(client)
    current_val = position_value(client, symbol)
    available = max(0.0, max_val - current_val)
    if available < latest_price:
        return 0
    return int(available // latest_price)


def execute_signals(
    signals: pd.DataFrame,
    prices: dict[str, float],
    dry_run: bool = False,
) -> list[dict]:
    """
    Submit orders based on signal DataFrame (index=symbol, column=signal).
    prices: latest close price per symbol.
    dry_run: log intended orders but don't submit.
    Returns list of order records.
    """
    client = get_client()
    records = []

    if drawdown_breached(client):
        log.warning("Drawdown limit hit — no orders will be placed today")
        return records

    for symbol, row in signals.iterrows():
        signal = row["signal"]
        price = prices.get(symbol)

        if price is None or price <= 0:
            log.warning("No price for %s, skipping", symbol)
            continue

        if signal == "BUY":
            qty = _shares_to_buy(client, symbol, price)
            if qty <= 0:
                log.info("BUY %s: position already at max or insufficient funds", symbol)
                continue

            log.info("%s BUY %d shares @ ~$%.2f", symbol, qty, price)
            if not dry_run:
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                order = client.submit_order(req)
                records.append({"symbol": symbol, "action": "BUY", "qty": qty, "order_id": str(order.id)})
            else:
                records.append({"symbol": symbol, "action": "BUY", "qty": qty, "order_id": "dry_run"})

        elif signal == "SELL":
            pos = current_position(client, symbol)
            if pos is None:
                log.info("SELL %s: no position to close", symbol)
                continue

            qty = abs(int(float(pos.qty)))
            log.info("%s SELL %d shares (close position)", symbol, qty)
            if not dry_run:
                try:
                    client.close_position(symbol)
                    records.append({"symbol": symbol, "action": "SELL", "qty": qty, "order_id": "close"})
                except Exception as e:
                    log.error("Failed to close %s: %s", symbol, e)
            else:
                records.append({"symbol": symbol, "action": "SELL", "qty": qty, "order_id": "dry_run"})

        # HOLD: do nothing

    return records
