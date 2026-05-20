"""Order submission via Alpaca."""

import logging
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from .risk import (
    get_client,
    drawdown_breached,
    max_position_value,
    position_value,
    current_position,
)

log = logging.getLogger(__name__)


def _target_qty(client: TradingClient, symbol: str, price: float) -> int:
    """Shares to open for a new long or short position (same sizing both sides)."""
    max_val = max_position_value(client)
    current_val = position_value(client, symbol)
    available = max(0.0, max_val - current_val)
    if available < price:
        return 0
    return int(available // price)


def execute_signals(
    signals: pd.DataFrame,
    prices: dict[str, float],
    dry_run: bool = False,
) -> list[dict]:
    """
    Daily rebalance: close stale positions, open/maintain BUY and SHORT targets.
    signals: DataFrame with index=symbol, columns including 'signal' (BUY/SHORT/HOLD).
    """
    client = get_client()
    records = []

    if drawdown_breached(client):
        log.warning("Drawdown limit hit — no orders will be placed today")
        return records

    buy_symbols = set(signals[signals["signal"] == "BUY"].index)
    short_symbols = set(signals[signals["signal"] == "SHORT"].index)

    # --- Step 1: close positions no longer in the target set ---
    try:
        open_positions = client.get_all_positions()
    except Exception as e:
        log.error("Failed to fetch open positions: %s", e)
        open_positions = []

    for pos in open_positions:
        sym = pos.symbol
        qty = float(pos.qty)
        if qty > 0 and sym in buy_symbols:
            continue  # long position we still want — keep
        if qty < 0 and sym in short_symbols:
            continue  # short position we still want — keep

        side_label = "long" if qty > 0 else "short"
        log.info("%s: closing %s position (%d shares)", sym, side_label, abs(int(qty)))
        if not dry_run:
            try:
                client.close_position(sym)
                records.append({"symbol": sym, "action": "CLOSE", "qty": abs(int(qty)), "order_id": "close"})
            except Exception as e:
                log.error("Failed to close %s: %s", sym, e)
        else:
            records.append({"symbol": sym, "action": "CLOSE", "qty": abs(int(qty)), "order_id": "dry_run"})

    # --- Step 2: open / top-up BUY positions ---
    for symbol in buy_symbols:
        price = prices.get(symbol)
        if not price or price <= 0:
            log.warning("No price for %s, skipping BUY", symbol)
            continue
        pos = current_position(client, symbol)
        if pos is not None and float(pos.qty) > 0:
            log.info("BUY %s: already long, holding", symbol)
            continue
        qty = _target_qty(client, symbol, price)
        if qty <= 0:
            log.info("BUY %s: insufficient funds", symbol)
            continue
        log.info("%s BUY %d shares @ ~$%.2f", symbol, qty, price)
        if not dry_run:
            req = MarketOrderRequest(
                symbol=symbol, qty=qty,
                side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
            )
            order = client.submit_order(req)
            records.append({"symbol": symbol, "action": "BUY", "qty": qty, "order_id": str(order.id)})
        else:
            records.append({"symbol": symbol, "action": "BUY", "qty": qty, "order_id": "dry_run"})

    # --- Step 3: open / top-up SHORT positions ---
    for symbol in short_symbols:
        price = prices.get(symbol)
        if not price or price <= 0:
            log.warning("No price for %s, skipping SHORT", symbol)
            continue
        pos = current_position(client, symbol)
        if pos is not None and float(pos.qty) < 0:
            log.info("SHORT %s: already short, holding", symbol)
            continue
        qty = _target_qty(client, symbol, price)
        if qty <= 0:
            log.info("SHORT %s: insufficient funds", symbol)
            continue
        log.info("%s SHORT %d shares @ ~$%.2f", symbol, qty, price)
        if not dry_run:
            req = MarketOrderRequest(
                symbol=symbol, qty=qty,
                side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
            )
            order = client.submit_order(req)
            records.append({"symbol": symbol, "action": "SHORT", "qty": qty, "order_id": str(order.id)})
        else:
            records.append({"symbol": symbol, "action": "SHORT", "qty": qty, "order_id": "dry_run"})

    return records
