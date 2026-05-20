"""Order submission via Alpaca."""

import logging
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

from config.settings import STOP_LOSS_PCT
from .risk import (
    get_client,
    drawdown_breached,
    max_position_value,
    position_value,
    current_position,
    take_profit_triggered,
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


def _cancel_open_orders(client: TradingClient, symbol: str, dry_run: bool) -> None:
    """Cancel all open orders for a symbol (cleans up GTC stop orders on position close)."""
    try:
        open_orders = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol],
        ))
        for order in open_orders:
            log.debug("Cancelling order %s for %s", order.id, symbol)
            if not dry_run:
                client.cancel_order_by_id(order.id)
    except Exception as e:
        log.warning("Could not cancel orders for %s: %s", symbol, e)


def _submit_stop_order(
    client: TradingClient,
    symbol: str,
    qty: int,
    entry_price: float,
    is_long: bool,
    dry_run: bool,
) -> None:
    """Submit a GTC stop-loss order at STOP_LOSS_PCT from entry price."""
    if is_long:
        stop_price = round(entry_price * (1 - STOP_LOSS_PCT), 2)
        side = OrderSide.SELL
    else:
        stop_price = round(entry_price * (1 + STOP_LOSS_PCT), 2)
        side = OrderSide.BUY

    log.info("%s: GTC stop at $%.2f (%+.0f%%)", symbol, stop_price,
             -STOP_LOSS_PCT * 100 if is_long else STOP_LOSS_PCT * 100)

    if not dry_run:
        try:
            req = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                stop_price=stop_price,
                time_in_force=TimeInForce.GTC,
            )
            client.submit_order(req)
        except Exception as e:
            log.error("Failed to submit stop order for %s: %s", symbol, e)


def execute_signals(
    signals: pd.DataFrame,
    prices: dict[str, float],
    dry_run: bool = False,
) -> list[dict]:
    """
    Daily rebalance: take profits, close stale positions, open/maintain BUY and SHORT targets.
    Stop losses are handled automatically by Alpaca GTC stop orders placed at position open.
    """
    client = get_client()
    records = []

    if drawdown_breached(client):
        log.warning("Drawdown limit hit — no orders will be placed today")
        return records

    buy_symbols = set(signals[signals["signal"] == "BUY"].index)
    short_symbols = set(signals[signals["signal"] == "SHORT"].index)

    try:
        open_positions = client.get_all_positions()
    except Exception as e:
        log.error("Failed to fetch open positions: %s", e)
        open_positions = []

    # --- Step 0: close positions that hit take-profit target ---
    took_profit = set()
    for pos in open_positions:
        if take_profit_triggered(pos):
            sym = pos.symbol
            _cancel_open_orders(client, sym, dry_run)
            log.info("%s: take-profit — closing position", sym)
            if not dry_run:
                try:
                    client.close_position(sym)
                    records.append({"symbol": sym, "action": "TAKE_PROFIT",
                                    "qty": abs(int(float(pos.qty))), "order_id": "close"})
                except Exception as e:
                    log.error("Failed to close %s on take-profit: %s", sym, e)
            else:
                records.append({"symbol": sym, "action": "TAKE_PROFIT",
                                "qty": abs(int(float(pos.qty))), "order_id": "dry_run"})
            took_profit.add(sym)
            buy_symbols.discard(sym)
            short_symbols.discard(sym)

    # --- Step 1: close positions no longer in the target set ---
    for pos in open_positions:
        sym = pos.symbol
        if sym in took_profit:
            continue
        qty = float(pos.qty)
        if qty > 0 and sym in buy_symbols:
            continue  # long we still want — keep
        if qty < 0 and sym in short_symbols:
            continue  # short we still want — keep

        side_label = "long" if qty > 0 else "short"
        _cancel_open_orders(client, sym, dry_run)
        log.info("%s: closing %s position (%d shares)", sym, side_label, abs(int(qty)))
        if not dry_run:
            try:
                client.close_position(sym)
                records.append({"symbol": sym, "action": "CLOSE",
                                "qty": abs(int(qty)), "order_id": "close"})
            except Exception as e:
                log.error("Failed to close %s: %s", sym, e)
        else:
            records.append({"symbol": sym, "action": "CLOSE",
                            "qty": abs(int(qty)), "order_id": "dry_run"})

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
            order_id = str(order.id)
            records.append({"symbol": symbol, "action": "BUY", "qty": qty, "order_id": order_id})
            _submit_stop_order(client, symbol, qty, price, is_long=True, dry_run=False)
        else:
            records.append({"symbol": symbol, "action": "BUY", "qty": qty, "order_id": "dry_run"})
            _submit_stop_order(client, symbol, qty, price, is_long=True, dry_run=True)

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
            order_id = str(order.id)
            records.append({"symbol": symbol, "action": "SHORT", "qty": qty, "order_id": order_id})
            _submit_stop_order(client, symbol, qty, price, is_long=False, dry_run=False)
        else:
            records.append({"symbol": symbol, "action": "SHORT", "qty": qty, "order_id": "dry_run"})
            _submit_stop_order(client, symbol, qty, price, is_long=False, dry_run=True)

    return records
