"""Order submission via Alpaca."""

import logging
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, GetOrdersRequest, StopLossRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderClass

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


def _open_position(
    client: TradingClient,
    symbol: str,
    qty: int,
    price: float,
    side: OrderSide,
    dry_run: bool,
) -> str | None:
    """
    Submit a bracket market order with an attached GTC stop-loss.
    Using bracket order avoids Alpaca's wash-trade rejection that occurs when
    a stop order is submitted separately against a pending market order.
    """
    is_long = side == OrderSide.BUY
    if is_long:
        stop_price = round(price * (1 - STOP_LOSS_PCT), 2)
    else:
        stop_price = round(price * (1 + STOP_LOSS_PCT), 2)

    action = "BUY" if is_long else "SHORT"
    log.info("%s %s %d shares @ ~$%.2f | stop at $%.2f (%+.0f%%)",
             symbol, action, qty, price, stop_price,
             -STOP_LOSS_PCT * 100 if is_long else STOP_LOSS_PCT * 100)

    if dry_run:
        return "dry_run"

    try:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.OTO,
            stop_loss=StopLossRequest(stop_price=stop_price),
        )
        order = client.submit_order(req)
        return str(order.id)
    except Exception as e:
        log.error("Failed to submit %s order for %s: %s", action, symbol, e)
        return None


def execute_signals(
    signals: pd.DataFrame,
    prices: dict[str, float],
    dry_run: bool = False,
) -> list[dict]:
    """
    Daily rebalance: take profits, close stale positions, open/maintain BUY and SHORT targets.
    Stop losses are attached as OTO bracket orders at position open.
    """
    client = get_client()
    records = []

    if drawdown_breached(client):
        log.warning("Drawdown limit hit - no orders will be placed today")
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
            log.info("%s: take-profit - closing position", sym)
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
            continue  # long we still want - keep
        if qty < 0 and sym in short_symbols:
            continue  # short we still want - keep

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
        order_id = _open_position(client, symbol, qty, price, OrderSide.BUY, dry_run)
        if order_id:
            records.append({"symbol": symbol, "action": "BUY", "qty": qty, "order_id": order_id})

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
        order_id = _open_position(client, symbol, qty, price, OrderSide.SELL, dry_run)
        if order_id:
            records.append({"symbol": symbol, "action": "SHORT", "qty": qty, "order_id": order_id})

    return records
