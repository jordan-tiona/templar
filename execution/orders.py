"""Order submission via Alpaca."""

import json
import logging
import time
import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, GetOrdersRequest, StopLossRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderClass

from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, STOP_LOSS_PCT
from .risk import (
    get_client,
    drawdown_breached,
    max_position_value,
    position_value,
    current_position,
    take_profit_triggered,
    stop_loss_breached,
)

log = logging.getLogger(__name__)

_data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def _live_price(symbol: str, fallback: float) -> float:
    """
    Latest trade price, used to size and submit orders. `fallback` (the prior
    day's close, used for signal generation) can be badly stale by the time
    orders go out -- a stock that gaps overnight on news will size a "5% of
    equity" position against last night's price and land at 2x that once the
    market order actually fills at the gapped-up price. Re-pricing right
    before submission keeps sizing (and therefore MAX_POSITION_PCT) honest.
    """
    try:
        trade = _data_client.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=symbol)
        )[symbol]
        if trade.price and trade.price > 0:
            return float(trade.price)
    except Exception as e:
        log.warning("%s: could not fetch live price (%s), falling back to $%.2f", symbol, e, fallback)
    return fallback


def _target_qty(client: TradingClient, symbol: str, price: float, n_targets: int = 0) -> int:
    """Shares to open for a new long or short position (same sizing both sides)."""
    max_val = max_position_value(client, n_targets)
    current_val = position_value(client, symbol)
    available = max(0.0, max_val - current_val)
    if available < price:
        return 0
    return int(available // price)


def _cancel_open_orders(client: TradingClient, symbol: str, dry_run: bool) -> None:
    """
    Cancel all open orders for a symbol (cleans up GTC stop orders on position close),
    then wait for the cancels to actually clear. Alpaca cancellation is async: the
    shares stay 'held_for_orders' for a moment after cancel_order_by_id returns, so a
    close_position call issued immediately after can fail with "insufficient qty
    available" even though the order was just cancelled.
    """
    try:
        open_orders = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol],
        ))
        if not open_orders:
            return
        for order in open_orders:
            log.debug("Cancelling order %s for %s", order.id, symbol)
            if not dry_run:
                client.cancel_order_by_id(order.id)
        if dry_run:
            return
        for _ in range(5):
            still_open = client.get_orders(filter=GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                symbols=[symbol],
            ))
            if not still_open:
                return
            time.sleep(1.0)
        log.warning("%s: open orders still present 5s after cancel", symbol)
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

    def _build_request(stop: float) -> MarketOrderRequest:
        # TIF must be GTC: OTO child legs inherit the parent's TIF, so a DAY
        # parent makes the stop-loss leg expire at the close of entry day,
        # leaving the position unprotected.
        return MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.OTO,
            stop_loss=StopLossRequest(stop_price=stop),
        )

    try:
        order = client.submit_order(_build_request(stop_price))
        return str(order.id)
    except APIError as e:
        # `price` is the prior close used for sizing; by order time the stock may
        # have gapped past the stop threshold, so Alpaca rejects the stale stop
        # against its own fill reference (base_price). Retry once against that.
        try:
            body = json.loads(str(e))
        except (TypeError, ValueError):
            body = {}
        base_price = body.get("base_price")
        if base_price is None or "stop_loss.stop_price" not in body.get("message", ""):
            log.error("Failed to submit %s order for %s: %s", action, symbol, e)
            return None
        base_price = float(base_price)
        retry_stop = round(base_price * (1 - STOP_LOSS_PCT), 2) if is_long else round(base_price * (1 + STOP_LOSS_PCT), 2)
        log.warning(
            "%s %s: stop $%.2f rejected vs base_price $%.2f (gap since last close) -- retrying at $%.2f",
            symbol, action, stop_price, base_price, retry_stop,
        )
        try:
            order = client.submit_order(_build_request(retry_stop))
            return str(order.id)
        except Exception as e2:
            log.error("Failed to submit %s order for %s on retry: %s", action, symbol, e2)
            return None
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
    # Size against the full intended book (before take-profit/borrowability
    # drops) so per-position budgets don't inflate as names fall out.
    n_targets = len(buy_symbols) + len(short_symbols)

    try:
        open_positions = client.get_all_positions()
    except Exception as e:
        log.error("Failed to fetch open positions: %s", e)
        open_positions = []

    # --- Step 0: close positions that hit take-profit or breached stop-loss ---
    # Stop-loss here is a safety net; the native GTC stop order should fire first.
    took_profit = set()
    for pos in open_positions:
        if take_profit_triggered(pos):
            action = "TAKE_PROFIT"
        elif stop_loss_breached(pos):
            action = "STOP_LOSS"
        else:
            continue
        sym = pos.symbol
        _cancel_open_orders(client, sym, dry_run)
        log.info("%s: %s - closing position", sym, action.lower().replace("_", "-"))
        if not dry_run:
            try:
                client.close_position(sym)
                records.append({"symbol": sym, "action": action,
                                "qty": abs(int(float(pos.qty))), "order_id": "close"})
            except Exception as e:
                log.error("Failed to close %s on %s: %s", sym, action.lower(), e)
        else:
            records.append({"symbol": sym, "action": action,
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
        price = _live_price(symbol, price)
        qty = _target_qty(client, symbol, price, n_targets)
        if qty <= 0:
            log.info("BUY %s: insufficient funds", symbol)
            continue
        order_id = _open_position(client, symbol, qty, price, OrderSide.BUY, dry_run)
        if order_id:
            records.append({"symbol": symbol, "action": "BUY", "qty": qty, "order_id": order_id})

    # --- Step 3: open / top-up SHORT positions ---
    # Filter out stocks that can't be shorted or aren't easy-to-borrow.
    # In paper trading Alpaca marks most stocks as shortable; this guard is critical for live.
    borrowable = set()
    for symbol in short_symbols:
        try:
            asset = client.get_asset(symbol)
            if not asset.shortable:
                log.warning("SHORT %s: not shortable on Alpaca, skipping", symbol)
                continue
            if not asset.easy_to_borrow:
                log.warning("SHORT %s: hard-to-borrow (CTB costs may erase edge), skipping", symbol)
                continue
            borrowable.add(symbol)
        except Exception as e:
            log.warning("SHORT %s: could not verify borrowability (%s), skipping", symbol, e)
    short_symbols = borrowable

    for symbol in short_symbols:
        price = prices.get(symbol)
        if not price or price <= 0:
            log.warning("No price for %s, skipping SHORT", symbol)
            continue
        pos = current_position(client, symbol)
        if pos is not None and float(pos.qty) < 0:
            log.info("SHORT %s: already short, holding", symbol)
            continue
        price = _live_price(symbol, price)
        qty = _target_qty(client, symbol, price, n_targets)
        if qty <= 0:
            log.info("SHORT %s: insufficient funds", symbol)
            continue
        order_id = _open_position(client, symbol, qty, price, OrderSide.SELL, dry_run)
        if order_id:
            records.append({"symbol": symbol, "action": "SHORT", "qty": qty, "order_id": order_id})

    return records
