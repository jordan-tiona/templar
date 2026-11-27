"""Read-only pull of the Alpaca paper account: account, positions, portfolio history,
all orders (every status), and all FILL activities. Writes CSV/JSON to postmortem/data/.
Nothing here places, modifies, or cancels orders."""
import json, sys, time
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "postmortem" / "data"
OUT.mkdir(parents=True, exist_ok=True)

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
import os
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (GetOrdersRequest, GetPortfolioHistoryRequest)
from alpaca.trading.enums import QueryOrderStatus

KEY = os.getenv("ALPACA_API_KEY"); SEC = os.getenv("ALPACA_SECRET_KEY")
assert KEY and SEC, "ALPACA keys missing from .env"
c = TradingClient(KEY, SEC, paper=True)

# --- account ---
a = c.get_account()
acct = {k: (str(v) if v is not None else None) for k, v in a.model_dump().items()}
(OUT / "account.json").write_text(json.dumps(acct, indent=2, default=str))
print("account", a.account_number, "status", a.status, "created", a.created_at)
print("equity", a.equity, "last_equity", a.last_equity, "cash", a.cash,
      "long_mv", a.long_market_value, "short_mv", a.short_market_value)

# --- positions ---
pos = c.get_all_positions()
pos_rows = [{k: (str(v) if v is not None else None) for k, v in p.model_dump().items()} for p in pos]
pd.DataFrame(pos_rows).to_csv(OUT / "positions.csv", index=False)
print("open positions:", len(pos))

# --- portfolio history (daily) ---
ph = None
for kwargs in (dict(period="1A", timeframe="1D"), dict(period="all", timeframe="1D"), dict(period="6M", timeframe="1D")):
    try:
        ph = c.get_portfolio_history(GetPortfolioHistoryRequest(**kwargs, extended_hours=False))
        print("portfolio history ok with", kwargs, "points:", len(ph.timestamp))
        break
    except Exception as e:
        print("portfolio history failed with", kwargs, "->", e)
if ph is not None:
    phdf = pd.DataFrame({
        "timestamp": [datetime.fromtimestamp(t, tz=timezone.utc) for t in ph.timestamp],
        "equity": ph.equity, "profit_loss": ph.profit_loss, "profit_loss_pct": ph.profit_loss_pct,
    })
    phdf["base_value"] = ph.base_value
    phdf.to_csv(OUT / "portfolio_history.csv", index=False)
    print(phdf.head(3).to_string()); print(phdf.tail(3).to_string())

# --- orders: all statuses, paginate backwards with `until` ---
orders = []
until = None
seen = set()
while True:
    req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500, direction="desc", nested=True)
    if until is not None:
        req.until = until
    batch = c.get_orders(filter=req)
    if not batch:
        break
    new = 0
    for o in batch:
        if o.id in seen:
            continue
        seen.add(o.id); new += 1
        d = o.model_dump()
        legs = d.pop("legs", None)
        d["n_legs"] = len(legs) if legs else 0
        orders.append({k: (str(v) if v is not None else None) for k, v in d.items()})
        # legs (stop-loss children) are also returned as top-level orders in ALL queries; keep them too
        for leg in (legs or []):
            if leg["id"] in seen:
                continue
            seen.add(leg["id"]); new += 1
            leg = dict(leg); leg.pop("legs", None); leg["n_legs"] = 0
            orders.append({k: (str(v) if v is not None else None) for k, v in leg.items()})
    oldest = min(o.submitted_at for o in batch)
    print(f"orders batch: {len(batch)} (new {new}), oldest submitted {oldest}")
    if new == 0 or len(batch) < 500:
        break
    until = oldest
    time.sleep(0.3)
odf = pd.DataFrame(orders)
odf.to_csv(OUT / "orders.csv", index=False)
print("orders total:", len(odf), "statuses:", odf["status"].value_counts().to_dict() if len(odf) else {})

# --- account activities via raw REST GET (TradingClient in alpaca-py 0.43 has no wrapper) ---
def _activities(activity_type=None):
    rows, page_token = [], None
    path = "/account/activities" + (f"/{activity_type}" if activity_type else "")
    while True:
        params = {"page_size": 100, "direction": "desc"}
        if page_token: params["page_token"] = page_token
        batch = c.get(path, params)
        if not batch: break
        rows.extend(batch)
        page_token = batch[-1]["id"]
        print(f"  {path}: +{len(batch)} (last {batch[-1].get('transaction_time') or batch[-1].get('date')})")
        if len(batch) < 100: break
        time.sleep(0.25)
    return rows
fills = _activities("FILL")
fdf = pd.DataFrame(fills); fdf.to_csv(OUT / "fills.csv", index=False)
print("fill activities total:", len(fdf))
allacts = _activities(None)
nta = [r for r in allacts if r.get("activity_type") != "FILL"]
pd.DataFrame(nta).to_csv(OUT / "non_trade_activities.csv", index=False)
print("non-trade activities:", len(nta), pd.Series([r["activity_type"] for r in nta]).value_counts().to_dict() if nta else {})
print("DONE")
