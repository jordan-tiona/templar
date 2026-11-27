"""Read-only: page through ALL orders (every status) using `until`, since the API caps each
page at 500. Writes postmortem/data/orders.csv (replaces the truncated first pull)."""
import time, os
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "postmortem" / "data"
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
c = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)
orders, seen, until = [], set(), None
while True:
    req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500, direction="desc", nested=False)
    if until is not None: req.until = until
    batch = c.get_orders(filter=req)
    if not batch: break
    new = 0
    for o in batch:
        if o.id in seen: continue
        seen.add(o.id); new += 1
        d = o.model_dump(); d.pop("legs", None)
        orders.append({k: (str(v) if v is not None else None) for k, v in d.items()})
    oldest = min(o.submitted_at for o in batch)
    print(f"batch {len(batch)} new {new} oldest {oldest}")
    if new == 0: break
    until = oldest
    time.sleep(0.3)
df = pd.DataFrame(orders)
df.to_csv(OUT / "orders.csv", index=False)
print("orders total:", len(df)); print(df.status.value_counts().to_dict()); print(df.type.value_counts().to_dict() if "type" in df else df.order_type.value_counts().to_dict())
print("submitted range:", df.submitted_at.min(), "->", df.submitted_at.max())
