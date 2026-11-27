"""Read-only market-data pull: daily bars (SIP feed) for SPY and every symbol that was ever
traded, 2026-04-01 .. today, saved to postmortem/data/bars_live_period.csv. Also pulls
minute bars around the open for a sample of fill days later if needed (not here)."""
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "postmortem" / "data"
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
fills = pd.read_csv(OUT / "fills.csv")
syms = sorted(set(fills.symbol.unique()) | {"SPY"})
print("symbols:", len(syms))
cl = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))
frames = []
for i in range(0, len(syms), 50):
    chunk = syms[i:i+50]
    for feed in ("sip", None):
        try:
            kw = dict(symbol_or_symbols=chunk, timeframe=TimeFrame.Day, start=datetime(2026, 4, 1), end=datetime.now(timezone.utc) - timedelta(minutes=16))
            if feed: kw["feed"] = feed
            df = cl.get_stock_bars(StockBarsRequest(**kw)).df.reset_index()
            df["feed"] = feed or "default"
            frames.append(df); print(f"chunk {i}: {len(df)} rows via feed={feed}")
            break
        except Exception as e:
            print(f"chunk {i} feed={feed} failed: {str(e)[:120]}")
bars = pd.concat(frames)
bars["date"] = pd.to_datetime(bars["timestamp"]).dt.tz_convert("America/New_York").dt.date
bars.to_csv(OUT / "bars_live_period.csv", index=False)
print(bars.groupby("feed").size().to_dict(), "date range", bars.date.min(), bars.date.max())
print(bars[bars.symbol=="SPY"].tail(3)[["date","open","close","volume"]].to_string(index=False))
