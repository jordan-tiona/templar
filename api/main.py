"""
Templar Dashboard API.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config.settings import DATA_CACHE

app = FastAPI(title="Templar Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/portfolio")
def get_portfolio():
    from execution.risk import get_client
    try:
        client = get_client()
        account = client.get_account()
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        return {
            "equity": equity,
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "last_equity": last_equity,
            "day_pl": equity - last_equity,
            "day_pl_pct": (equity - last_equity) / last_equity if last_equity else 0.0,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Alpaca error: {e}")


@app.get("/api/positions")
def get_positions():
    from execution.risk import get_client
    try:
        client = get_client()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": "long" if float(p.qty) > 0 else "short",
                "avg_entry": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
            for p in client.get_all_positions()
        ]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Alpaca error: {e}")


@app.get("/api/signals/latest")
def get_latest_signals():
    path = DATA_CACHE / "latest_signals.json"
    if not path.exists():
        return {"generated_at": None, "signals": []}
    return json.loads(path.read_text())


@app.get("/api/signals/history")
def get_signals_history():
    path = DATA_CACHE / "signals_history.jsonl"
    if not path.exists():
        return []
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    return list(reversed(records))


@app.get("/api/signals/{symbol}/explain")
def get_explain(symbol: str):
    path = DATA_CACHE / "latest_explain.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No explain data. Run the pipeline first.")
    explain = json.loads(path.read_text())
    sym = symbol.upper()
    if sym not in explain:
        raise HTTPException(status_code=404, detail=f"{sym} not found in latest explain data.")
    return explain[sym]


@app.get("/api/watchlist")
def get_watchlist():
    from data.screener import get_watchlist
    return {"watchlist": get_watchlist()}
