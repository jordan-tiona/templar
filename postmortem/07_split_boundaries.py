"""Where do the train / val / test boundaries actually fall in calendar time, using the project's own
split code on the processed feature frames written by the last training run?"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np
from models.dataset import prepare_combined, train_val_test_split, add_target, PREDICTION_HORIZON
frames = {s: pd.read_parquet(ROOT / f"data/processed/{s}.parquet") for s in ["AAPL", "MSFT", "JPM", "XOM", "SNDK", "KVUE"]}
# keep calendar dates alongside time_idx
dated = {}
for s, df in frames.items():
    t = add_target(df); t = t.copy(); t["time_idx"] = np.arange(len(t)); dated[s] = pd.Series(t.index, index=t["time_idx"].values)
comb = prepare_combined(frames)
tr, va, te = train_val_test_split(comb)
rows = []
for s in frames:
    d = dated[s]
    def rng(x):
        x = x[x.symbol == s]
        return (d[x.time_idx.min()].date(), d[x.time_idx.max()].date(), len(x)) if len(x) else (None, None, 0)
    a, b, c = rng(tr), rng(va), rng(te)
    rows.append(dict(symbol=s, bars=len(frames[s]), train=f"{a[0]} -> {a[1]} ({a[2]})", val=f"{b[0]} -> {b[1]} ({b[2]})", test=f"{c[0]} -> {c[1]} ({c[2]})"))
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))
print(f"\nhorizon {PREDICTION_HORIZON} trading days; test rows per full-history symbol ~{len(te[te.symbol=='AAPL'])} -> ~{len(te[te.symbol=='AAPL'])/PREDICTION_HORIZON:.0f} non-overlapping 20-day periods")
print("processed AAPL columns:", list(frames["AAPL"].columns))
print("sentiment_mean unique values in processed AAPL:", frames["AAPL"].sentiment_mean.unique()[:5], " sentiment_count:", frames["AAPL"].sentiment_count.unique()[:5])
