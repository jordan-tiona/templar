"""Survey the cached raw bars: history length per symbol, last date, and where each
symbol's train/val/test boundaries fall in calendar time under the per-symbol split."""
from pathlib import Path
import pandas as pd, numpy as np, glob, os
ROOT = Path(__file__).resolve().parent.parent
rows = []
for p in sorted(glob.glob(str(ROOT / "data/raw/*.parquet"))):
    df = pd.read_parquet(p)
    n = len(df)
    # mimic dataset.py: add_target drops last 20 rows; build_features dropna drops ~50 warmup rows at the start
    n_eff = max(n - 20 - 50, 0)
    t1 = int(n_eff * 0.70); t2 = int(n_eff * 0.85)
    idx = df.index[50:50 + n_eff] if n_eff else df.index[:0]
    rows.append(dict(sym=os.path.basename(p)[:-8], n=n, first=df.index.min().date(), last=df.index.max().date(),
                     val_start=idx[t1].date() if n_eff and t1 < len(idx) else None,
                     test_start=idx[t2].date() if n_eff and t2 < len(idx) else None))
r = pd.DataFrame(rows)
print("symbols:", len(r))
print("last-date distribution:", r["last"].astype(str).value_counts().head(5).to_dict())
print("bars: min", r.n.min(), "median", int(r.n.median()), "max", r.n.max())
full = r[r.n >= r.n.median() - 5]
print(f"symbols with full history (>= median-5 bars): {len(full)}; their test_start range: {full.test_start.min()} .. {full.test_start.max()}")
print("short-history symbols (their 'test' window starts later than everyone else's):")
print(r[r.n < r.n.median() - 5].sort_values("n").to_string(index=False))
print("\nraw columns:", list(pd.read_parquet(ROOT / "data/raw/AAPL.parquet").columns))
print(pd.read_parquet(ROOT / "data/raw/AAPL.parquet").tail(3).to_string())
r.to_csv(ROOT / "postmortem/out/data_coverage.csv", index=False)
