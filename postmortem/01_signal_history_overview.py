"""Summarize the logged signal history: per-run counts of BUY/SHORT and the xgb_pred
distribution, to show how the min-max score normalization drives the BUY count."""
import json, sys
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path(__file__).resolve().parent.parent
rows = []
for line in open(ROOT / "data/cache/signals_history.jsonl"):
    d = json.loads(line)
    ts = d["generated_at"]
    for s in d["signals"]:
        rows.append({"generated_at": ts, **s})
df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["generated_at"]).dt.date
print("runs:", df["generated_at"].nunique(), "rows:", len(df))
g = df.groupby("generated_at")
summ = pd.DataFrame({
    "n": g.size(),
    "buy": g.apply(lambda x: (x.signal == "BUY").sum()),
    "short": g.apply(lambda x: (x.signal == "SHORT").sum()),
    "xgb_min": g.xgb_pred.min(), "xgb_p50": g.xgb_pred.median(), "xgb_p90": g.xgb_pred.quantile(0.9),
    "xgb_max": g.xgb_pred.max(),
    "n_score_ge_0.6": g.apply(lambda x: (x.ensemble_score >= 0.6).sum()),
    "max_sym": g.apply(lambda x: x.loc[x.xgb_pred.idxmax(), "symbol"] if x.xgb_pred.notna().any() else None),
})
summ.index = pd.to_datetime(summ.index).strftime("%Y-%m-%d %H:%M")
pd.set_option("display.width", 200)
print(summ.to_string(float_format=lambda v: f"{v:.3f}"))
summ.to_csv(ROOT / "postmortem/out/signal_runs_summary.csv")
