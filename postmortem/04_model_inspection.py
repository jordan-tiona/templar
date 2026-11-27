"""What did the XGBoost model actually learn? Uses the latest explain cache (SHAP per symbol)
and the latest feature snapshot. Also compares xgb_pred to simple features across symbols."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parent.parent
ex = json.loads((ROOT / "data/cache/latest_explain.json").read_text())
rows, shap_rows = [], []
for sym, e in ex.items():
    snap = e.get("snapshot") or {}
    rows.append(dict(symbol=sym, xgb_pred=e.get("xgb_pred"), signal=e.get("signal"), score=e.get("ensemble_score"), **snap))
    for s in e.get("shap", []): shap_rows.append(dict(symbol=sym, feature=s["feature"], value=s["value"]))
df = pd.DataFrame(rows); sh = pd.DataFrame(shap_rows)
print("explain date:", {e.get("date") for e in ex.values()}, "symbols:", len(df))
print("\nmean |SHAP| by feature (share of total), latest run:")
imp = sh.assign(a=sh.value.abs()).groupby("feature").a.mean().sort_values(ascending=False)
print((imp / imp.sum() * 100).round(1).to_string())
print("\nSpearman corr of xgb_pred with each snapshot feature across the universe (latest run):")
df["atr_pct"] = df.ATRr_14 / df.close
out = []
for c in ["return_1d", "return_5d", "return_20d", "RSI_14", "STOCHk_14_3_3", "BBP_20_2_0_2_0", "BBB_20_2_0_2_0", "atr_pct", "ATRr_14", "close", "volume", "OBV", "VWAP_D", "MACDh_12_26_9", "sentiment_mean", "sentiment_count", "EMA_50"]:
    d = df.dropna(subset=["xgb_pred", c])
    out.append(dict(feature=c, rho=spearmanr(d.xgb_pred, d[c]).statistic, n=len(d)))
print(pd.DataFrame(out).round(3).to_string(index=False))
print("\nxgb_pred distribution:", df.xgb_pred.describe().round(3).to_dict())
print("BUY names:", df[df.signal == "BUY"].symbol.tolist())
print("SHORT names:", df[df.signal == "SHORT"].symbol.tolist())
print("\nmean feature values by signal group:")
print(df.groupby("signal")[["return_5d", "return_20d", "RSI_14", "atr_pct", "sentiment_mean", "close"]].mean().round(3).to_string())
