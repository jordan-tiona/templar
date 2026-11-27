"""Follow-ups: (a) P&L reconciliation vs equity; (b) robustness of the 20-day basket result;
(c) cost/benefit of the stop-then-re-enter loop; (d) turnover and cost run-rate."""
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent; D = ROOT / "postmortem/data"; O = ROOT / "postmortem/out"
ET = "America/New_York"
ep = pd.read_csv(O / "episodes.csv", parse_dates=["entry_date", "exit_date"])
legs = pd.read_csv(O / "fill_legs.csv", parse_dates=["date"])
ph = pd.read_csv(D / "portfolio_history.csv"); ph["date"] = (pd.to_datetime(ph.timestamp) - pd.Timedelta(days=1)).dt.date
nta = pd.read_csv(D / "non_trade_activities.csv"); nta["net_amount"] = nta.net_amount.astype(float)
pos = pd.read_csv(D / "positions.csv")
print("=== (a) reconciliation ===")
eq0, eq1 = 100000.0, float(ph[ph.date == pd.Timestamp("2026-11-17").date()].equity.iloc[0])
fees = nta[nta.activity_type == "FEE"].net_amount.sum(); divs = nta[nta.activity_type == "DIV"].net_amount.sum()
unreal = pos.unrealized_pl.astype(float).sum()
print(f"equity 2026-11-17 close {eq1:,.2f}; change since start {eq1-eq0:,.2f}")
print(f"realized (closed episodes) {ep.pnl.sum():,.2f} + Alpaca unrealized now {unreal:,.2f} + fees {fees:,.2f} + dividends {divs:,.2f} = {ep.pnl.sum()+unreal+fees+divs:,.2f}")
print("  (unrealized is as of now, 11-18 intraday; equity is 11-17 close, so a few hundred $ of mismatch is expected)")
print(f"execution cost total {legs.cost_usd.sum():,.2f} = {legs.cost_usd.sum()/abs(eq1-eq0)*100:.0f}% of the loss; traded notional {legs.notional.sum():,.0f} "
      f"= {legs.notional.sum()/ ((eq0+eq1)/2):.0f}x average equity over the span")
span_days = (pd.Timestamp("2026-11-17") - pd.Timestamp("2026-05-20")).days
print(f"cost run-rate: ${legs.cost_usd.sum()/span_days*365:,.0f} per year = {legs.cost_usd.sum()/span_days*365/((eq0+eq1)/2)*100:.0f}% of average equity per year")
big = legs[legs.date >= pd.Timestamp("2026-07-09")]
print(f"since Jul 9: traded ${big.notional.sum():,.0f} over {big.date.nunique()} fill days = ${big.notional.sum()/big.date.nunique():,.0f}/day; "
      f"cost ${big.cost_usd.sum()/big.date.nunique():,.0f}/day")

print("\n=== (c) stop -> same-side re-entry within 1 trading day: did the stop help? ===")
# compare the stop exit price with the re-entry price: for a long, re-entering higher than the stop fill = the stop cost money
bars = pd.read_csv(D / "bars_live_period.csv"); bars["date"] = pd.to_datetime(bars.date)
tdays = sorted(bars[bars.symbol == "SPY"].date.unique()); tdi = {d: i for i, d in enumerate(tdays)}
ep = ep.sort_values("entry_t"); rows = []
for sym, g in ep.groupby("symbol"):
    g = g.reset_index(drop=True)
    for i in range(len(g) - 1):
        a, b = g.loc[i], g.loc[i + 1]
        if a.exit_reason in ("NATIVE_STOP", "STOP_LOSS_BACKSTOP") and a.side == b.side and tdi.get(b.entry_date, 1e9) - tdi.get(a.exit_date, 0) <= 1:
            sgn = 1 if a.side == "long" else -1
            rows.append(dict(symbol=sym, side=a.side, stop_exit=a.avg_exit, reentry=b.avg_entry, qty=min(a.qty, b.qty),
                             gain_from_stop=(a.avg_exit - b.avg_entry) * sgn * min(a.qty, b.qty), next_pnl=b.pnl, next_reason=b.exit_reason))
r = pd.DataFrame(rows)
print(f"stop exits followed by same-side re-entry within 1 day: {len(r)} of {(ep.exit_reason.isin(['NATIVE_STOP','STOP_LOSS_BACKSTOP'])).sum()} stop exits")
print(f"  money saved by being out between stop and re-entry (positive = stop helped): ${r.gain_from_stop.sum():,.0f}  (median per event ${r.gain_from_stop.median():,.0f})")
print(f"  P&L of the re-entered position: ${r.next_pnl.sum():,.0f}; re-entry then stopped out again: {(r.next_reason.isin(['NATIVE_STOP','STOP_LOSS_BACKSTOP'])).mean()*100:.0f}%")
print("  by side:"); print(r.groupby("side").agg(n=("qty", "size"), saved=("gain_from_stop", "sum"), next_pnl=("next_pnl", "sum")).round(0).to_string())
# chains: how many times was the same symbol stopped out in a row within a 10-day window?
stops = ep[ep.exit_reason.isin(["NATIVE_STOP", "STOP_LOSS_BACKSTOP"])]
print("  symbols with the most stop-outs:", stops.groupby("symbol").size().sort_values(ascending=False).head(8).to_dict())
print("  stop-out loss by side:", stops.groupby("side").pnl.sum().round(0).to_dict(), " take-profit gain by side:", ep[ep.exit_reason == "TAKE_PROFIT"].groupby("side").pnl.sum().round(0).to_dict())

print("\n=== (b) 20-day basket robustness ===")
sig = []
for line in open(ROOT / "data/cache/signals_history.jsonl"):
    d = json.loads(line); ts = pd.Timestamp(d["generated_at"]).tz_localize("UTC").tz_convert(ET)
    for s in d["signals"]: sig.append(dict(run=ts, date=pd.Timestamp(ts.date()), **s))
sig = pd.DataFrame(sig).sort_values("run").groupby(["date", "symbol"]).last().reset_index()
op = bars.pivot(index="date", columns="symbol", values="open")
dates = list(op.index); di = {d: i for i, d in enumerate(dates)}
def fwd_oo(sym, d, h):
    i = di.get(d)
    if i is None or sym not in op.columns or i + h >= len(dates): return np.nan
    return op.iat[i + h, op.columns.get_loc(sym)] / op.iat[i, op.columns.get_loc(sym)] - 1
big = sig[(sig.date >= pd.Timestamp("2026-07-09")) & sig.date.isin(set(dates))].copy()
for h in (1, 5, 10, 20): big[f"r{h}"] = [fwd_oo(s, d, h) for s, d in zip(big.symbol, big.date)]
def ls(df, h, agg="mean"):
    g = df.dropna(subset=[f"r{h}"]).groupby(["date", "signal"])[f"r{h}"].agg(agg).unstack()
    return (g["BUY"] - g["SHORT"]) * 100
for h in (5, 10, 20):
    m = ls(big, h); md = ls(big, h, "median"); ex = ls(big[big.symbol != "MRNA"], h)
    print(f"h={h:2d}: LS mean-of-means {m.mean():+.2f}% (n={len(m)}, hit {(m>0).mean():.0%}) | mean-of-medians {md.mean():+.2f}% (hit {(md>0).mean():.0%}) | ex-MRNA {ex.mean():+.2f}%")
m20 = ls(big, 20)
print("20d long-short by signal month (non-overlapping-ish view):")
print(m20.groupby(m20.index.to_period("M")).agg(["mean", "median", "count"]).round(2).to_string())
# non-overlapping sample: every 20th signal day
nov = m20.iloc[::20]; print(f"non-overlapping 20d samples (every 20th day): {nov.round(2).to_dict()}  mean {nov.mean():+.2f}%")
# what did the actual positions do over the next 20 days vs what the strategy realized?  entry-day picks held 20d, gross, 1x long/1x short
print("\nper-signal-day BUY/SHORT basket mean 20d returns (first 8 and last 8 complete days):")
g = big.dropna(subset=["r20"]).groupby(["date", "signal"]).r20.mean().unstack() * 100
print(pd.concat([g.head(8), g.tail(8)]).round(2).to_string())

print("\n=== (d) daily turnover ===")
ex = pd.read_csv(O / "exposure_daily.csv", parse_dates=["date"])
t = legs.groupby("date").notional.sum().rename("traded").to_frame().join(ex.set_index("date")[["long_mv", "short_mv", "equity"]])
t["gross_mv"] = t.long_mv - t.short_mv
t = t[t.index >= pd.Timestamp("2026-07-09")]
print(f"median daily traded notional / gross book: {(t.traded / t.gross_mv).median()*100:.0f}%   mean {(t.traded / t.gross_mv).mean()*100:.0f}%  -> book fully turns over every ~{1/(t.traded/t.gross_mv/2).median():.1f} trading days")
print(f"median daily traded notional / equity: {(t.traded / t.equity).median()*100:.0f}%")
