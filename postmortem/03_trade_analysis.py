"""Core trade analysis from the Alpaca fills/orders pulled by 00_*.py.
Builds a FIFO position ledger, round trips (flat -> open -> flat episodes), execution-cost
decomposition against the opening print, exposure over time, benchmark comparisons,
and prediction-vs-realized checks from data/cache/signals_history.jsonl.
Outputs: postmortem/out/*.csv and postmortem/out/summary_03.txt. Read-only; no API calls."""
import json, re, glob, io
from pathlib import Path
from collections import defaultdict, deque
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "postmortem" / "data"; O = ROOT / "postmortem" / "out"; O.mkdir(exist_ok=True)
buf = io.StringIO()
def P(*a):
    s = " ".join(str(x) for x in a); print(s); buf.write(s + "\n")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
ET = "America/New_York"

# ---------------------------------------------------------------- load
fills = pd.read_csv(D / "fills.csv")
fills["t"] = pd.to_datetime(fills.transaction_time, utc=True).dt.tz_convert(ET)
fills["date"] = fills.t.dt.date
fills["price"] = fills.price.astype(float); fills["qty"] = fills.qty.astype(float)
fills = fills.sort_values("t").reset_index(drop=True)
orders = pd.read_csv(D / "orders.csv")
orders["otype"] = orders["type"].str.replace("OrderType.", "", regex=False)
otype = orders.set_index("id")["otype"].to_dict()
ostop = orders.set_index("id")["stop_price"].astype(float).to_dict()
fills["otype"] = fills.order_id.map(otype).fillna("UNKNOWN")
fills["stop_price"] = fills.order_id.map(ostop)
bars = pd.read_csv(D / "bars_live_period.csv"); bars["date"] = pd.to_datetime(bars.date).dt.date
bars = bars.sort_values(["symbol", "date"])
opn = bars.set_index(["symbol", "date"])["open"].to_dict()
cls = bars.set_index(["symbol", "date"])["close"].to_dict()
trading_days = sorted(bars[bars.symbol == "SPY"].date.unique())
tdi = {d: i for i, d in enumerate(trading_days)}
ph = pd.read_csv(D / "portfolio_history.csv")
# Alpaca stamps each daily point at 01:00 UTC on the *following* calendar day; shift to the trading date
ph["date"] = (pd.to_datetime(ph.timestamp) - pd.Timedelta(days=1)).dt.date
ph = ph[ph.date >= pd.Timestamp("2026-05-19").date()].reset_index(drop=True)
positions = pd.read_csv(D / "positions.csv")
acct = json.loads((D / "account.json").read_text())

P("=" * 100); P("ACCOUNT"); P("=" * 100)
P(f"equity now {float(acct['equity']):,.2f}  last_equity {float(acct['last_equity']):,.2f}  open positions {len(positions)}  "
  f"long MV {float(acct['long_market_value']):,.0f}  short MV {float(acct['short_market_value']):,.0f}")
P(f"fills: {len(fills)} records, {fills.order_id.nunique()} orders, {fills.date.nunique()} fill days, {fills.symbol.nunique()} symbols, "
  f"{fills.date.min()} -> {fills.date.max()}")
P("fill side counts:", fills.side.value_counts().to_dict(), " by order type:", fills.otype.value_counts().to_dict())

# ---------------------------------------------------------------- FIFO ledger -> episodes
fills["sgn"] = np.where(fills.side == "buy", 1, -1)
lots = defaultdict(deque)
pos = defaultdict(float)
episodes = []
cur = {}
legs = []
for r in fills.itertuples():
    sym, q, px = r.symbol, r.qty * r.sgn, r.price
    before = pos[sym]
    remaining = q
    if before != 0 and np.sign(before) != np.sign(q):
        close_q = min(abs(q), abs(before)) * np.sign(q)
        ep = cur[sym]
        ep["exit_fills"].append((r.t, px, abs(close_q), r.order_id, r.otype, r.stop_price))
        remaining_close = abs(close_q)
        while remaining_close > 1e-9 and lots[sym]:
            lot = lots[sym][0]
            take = min(remaining_close, abs(lot[0]))
            ep["realized"] += (px - lot[1]) * take * np.sign(lot[0])
            lot[0] -= take * np.sign(lot[0]); remaining_close -= take
            if abs(lot[0]) < 1e-9: lots[sym].popleft()
        legs.append(dict(t=r.t, date=r.date, symbol=sym, kind="close", side=r.side, qty=abs(close_q), price=px,
                         otype=r.otype, stop_price=r.stop_price, order_id=r.order_id, sgn=r.sgn))
        pos[sym] = before + close_q
        remaining = q - close_q
        if abs(pos[sym]) < 1e-9:
            ep["exit_t"] = r.t; ep["exit_date"] = r.date
            episodes.append(ep); del cur[sym]; pos[sym] = 0.0
    if abs(remaining) > 1e-9:
        if sym not in cur:
            cur[sym] = dict(symbol=sym, side="long" if remaining > 0 else "short", entry_t=r.t, entry_date=r.date,
                            entry_fills=[], exit_fills=[], realized=0.0)
        cur[sym]["entry_fills"].append((r.t, px, abs(remaining), r.order_id, r.otype))
        lots[sym].append([remaining, px, r.t, r.order_id, r.otype])
        legs.append(dict(t=r.t, date=r.date, symbol=sym, kind="open", side=r.side, qty=abs(remaining), price=px,
                         otype=r.otype, stop_price=r.stop_price, order_id=r.order_id, sgn=r.sgn))
        pos[sym] += remaining

last_close = {s: g.iloc[-1].close for s, g in bars.groupby("symbol")}
open_eps = []
for sym, ep in cur.items():
    q = sum(f[2] for f in ep["entry_fills"]) - sum(f[2] for f in ep["exit_fills"])
    avg_in = sum(f[1] * f[2] for f in ep["entry_fills"]) / sum(f[2] for f in ep["entry_fills"])
    mtm = (last_close.get(sym, avg_in) - avg_in) * q * (1 if ep["side"] == "long" else -1)
    open_eps.append(dict(symbol=sym, side=ep["side"], entry_date=ep["entry_date"], qty=q, avg_entry=avg_in,
                         unrealized_mtm=mtm + ep["realized"]))
open_df = pd.DataFrame(open_eps)

rows = []
for ep in episodes:
    qin = sum(f[2] for f in ep["entry_fills"]); qout = sum(f[2] for f in ep["exit_fills"])
    avg_in = sum(f[1] * f[2] for f in ep["entry_fills"]) / qin
    avg_out = sum(f[1] * f[2] for f in ep["exit_fills"]) / qout
    exit_types = {f[4] for f in ep["exit_fills"]}
    rows.append(dict(symbol=ep["symbol"], side=ep["side"], entry_date=ep["entry_date"], exit_date=ep["exit_date"],
                     entry_t=ep["entry_t"], exit_t=ep["exit_t"], qty=qin, avg_entry=avg_in, avg_exit=avg_out,
                     notional=avg_in * qin, pnl=ep["realized"], ret=ep["realized"] / (avg_in * qin),
                     exit_type="STOP" if exit_types == {"STOP"} else ("MIXED" if "STOP" in exit_types else "MARKET"),
                     n_entry_fills=len(ep["entry_fills"]), n_exit_fills=len(ep["exit_fills"])))
ep_df = pd.DataFrame(rows)
ep_df["hold_tdays"] = [tdi.get(b, np.nan) - tdi.get(a, np.nan) for a, b in zip(ep_df.entry_date, ep_df.exit_date)]
ep_df["hold_cdays"] = [(b - a).days for a, b in zip(ep_df.entry_date, ep_df.exit_date)]

reasons = {}
for f in glob.glob(str(ROOT / "logs/daily_*.log")):
    d = pd.Timestamp(Path(f).stem.replace("daily_", "")).date()
    txt = open(f, "rb").read().replace(b"\x00", b"").decode("utf-8", "replace")
    for m in re.finditer(r"\|([A-Z.]+): (take-profit|stop-loss) - closing position", txt):
        reasons[(d, m.group(1))] = "TAKE_PROFIT" if m.group(2) == "take-profit" else "STOP_LOSS_BACKSTOP"
    for m in re.finditer(r"\|([A-Z.]+): closing (long|short) position", txt):
        reasons.setdefault((d, m.group(1)), "REBALANCE_DROP")
def _reason(r):
    if r.exit_type == "STOP": return "NATIVE_STOP"
    return reasons.get((r.exit_date, r.symbol), "MANUAL/OTHER")
ep_df["exit_reason"] = ep_df.apply(_reason, axis=1)
legs_df = pd.DataFrame(legs)

PERIODS = [("P1 May21-Jul08 small 50-sym book", "2026-05-20", "2026-07-08"),
           ("P2 Jul09-Aug11 150-sym book, exec bugs", "2026-07-09", "2026-08-11"),
           ("P3 Aug12-Oct03 clean eval, retrain OK", "2026-08-12", "2026-10-03"),
           ("P4 Oct04-Nov18 retrain job failing", "2026-10-04", "2026-11-18")]
def period_of(d):
    for name, a, b in PERIODS:
        if pd.Timestamp(a).date() <= d <= pd.Timestamp(b).date(): return name
    return "other"
ep_df["period"] = ep_df.exit_date.map(period_of)

P(); P("=" * 100); P("ROUND TRIPS (closed flat->open->flat episodes)"); P("=" * 100)
n_days = len([d for d in trading_days if fills.date.min() <= d <= fills.date.max()])
n_active = ep_df.exit_date.nunique()
P(f"closed episodes: {len(ep_df)}   still open: {len(open_df)}   trading days in span: {n_days}   "
  f"-> {len(ep_df)/n_days:.1f} round trips per trading day (span), {len(ep_df)/max(n_active,1):.1f} per day with any exit")
big_eps = ep_df[ep_df.entry_date >= pd.Timestamp("2026-07-09").date()]
nd_big = len([d for d in trading_days if pd.Timestamp("2026-07-09").date() <= d <= fills.date.max()])
P(f"since Jul 9 (full book): {len(big_eps)} closed episodes over {nd_big} trading days -> {len(big_eps)/nd_big:.1f} round trips per day; "
  f"trading days with at least one fill: {fills[fills.date >= pd.Timestamp('2026-07-09').date()].date.nunique()}")
P(f"holding time (trading days): mean {ep_df.hold_tdays.mean():.1f}  median {ep_df.hold_tdays.median():.0f}  "
  f"p25 {ep_df.hold_tdays.quantile(.25):.0f}  p75 {ep_df.hold_tdays.quantile(.75):.0f}  max {ep_df.hold_tdays.max():.0f}")
P("holding-time distribution (trading days, capped at 30):", ep_df.hold_tdays.clip(upper=30).value_counts().sort_index().to_dict())
wins = ep_df[ep_df.pnl > 0]; losses = ep_df[ep_df.pnl <= 0]
P(f"win rate {len(wins)/len(ep_df)*100:.1f}%   avg win ${wins.pnl.mean():,.2f} ({wins.ret.mean()*100:.2f}%)   "
  f"avg loss ${losses.pnl.mean():,.2f} ({losses.ret.mean()*100:.2f}%)   avg P&L/trade ${ep_df.pnl.mean():,.2f} ({ep_df.ret.mean()*100:.3f}%)   "
  f"median P&L ${ep_df.pnl.median():,.2f}   profit factor {wins.pnl.sum()/abs(losses.pnl.sum()):.2f}")
P(f"realized P&L closed episodes: ${ep_df.pnl.sum():,.2f}   unrealized on open episodes (marked to last close): ${open_df.unrealized_mtm.sum():,.2f}   "
  f"Alpaca unrealized_pl on open positions: ${positions.unrealized_pl.astype(float).sum():,.2f}")
P(f"avg notional per episode ${ep_df.notional.mean():,.0f};  total traded notional (all fills) ${(legs_df.qty*legs_df.price).sum():,.0f}")
P(); P("by side:"); P(ep_df.groupby("side").agg(n=("pnl", "size"), win_rate=("pnl", lambda x: (x > 0).mean()), pnl=("pnl", "sum"),
                                              avg_pnl=("pnl", "mean"), avg_ret_pct=("ret", lambda x: x.mean() * 100),
                                              hold=("hold_tdays", "mean")).round(3).to_string())
P(); P("by exit reason:"); P(ep_df.groupby("exit_reason").agg(n=("pnl", "size"), win_rate=("pnl", lambda x: (x > 0).mean()), pnl=("pnl", "sum"),
                                                     avg_pnl=("pnl", "mean"), avg_ret_pct=("ret", lambda x: x.mean() * 100),
                                                     hold=("hold_tdays", "mean")).round(3).to_string())
P(); P("by period (episode exit date):"); P(ep_df.groupby("period").agg(n=("pnl", "size"), win_rate=("pnl", lambda x: (x > 0).mean()), pnl=("pnl", "sum"),
                                                              avg_pnl=("pnl", "mean"), hold=("hold_tdays", "mean"),
                                                              stops=("exit_reason", lambda x: (x == "NATIVE_STOP").sum())).round(3).to_string())
P(); P("by side x period (sum pnl / count):"); P(ep_df.pivot_table(index="period", columns="side", values="pnl", aggfunc=["sum", "size"]).round(0).to_string())
P(); P("top 10 / bottom 10 symbols by realized P&L:")
sp = ep_df.groupby("symbol").pnl.agg(["sum", "size"]).sort_values("sum")
P(pd.concat([sp.head(10), sp.tail(10)]).round(0).to_string())

ep_sorted = ep_df.sort_values("entry_t")
reentry = []
for sym, g in ep_sorted.groupby("symbol"):
    g = g.reset_index()
    for i in range(len(g) - 1):
        gap = tdi.get(g.loc[i + 1, "entry_date"], np.nan) - tdi.get(g.loc[i, "exit_date"], np.nan)
        reentry.append(dict(symbol=sym, exit_reason=g.loc[i, "exit_reason"], gap_tdays=gap, same_side=g.loc[i, "side"] == g.loc[i + 1, "side"],
                            next_pnl=g.loc[i + 1, "pnl"]))
re_df = pd.DataFrame(reentry)
P(); P("re-entry after exit (same symbol): share re-entered within 1 / 5 trading days, by exit reason:")
P(re_df.groupby("exit_reason").agg(n=("gap_tdays", "size"), within1=("gap_tdays", lambda x: (x <= 1).mean()),
                                   within5=("gap_tdays", lambda x: (x <= 5).mean()), same_side=("same_side", "mean"),
                                   next_trade_pnl=("next_pnl", "sum")).round(3).to_string())

# ---------------------------------------------------------------- execution cost vs opening print
legs_df["open_px"] = [opn.get((s, d), np.nan) for s, d in zip(legs_df.symbol, legs_df.date)]
legs_df["hhmm"] = legs_df.t.dt.strftime("%H:%M")
is_open_mkt = (legs_df.otype == "MARKET") & (legs_df.hhmm >= "09:30") & (legs_df.hhmm <= "09:40") & legs_df.open_px.notna()
bench = np.where(is_open_mkt, legs_df.open_px, np.where((legs_df.otype == "STOP") & legs_df.stop_price.notna(), legs_df.stop_price, legs_df.price))
legs_df["bench"] = bench
legs_df["cost_usd"] = legs_df.sgn * (legs_df.price - legs_df.bench) * legs_df.qty
legs_df["cost_bps"] = legs_df.sgn * (legs_df.price - legs_df.bench) / legs_df.bench * 1e4
legs_df["bucket"] = np.where(is_open_mkt, "market@open", np.where(legs_df.otype == "STOP", "stop", "market intraday/other"))
legs_df["notional"] = legs_df.qty * legs_df.price
P(); P("=" * 100); P("EXECUTION COST (fill vs benchmark: opening print for pre-open market orders, stop price for stop fills)"); P("=" * 100)
P(legs_df.groupby("bucket").agg(fills=("qty", "size"), notional=("notional", "sum"), cost_usd=("cost_usd", "sum"),
                                 mean_bps=("cost_bps", "mean"), median_bps=("cost_bps", "median")).round(2).to_string())
P(legs_df[is_open_mkt].groupby(["kind", "side"]).agg(fills=("qty", "size"), mean_bps=("cost_bps", "mean"), median_bps=("cost_bps", "median"),
                                                     cost_usd=("cost_usd", "sum")).round(2).to_string())
tot_cost = legs_df.cost_usd.sum()
P(f"TOTAL execution cost vs benchmarks: ${tot_cost:,.2f}  ({tot_cost / legs_df.notional.sum() * 1e4:.1f} bps of traded notional)")
legs_df["period"] = legs_df.date.map(period_of)
P("execution cost and turnover by period:")
P(legs_df.groupby("period").agg(fills=("qty", "size"), notional=("notional", "sum"), cost_usd=("cost_usd", "sum")).round(0).to_string())

cost_by_ep = defaultdict(float)
ep_lookup = defaultdict(list)
for i, r in ep_df.iterrows(): ep_lookup[r.symbol].append((r.entry_date, r.exit_date, i))
for r in legs_df.itertuples():
    for a, b, i in ep_lookup.get(r.symbol, []):
        if a <= r.date <= b:
            cost_by_ep[i] += r.cost_usd; break
ep_df["exec_cost"] = ep_df.index.map(lambda i: cost_by_ep.get(i, 0.0))
ep_df["pnl_at_bench"] = ep_df.pnl + ep_df.exec_cost
P(); P(f"closed-episode P&L decomposition: realized ${ep_df.pnl.sum():,.0f} = price movement at benchmarks ${ep_df.pnl_at_bench.sum():,.0f} "
       f"minus execution cost ${ep_df.exec_cost.sum():,.0f}")
P(ep_df.groupby("period").agg(realized=("pnl", "sum"), price_move=("pnl_at_bench", "sum"), exec_cost=("exec_cost", "sum")).round(0).to_string())
ep_df.to_csv(O / "episodes.csv", index=False); legs_df.to_csv(O / "fill_legs.csv", index=False)

# ---------------------------------------------------------------- daily exposure from the ledger
day_fills = fills.copy(); day_fills["q"] = day_fills.qty * day_fills.sgn
qty_by_day = day_fills.groupby(["date", "symbol"]).q.sum().unstack(fill_value=0.0)
all_days = sorted(set(qty_by_day.index) | set(d for d in trading_days if d >= qty_by_day.index.min()))
qty_by_day = qty_by_day.reindex(all_days).fillna(0.0).cumsum()
expo = []
for d, row in qty_by_day.iterrows():
    lmv = smv = 0.0; nl = ns = 0
    for s, q in row[row.abs() > 1e-9].items():
        c = cls.get((s, d), np.nan)
        if np.isnan(c): continue
        if q > 0: lmv += q * c; nl += 1
        else: smv += q * c; ns += 1
    expo.append(dict(date=d, long_mv=lmv, short_mv=smv, n_long=nl, n_short=ns))
ex = pd.DataFrame(expo).merge(ph[["date", "equity"]], on="date", how="left")
ex["gross"] = (ex.long_mv - ex.short_mv) / ex.equity; ex["net"] = (ex.long_mv + ex.short_mv) / ex.equity
ex.to_csv(O / "exposure_daily.csv", index=False)
P(); P("=" * 100); P("EXPOSURE (reconstructed from fills, marked at daily close)"); P("=" * 100)
ex["period"] = ex.date.map(period_of)
P(ex.groupby("period").agg(days=("date", "size"), n_long=("n_long", "mean"), n_short=("n_short", "mean"), long_mv=("long_mv", "mean"),
                           short_mv=("short_mv", "mean"), gross_x=("gross", "mean"), net_x=("net", "mean"), net_min=("net", "min"), net_max=("net", "max")).round(2).to_string())
P("sample dates:")
sd = [pd.Timestamp(x).date() for x in ["2026-06-01", "2026-07-08", "2026-07-09", "2026-07-15", "2026-08-12", "2026-09-04", "2026-09-08", "2026-09-15", "2026-10-02", "2026-10-12", "2026-11-17"]]
P(ex[ex.date.isin(sd)][["date", "n_long", "n_short", "long_mv", "short_mv", "gross", "net", "equity"]].round(2).to_string(index=False))

# ---------------------------------------------------------------- equity curve by period + benchmarks
P(); P("=" * 100); P("EQUITY CURVE vs BENCHMARKS"); P("=" * 100)
eq = ph.set_index("date").equity
spy = bars[bars.symbol == "SPY"].set_index("date").close
univ = sorted(set(json.load(open(ROOT / "watchlist.json"))) & set(bars.symbol))
piv = bars[bars.symbol.isin(univ)].pivot(index="date", columns="symbol", values="close")
def window_stats(a, b):
    a, b = pd.Timestamp(a).date(), pd.Timestamp(b).date()
    e = eq[(eq.index >= a) & (eq.index <= b)]
    if len(e) < 2: return None
    r = e.pct_change().dropna()
    dd = (e / e.cummax() - 1).min()
    s = spy[(spy.index >= a) & (spy.index <= b)]
    pv = piv[(piv.index >= a) & (piv.index <= b)].dropna(axis=1, how="any")
    ew_bh = (pv.iloc[-1] / pv.iloc[0] - 1).mean()
    return dict(start=str(e.index[0]), end=str(e.index[-1]), days=len(e) - 1, equity_start=e.iloc[0], equity_end=e.iloc[-1],
                ret_pct=(e.iloc[-1] / e.iloc[0] - 1) * 100, daily_mean_pct=r.mean() * 100, daily_std_pct=r.std() * 100,
                sharpe_ann=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan, max_dd_pct=dd * 100,
                spy_ret_pct=(s.iloc[-1] / s.iloc[0] - 1) * 100, ew_universe_bh_pct=ew_bh * 100, n_univ=pv.shape[1])
tbl = []
for name, a, b in PERIODS + [("FULL May20-Nov17", "2026-05-19", "2026-11-18"), ("EVAL Aug11-Nov17 (restart)", "2026-08-11", "2026-11-18"),
                             ("Jul09-Nov17 (full book)", "2026-07-08", "2026-11-18")]:
    w = window_stats(a, b)
    if w: w["period"] = name; tbl.append(w)
wt = pd.DataFrame(tbl).set_index("period")
P(wt.round(2).to_string())
wt.to_csv(O / "period_stats.csv")
r_eq = eq.pct_change().dropna(); r_spy = spy.pct_change().dropna()
j = pd.concat([r_eq, r_spy], axis=1, keys=["bot", "spy"]).dropna()
j = j[j.index >= pd.Timestamp("2026-07-09").date()]
beta = np.cov(j.bot, j.spy)[0, 1] / j.spy.var()
P(f"\ndaily return corr with SPY (Jul 9 ->): {j.bot.corr(j.spy):.2f}, beta {beta:.2f}  (a market-neutral book should be near 0)")
P(f"share of down days (Jul 9 ->): {(j.bot < 0).mean()*100:.0f}%   mean daily P&L ${(j.bot * eq.reindex(j.index).shift(1)).mean():,.0f}")
P("worst 10 days (%):"); P((r_eq.sort_values().head(10) * 100).round(2).to_string())
P("best 5 days (%):"); P((r_eq.sort_values().tail(5) * 100).round(2).to_string())

# ---------------------------------------------------------------- predictions vs realized
P(); P("=" * 100); P("PREDICTIONS vs WHAT HAPPENED NEXT (signals_history.jsonl)"); P("=" * 100)
sig = []
for line in open(ROOT / "data/cache/signals_history.jsonl"):
    d = json.loads(line); ts = pd.Timestamp(d["generated_at"]).tz_localize("UTC").tz_convert(ET)
    for s in d["signals"]: sig.append(dict(run=ts, date=ts.date(), **s))
sig = pd.DataFrame(sig)
sig = sig.sort_values("run").groupby(["date", "symbol"]).last().reset_index()
sig = sig[sig.date.isin(set(trading_days))]
op = bars.pivot(index="date", columns="symbol", values="open"); cl_ = bars.pivot(index="date", columns="symbol", values="close")
dates = list(op.index); di = {d: i for i, d in enumerate(dates)}
def fwd(sym, d, h, kind):
    i = di.get(d)
    if i is None or sym not in op.columns: return np.nan
    try:
        if kind == "oo":
            a = op.iat[i, op.columns.get_loc(sym)]; b = op.iat[i + h, op.columns.get_loc(sym)]
        else:
            a = cl_.iat[i - 1, cl_.columns.get_loc(sym)]; b = cl_.iat[i + h - 1, cl_.columns.get_loc(sym)]
        return b / a - 1
    except IndexError: return np.nan
for h in (1, 5, 20):
    for kind in ("oo", "cc"):
        sig[f"r_{kind}_{h}"] = [fwd(s, d, h, kind) for s, d in zip(sig.symbol, sig.date)]
def ic_table(df, col="xgb_pred"):
    out = []
    for h in (1, 5, 20):
        for kind in ("oo", "cc"):
            c = f"r_{kind}_{h}"
            ics = df.dropna(subset=[col, c]).groupby("date").apply(lambda g: spearmanr(g[col], g[c]).statistic if len(g) >= 20 else np.nan).dropna()
            if len(ics) == 0: continue
            out.append(dict(horizon_days=h, returns=("open->open" if kind == "oo" else "prior close->close"), n_dates=len(ics), mean_IC=ics.mean(),
                            IC_tstat=ics.mean() / ics.std() * np.sqrt(len(ics)) if ics.std() > 0 else np.nan, hit_rate=(ics > 0).mean()))
    return pd.DataFrame(out)
big = sig[sig.date >= pd.Timestamp("2026-07-09").date()]
complete20 = [d for d in big.date.unique() if di.get(d, 10**9) + 19 < len(dates)]
P(f"signal days with the full universe (Jul 9 ->): {big.date.nunique()}; 20-day forward windows complete for runs up to {max(complete20)}")
P("cross-sectional Spearman IC of xgb_pred vs realized forward return (20d windows overlap heavily, so n_dates overstates independence):")
P(ic_table(big).round(4).to_string(index=False))
def basket(df, h, kind):
    c = f"r_{kind}_{h}"
    g = df.dropna(subset=[c]).groupby(["date", "signal"])[c].mean().unstack()
    g["long_short"] = g["BUY"] - g["SHORT"]
    return g
P("\nmean forward return (%) of the BUY basket, SHORT basket, the HOLD rest, and long-short, equal-weight:")
bt = []
for h in (1, 5, 20):
    for kind in ("oo", "cc"):
        g = basket(big, h, kind)
        bt.append(dict(horizon=h, returns=("open->open" if kind == "oo" else "prior close->close"), n_days=len(g), BUY=g["BUY"].mean() * 100, SHORT=g["SHORT"].mean() * 100,
                       HOLD=g["HOLD"].mean() * 100 if "HOLD" in g else np.nan, long_short=g.long_short.mean() * 100,
                       ls_tstat=g.long_short.mean() / g.long_short.std() * np.sqrt(len(g)) if g.long_short.std() > 0 else np.nan,
                       ls_hit=(g.long_short > 0).mean()))
P(pd.DataFrame(bt).round(3).to_string(index=False))
P("\nIC (open->open) by period:")
for name, a, b in PERIODS[1:]:
    sub = big[(big.date >= pd.Timestamp(a).date()) & (big.date <= pd.Timestamp(b).date())]
    t = ic_table(sub); t = t[t.returns == "open->open"]
    P(f"  {name}: " + "; ".join(f"h={r.horizon_days}: IC {r.mean_IC:+.4f} (n={r.n_dates}, hit {r.hit_rate:.0%})" for r in t.itertuples()))
# day-to-day stability of the picks
P("\nsignal stability: share of today's BUY (SHORT) names that were also BUY (SHORT) yesterday:")
days = sorted(big.date.unique()); stab = []
for a, b in zip(days[:-1], days[1:]):
    sa = big[big.date == a].set_index("symbol").signal; sb = big[big.date == b].set_index("symbol").signal
    for sgn in ("BUY", "SHORT"):
        today = set(sb[sb == sgn].index); yest = set(sa[sa == sgn].index)
        if today: stab.append(dict(date=b, signal=sgn, n=len(today), kept=len(today & yest) / len(today)))
st = pd.DataFrame(stab)
P(st.groupby("signal").agg(days=("kept", "size"), mean_kept=("kept", "mean"), median_kept=("kept", "median")).round(3).to_string())
P("\nSIGN CHECK: for each opening market fill on day d, does the symbol's logged signal that morning match the side?")
sigmap = sig.set_index(["date", "symbol"]).signal.to_dict()
ol = legs_df[(legs_df.kind == "open") & (legs_df.otype == "MARKET")].copy()
ol["signal"] = [sigmap.get((d, s)) for d, s in zip(ol.date, ol.symbol)]
ol["expected"] = np.where(ol.side == "buy", "BUY", "SHORT")
m = ol.dropna(subset=["signal"])
P(f"  opening market fills with a same-day signal: {len(m)}; side matches signal: {(m.signal == m.expected).mean()*100:.1f}%; "
  f"mismatches: {(m.signal != m.expected).sum()} -> {m[m.signal != m.expected].groupby(['signal','expected']).size().to_dict()}")
P(f"  opening fills on days with no logged signal for that symbol: {ol.signal.isna().sum()} (manual trades / reruns)")

# ---------------------------------------------------------------- mass-liquidation days
P(); P("=" * 100); P("WHOLE-BOOK LIQUIDATION DAYS (>= 20 closing fills' worth of episodes in one day) and what they cost"); P("=" * 100)
closes_per_day = ep_df.groupby("exit_date").agg(n=("pnl", "size"), realized=("pnl", "sum"))
liq = closes_per_day[closes_per_day.n >= 20]
P(liq.round(0).to_string())
for d in liq.index:
    closed = ep_df[ep_df.exit_date == d]
    nxt = ep_df[(ep_df.entry_date > d) & (ep_df.symbol.isin(closed.symbol))].sort_values("entry_date").groupby("symbol").first()
    nxt = nxt[[tdi.get(e, 10**9) - tdi[d] <= 5 for e in nxt.entry_date]]
    same = sum(1 for s, r in nxt.iterrows() if r.side == closed.set_index("symbol").loc[s, "side"])
    cost_d = legs_df[(legs_df.date == d) & (legs_df.kind == "close")].cost_usd.sum()
    P(f"  {d}: {len(closed)} episodes closed (realized ${closed.pnl.sum():,.0f}, exit exec cost ${cost_d:,.0f}); "
      f"{len(nxt)} of those names re-opened within 5 trading days, {same} on the same side")

(O / "summary_03.txt").write_text(buf.getvalue(), encoding="utf-8")
print("\nwritten:", O / "summary_03.txt")
