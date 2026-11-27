# Templar post-mortem

Written 2026-11-18 from the code at commit `daace9c` (plus the uncommitted Nov 15 model files), the git history, 282 log files, the logged prediction history, and a read-only pull of the Alpaca paper account (4,262 orders, 7,215 fills, daily equity). Nothing was changed, no orders were touched. Analysis scripts and their raw outputs are in `postmortem/` (see the appendix).

Throughout, **Verified** means the number comes straight from the account data, logs, or code. **Inferred** means it is my reading of what those numbers mean.

---

## 1. Summary: why the account went from $100k to $77k

The account lost $24,023 (−24.0%) between 2026-05-20 and the 2026-11-17 close. Over the same dates SPY was down 0.5% and an equal-weight buy-and-hold of the bot's own 147-stock universe was down 0.2%. The bot's daily returns had a 0.05 correlation with SPY, so this was not the market. It was the strategy.

Ranked causes, with confidence:

| # | Cause | Share of loss | Confidence |
|---|---|---|---|
| 1 | **Turnover far too high for any edge the model could have had.** 14 round trips per day, median hold 2 trading days, $9.1M traded on a ~$88k account (104x). Execution cost alone was $12,232, about half the loss, running at 28% of equity per year. | ~51% | High (verified from fills) |
| 2 | **The model's predictions had no useful information at the horizon actually traded.** It was trained to rank 20-day returns, but positions were re-ranked and replaced daily. At 1 to 5 days the live predictions were slightly *anti*-predictive (5-day rank correlation −0.03, long-short basket −0.46% per 5 days). | most of the other ~49% | High |
| 3 | **A 7% stop-loss / 20% take-profit overlay on a daily-rebalanced book of high-volatility names.** 299 stop exits realized −$64,286; 205 of them were re-entered on the same side within one trading day, at worse prices on net, and 47% of those were stopped again. The stops didn't change what the book held; they just converted drawdowns into realized losses plus two more spreads. | overlaps with 1 and 2 | High |
| 4 | **The production model was effectively frozen on data ending April 2025.** The split code trains on the first 70% of each stock's history, so every "weekly retrain" added about 3.5 trading days of 2025 data. The model that traded Jul–Nov 2026 never saw the prior 18 months. | hard to quantify | High that it's true; medium that it mattered much, since the signal was weak anyway |
| 5 | **No evaluation loop that connected model quality to money.** There was never a backtest. The only metric was a rank correlation on one fixed, overlapping 9-month window that overstated significance by roughly 4x. Changes were judged on that number or on a few weeks of paper equity. | enabling cause | High |
| 6 | **Execution bugs in May–Aug** (stops expiring at the close of entry day, inverted short take-profit, 2.8x gross leverage on the first full day, features 20 days stale until Jul 9, sizing against stale prices). | ~$7k of adverse price movement in Jul–Aug | High that they existed; medium on the dollar attribution |
| 7 | **Silent data bug: the sentiment features were zero in every row, in training and live, since day one.** The bars index carries 04:00/05:00 timestamps and the sentiment index is midnight, so the join never matched. The whole FinBERT/Benzinga pipeline contributed nothing. | $0 directly; shows the absence of data checks | High (verified: 0 of 180,733 processed rows non-zero) |

**Did the model ever have predictive power?** On the 20-day horizon it was trained for, the live predictions showed a weak positive rank correlation in Jul–Sep (+0.05 to +0.11, similar to the +0.05 to +0.07 reported on the test split) that flipped to −0.10 in October. That is a handful of overlapping monthly samples and is not statistical evidence of an edge. At the 1–5 day horizon the strategy actually traded, it was slightly negative. Whatever faint 20-day signal existed could not have paid for 27 bps per round trip at a 2-day hold. The honest read: **no demonstrated edge, and the trading machinery guaranteed a loss even if a small one existed.**

**Did the retraining breakage cause the decline?** No. The job did not run to completion on 5 of 6 Sundays since Oct 4 because Jordan closed its window (confirmed; see section 5), but the weekly retrain never materially changed the model anyway (see cause 4), and XGBoost was in fact refreshed on Oct 12 and Nov 15 by partial runs. The October–November acceleration (−13.8% in 31 days) came from the long book (−$13,272 realized) during a market sell-off, with the model's 20-day correlation going negative. Different cause, same strategy.

---

## 2. How the system works

Plain-language description of what the code at `daace9c` does. All verified by reading the code; where live behavior was checked against logs or data, I say so.

**Data.** Daily OHLCV bars from Alpaca (SIP feed, 5 years, 150 S&P names chosen by average volume plus a few legacy names; `watchlist.json`). Benzinga headlines via the Alpaca News API scored by FinBERT and aggregated to a daily mean and count per symbol. A StockTwits filter exists but the API has been returning 403 since June, so it does nothing.

**Features (28).** Raw price and volume levels (open, high, low, close, volume, log volume), 1/5/20-day returns, EMA 9/21/50, MACD (3), RSI, stochastic (2), Bollinger bands (lower/mid/upper/width/%B), ATR, OBV, a "VWAP_D" that on daily bars just equals the day's typical price, and the two sentiment columns. Everything is computed from the bar's own and prior days. The raw level features (close, EMAs, BB levels, OBV, ATR in dollars) are not normalized per stock, so a $700 stock and a $15 stock look entirely different to the model regardless of what they are doing.

**Target.** The 20-trading-day forward close-to-close return, z-scored across all symbols on each date. So the model is asked: "relative to the other 148 names, how will this one do over the next 4 weeks?"

**Models.** XGBoost regressor (Optuna-tuned: depth 2, 483 trees, learning rate 0.0075, min child weight 77, so a very shallow, nearly additive model) and a Temporal Fusion Transformer. The ensemble weight on the TFT is 0.0 and has been since May 24 because its test correlation was always zero or negative. In practice **the strategy is XGBoost only**; the TFT is still trained for hours every week and run for every symbol every morning (which also created 15,889 `lightning_logs/version_*` directories).

**From predictions to orders (daily, 7:15 AM MT).** Fetch yesterday's bar, build features, predict one number per symbol from the latest row. Min-max scale the 149 predictions to [0, 1]. Names with score ≥ 0.6 are BUY candidates; the top 29 (20% of the universe) become BUY; the 29 lowest-scored remaining names become SHORT. Then a full daily rebalance: close any open position whose name is not in today's BUY (if long) or SHORT (if short) set; close anything up 20% (take-profit) or down 7% (a backstop behind the native stop); open new positions for the rest. Orders are GTC market orders queued ~12 minutes before the open with an attached GTC stop 7% away. Verified in logs: orders go out at ~9:21 ET; 85% of fills land between 9:30 and 9:35 ET.

**Sizing.** Each position gets min(5% of equity, 1.8 × equity / number of targets) → with 58 targets, about 3.1% of equity (~$2,900). Positions are sized once at entry and never resized. Target gross exposure 1.8x; verified average gross was 1.5x in Aug–Nov, 2.0x in Jul–Aug (2.8x on Jul 9).

**Risk halt.** `drawdown_breached` compares equity to `last_equity` (yesterday's close), so the "15% max drawdown" rule can only fire on a 15% *single-day* loss. It never fired. Verified: zero "Drawdown limit hit" lines across 149 daily logs while the account fell 26% peak-to-trough.

**Training and evaluation.** Per symbol, the first 70% of rows (minus a 20-row purge) are train, the next 15% val, the last 15% test. Using the project's own split code on the processed data from the Nov 15 run (`postmortem/07_split_boundaries.py`):

| | Train | Validation | Test |
|---|---|---|---|
| Dates (full-history symbols) | 2022-01-28 → **2025-04-17** | 2025-05-19 → 2026-01-02 | 2026-02-03 → 2026-10-16 |
| Rows per symbol | 808 | 158 | 178 |

Optuna maximizes mean per-date Spearman correlation on validation (50 trials, first Sunday of the month). After training, the code logs the per-date Spearman correlation ("IC") on the test split. That is the only evaluation that exists. There is no backtest, no simulated P&L, no turnover or cost estimate, and `tests/` is empty. Note the test window covers the same months the bot was live.

**What changed over time, and how it was measured** (from 69 commits, May 19 → Sep 27):

- May 19 (one day, 24 commits): scaffold, then horizon 5→20 days ("val_loss plateaus"), z-scored target ("eliminates the 48/50 SELL problem"), TFT capacity up then dropout up, ensemble normalization fixed twice, shorting added, stops and take-profit added, universe set to 50 names. Each justified in the commit message by a loss value or an observation about that day's signals.
- May 24: TFT weight set to 0 on a test IC of −0.12 vs XGBoost +0.12; IC evaluation added.
- Jun 3: Optuna search space tightened "to reduce overfitting".
- Jul 8: three training fixes (softplus target bug, purged splits, IC-based tuning), universe 50→150, sentiment pipeline rebuilt; commit says sentiment "improved out-of-sample IC". Given the features were identically zero, that improvement came from something else (most likely the universe change).
- Jul 9: live predictions had been made from features 20 days stale since inception; fixed.
- Aug 11: stop-loss expiry, inverted short take-profit, 2.9x leverage, stale-stop rejections, cancel/close race fixed. Evaluation clock reset to "day one".
- Aug 19: sizing against stale prices fixed after MRNA gapped +148%.
- Sep 27: stop-coverage check added.

Every model-affecting change was measured against either val_loss, the test IC on the same fixed window, or how paper equity looked over the following days. None was measured against a simulated P&L with costs, and no two model variants were ever compared on the same metric side by side. Weekly retrains logged test IC of 0.095, 0.069, 0.062, 0.052, 0.078, 0.053, 0.057, 0.048, 0.070, 0.072, 0.072, 0.068 (Jul 8 → Sep 27). Training set size over that span grew from 119,403 to 119,608 rows (+0.17%).

---

## 3. What the trade data shows

All numbers verified from the Alpaca pull unless noted. Scripts: `03_trade_analysis.py`, `06_robustness.py`; raw output in `postmortem/out/summary_03.txt`.

### Equity

| Period | Dates | Days | Bot | SPY | EW universe B&H | Notes |
|---|---|---|---|---|---|---|
| P1 | May 20 – Jul 8 | 32 | −0.13% | +0.56% | +1.25% | 50-name universe, 2 longs/2 shorts; **no trades Jun 16 – Jul 7** (task fired before Alpaca's clock opened, market-closed guard skipped every day) |
| P2 | Jul 9 – Aug 11 | 23 | −3.65% | +2.51% | +2.43% | 150 names, ~27/side; execution bugs live |
| P3 | Aug 12 – Oct 2 | 36 | −8.22% | −0.37% | −1.64% | "clean" evaluation period, retrain job working |
| P4 | Oct 5 – Nov 17 | 31 | −13.75% | −5.76% | −3.84% | retrain job failing most Sundays |
| Full | May 20 – Nov 17 | 126 | **−24.02%** | −0.48% | −0.23% | max drawdown −26.1%, annualized Sharpe −3.0 |
| Since restart | Aug 11 – Nov 17 | 69 | −21.26% | −5.24% | −5.62% | |

Equity milestones: $100,000 (May 20) → $101,281 peak (May 28) → $99,951 (Jul 8) → $96,493 (Aug 11) → $88,364 (Oct 2) → $74,814 low (Nov 12) → $75,977 (Nov 17 close) → $77,295 intraday Nov 18.

Since Jul 9: 61% of days were down, mean daily P&L −$263, daily correlation with SPY 0.05, beta 0.07. Worst day Oct 22 (−4.76%); best day Aug 19 (+4.68%, mostly the MRNA gap).

### Trades

| Metric | Value |
|---|---|
| Fills / filled orders / all orders | 7,215 / 2,928 / 4,262 (2,713 market, 1,549 stop; 1,175 canceled, 103 expired) |
| Closed round trips (flat → open → flat) | 1,436, plus 56 open now |
| Round trips per trading day (Jul 9 →) | **14.4** (1,349 over 94 days) |
| Holding time, trading days | mean 3.1, **median 2**, p25 1, p75 4; 545 trades held exactly 1 day, 139 same-day |
| Win rate | **40.0%** |
| Average win / average loss | +$131.64 (+4.17%) / −$118.69 (−3.88%) |
| Average P&L per trade | **−$18.45 (−0.66%)**; median −$19.58; profit factor 0.74 |
| Average position | $3,131 |
| Total traded notional | $9.12M = 104x average equity; $88.9k/day since Jul 9 |
| Realized P&L (closed) | −$26,497 |
| Unrealized (56 open positions, Alpaca) | +$3,903 |
| Fees / dividends | −$56.70 / −$54.85 |
| Reconciliation | −$22,705 vs equity change −$24,023 to Nov 17 close; the gap is Nov 18's intraday move (+$1,318) |

By side: long 677 trades, 40.5% win, −$14,523; short 759 trades, 39.7% win, −$11,974.

By period (trade exit date):

| Period | Trades | Win rate | Realized | of which long / short | Stop exits |
|---|---|---|---|---|---|
| P1 | 87 | 52.9% | −$47 | +$1,344 / −$1,391 | 1 |
| P2 | 363 | 38.3% | −$9,997 | −$6,379 / −$3,617 | 60 |
| P3 | 485 | 43.3% | −$2,039 | +$3,785 / −$5,824 | 65 |
| P4 | 501 | 35.9% | −$14,414 | **−$13,272** / −$1,142 | 96 |

### How positions ended

| Exit reason | Trades | Win rate | Realized | Avg | Avg hold (days) |
|---|---|---|---|---|---|
| Dropped from today's top/bottom 29 | 1,037 (72%) | 48.8% | +$12,032 | +$12 | 2.9 |
| Native 7% stop filled | 222 | 0% | **−$41,795** | −$188 (−6.3%) | 3.0 |
| Daily backstop (already past −7%) | 77 | 0% | **−$22,492** | −$292 (−9.7%) | 3.9 |
| Take-profit (+20%) | 33 | 100% | +$24,145 | +$732 | 10.5 |
| Manual (Sep 22–27 trims) | 67 | 53.7% | +$1,613 | | 1.4 |

The signal-driven exits roughly broke even. The stop/take-profit overlay lost $40k net. And it did not change what the book held:

- After a stop exit, the same name was re-entered **on the same side within one trading day in 205 of 299 cases** (97% within 5 days). Being flat between the stop and the re-entry lost a further $4,712 net (re-entered at worse prices on average). 47% of those re-entries were stopped out again. Most-stopped names: COHR 16, INTC 15, SMCI 14, GLW 10, DDOG 10, AMD 10.
- After a take-profit exit, 85% were re-bought within one day (97% same side).

### Spread and slippage versus price movement

Benchmark: the official opening print for the market orders queued pre-open (6,240 fills), the stop price for stop fills (505), and no cost assigned to the 470 intraday manual fills.

| Bucket | Fills | Notional | Cost | Mean | Median |
|---|---|---|---|---|---|
| Market orders at the open | 6,240 | $7.58M | $10,825 | **13.3 bps** | 12.9 bps |
| Stop fills | 505 | $0.67M | $1,407 | 24.3 bps through the stop | 4.1 bps |
| Total | | $9.12M | **$12,232** | 13.4 bps per side | |

The cost is symmetric: buys 12.7–15.0 bps, sells 12.8–13.1 bps, opens and closes alike. So a round trip cost about 27 bps on a trade held a median of 2 days. Annualized, execution cost ran at **$24,700 per year, 28% of average equity**.

Decomposition of the loss (to now): execution cost −$12,232 (≈51%); adverse price movement net of costs −$10,400 (closed trades −$14,394, open positions +$4,000); fees and dividends −$112. By period, price movement was −$6,898 in P2 (bugs: 2.8x leverage on day one, no overnight stops, inverted short take-profit), **+$2,674 in P3** (the clean period: the picks made money before costs and lost after), and −$10,724 in P4.

Caveat (inferred): Alpaca paper fills market orders at the far side of the quote in the first minutes of trading. 13 bps per side is plausible for that style on this universe, but a live account could see a different number. The direction of the conclusion does not depend on it.

### Long versus short exposure

Reconstructed from fills and marked at daily closes (`postmortem/out/exposure_daily.csv`):

| Period | Avg longs / shorts | Avg long MV | Avg short MV | Gross / equity | Net / equity (min, max) |
|---|---|---|---|---|---|
| P1 | 1.5 / 1.5 | $7.1k | −$7.2k | 0.14x | 0.00 (−0.02, +0.01) |
| P2 | 26.8 / 26.0 | $103k | −$93k | 2.01x | +0.11 (−0.15, +0.28) |
| P3 | 22.4 / 22.2 | $75k | −$65k | 1.49x | +0.10 (−0.02, **+0.43**) |
| P4 | 24.2 / 25.8 | $62k | −$63k | 1.53x | −0.01 (−0.26, +0.28) |

Jul 9 (first full day): 29 longs, 27 shorts, gross 2.80x. Book nearly empty on Sep 8 (5/5), Sep 15 (1/1), Oct 12 (7/8): see "whole-book flips" below. The book was roughly dollar-neutral on average but drifted to +43% net long in September because positions are never resized (per-name budget depends on that day's target count).

### Did the predictions relate to what happened next?

Predictions were logged (`data/cache/signals_history.jsonl`, 118 runs, 91 trading days with the full universe since Jul 9). Cross-sectional Spearman correlation between the morning's `xgb_pred` and the subsequent open-to-open return:

| Horizon | Days | Mean IC | Days with IC > 0 | BUY basket | SHORT basket | HOLD rest | Long-short |
|---|---|---|---|---|---|---|---|
| 1 day | 90 | −0.011 | 48% | −0.20% | −0.07% | +0.02% | −0.13% |
| 5 days | 87 | **−0.030** (t ≈ −2.5) | 41% | −0.35% | +0.12% | +0.21% | **−0.46%** |
| 20 days | 72 (overlapping) | +0.053 | 83% | +3.09% | −0.43% | +0.34% | +3.51% (ex-MRNA +1.33%; median-based +2.86%) |

By period (open-to-open): P2 20-day IC +0.108, P3 +0.067, **P4 −0.100** (12 days with complete windows). 20-day long-short by signal month: Jul +5.0%, Aug +3.4%, Sep +5.4%, Oct −1.0% (median −4.4%). Using prior-close-to-close returns (the label's definition) gives essentially the same numbers, so the overnight gap did not flatter the training label.

Reading (inferred): at the 1–5 day horizon the book actually traded, the picks did slightly worse than the names it ignored. At 20 days there was a positive spread for three overlapping months and a negative one in the fourth. That is roughly four independent observations and not evidence of an edge; it is consistent with the model's test IC of ~0.05–0.07 being a weak, unstable effect. Both the BUY and SHORT baskets were high-volatility names that had fallen a lot (BUY group 20-day return −6.0%, SHORT group −10.4%, HOLD −0.4%; ATR/price 6.0% and 5.0% vs 3.3%). The model mostly sorts by volatility and price level: SHAP shares on the latest run are Bollinger width 23%, ATR 23%, 20-day return 17%, OBV 10%; `xgb_pred` correlates +0.33 with Bollinger width, +0.30 with ATR%, +0.20 with the raw price level, and 0.00 with sentiment (constant).

Day-to-day stability of the picks: 79% of today's BUY names and 76% of SHORT names were also picked yesterday, i.e. 6–7 names per side are swapped every day. That is the turnover engine.

### Whole-book flips from the scoring rule

Because the BUY threshold (0.6) is applied to a min-max scaled score and the prediction distribution is heavily left-skewed (daily median ≈ 0.00, minimum −0.7 to −1.8, maximum usually 0.1–0.3), on normal days 145 of 149 names clear 0.6 and the top 29 are taken. When one name gets an unusually high prediction, everyone else compresses below 0.6. Verified from the signal log: on Sep 8 MRNA's prediction was 0.662 against a minimum of −0.846, only 5 names cleared 0.6, and **47 positions were closed; Sep 15: 52 closed, 49 of those names re-opened within 5 days (48 same side); Oct 12: 37 closed; Oct 15: 45 closed, 43 re-opened (40 same side).** Each flip cost roughly $300 to exit and $300 to re-enter. Small individually, but it shows the selection rule is a function of one outlier, not of the signal.

### Sign check

3,767 opening fills matched against that morning's logged signal: **98.1% agree** (buy fills on BUY names, short fills on SHORT names). The 72 mismatches are 66 names logged HOLD that were opened (consistent with same-day re-runs on Jul 9 and Aug 11) and 6 buy fills that created a long on a SHORT name (consistent with a stop-buy filling after the short was already closed, the cancel/close race fixed Aug 11). No systematic inversion.

### Did performance change when retraining broke?

The first failed Sunday run was Oct 4. P3 (Aug 12 – Oct 2) lost 8.2% with the job working; P4 (Oct 5 – Nov 17) lost 13.8% with it failing. But XGBoost was refreshed anyway on Oct 12 and Nov 15 (the runs were killed during the TFT phase, after XGBoost had been saved), and in any case each weekly retrain only added a few days of early-2025 data. The P4 loss came from the long side (−$13,272) during a −5.8% SPY stretch while the 20-day IC went negative. **Inferred: the retraining failure did not cause the decline; the strategy did.**

---

## 4. Findings on the usual failure causes

**Lookahead leakage — evidence against (no leak found), with one design caveat.**
Verified: the target is `close.shift(-20)/close − 1` (a true forward return), every indicator is trailing, live features come from the last completed bar (the Nov 18 run's feature date was Nov 17 for 147 of 149 names), and the time split is purged by 20 rows. The label starts at the prior close while the trade enters at the next open; the live IC computed both ways is the same (+0.053 vs +0.046 at 20 days), so that misalignment did not flatter the signal. Four symbols with short histories (PSKY, SNDK, KVUE, WBD) have split boundaries at different dates from everyone else, so their training rows overlap other names' test dates; with 148 of 152 aligned, this is negligible.

**Train/test split problems — evidence for, though not the random-split kind.**
Verified: the split is time-ordered. The problems are (a) the model is trained only on the first 70% of history, so the production model's data ends **2025-04-17** and "weekly retraining" is cosmetic (+205 rows over 18 weeks); (b) the test window is the same ~9 months every week, overlaps the live trading period, and was used to make decisions (TFT weight, "IC-tuned params", judging each retrain), so it stopped being a clean holdout; (c) "178 test dates" are 178 overlapping 20-day windows, about 9 independent observations, so the logged ICIR of 0.6 and hit rates of 75% overstate significance roughly 4x; (d) Optuna's 50 trials optimized the same kind of overlapping-window IC on validation. The IC evaluation also groups by `time_idx` rather than calendar date, which only matters for the four misaligned symbols.

**Overfitting (good backtest, poor live) — evidence against classic overfitting; the gap is elsewhere.**
There was never a backtest to be good. The test IC was 0.05–0.07; the live 20-day IC Jul–Sep was 0.07–0.11 and then −0.10 in October. The model did roughly what its test metric said it would do: rank 20-day returns very weakly. The strategy then traded 2-day holds with a 7% stop, where the same predictions were slightly negative and costs were 27 bps per round trip. The failure is between "IC" and "P&L", not between test and live.

**Sign or logic bugs between prediction and order side — no sign bug; several logic bugs.**
Verified sign check: 98.1% of opening fills match the signal. Verified logic bugs, fixed before Aug 19: stops attached to DAY-TIF parents expired at the close of entry day (May 20 – Aug 11); short take-profit inverted so losing shorts were closed as "take profit" and winning ones never were; per-position sizing ignored book size (2.8x gross Jul 9); live predictions from features 20 trading days stale (May 21 – Jul 9); sizing against the prior close on gap days. Still present at `daace9c`: the min-max threshold that empties the book on an outlier; the drawdown halt that compares to yesterday's equity and so never fires on a slow bleed; stops and take-profits whose exits are re-entered the next morning; the sentiment join that never matches; `combine()` drops any symbol whose TFT prediction is NaN even at weight 0 (no evidence it happened); the stop-coverage check runs right after submitting the day's orders, so it flags positions that are mid-close (473 ERROR lines of noise since Sep 28). Also: `.env.example` is byte-identical to `.env` and tracked in git, so the API keys are in the repository.

**Turnover too high for the size of the edge — strong evidence for.**
Verified: 14.4 round trips/day, median hold 2 days, the book fully turns over every ~3.7 trading days, $9.1M traded, 13.4 bps per side, $12,232 cost = 51% of the loss, 28% of equity per year. Average trade −0.66%, of which the execution component alone is −0.27%. A 20-day-horizon ranking model re-ranked daily is a turnover machine by construction.

**Target close to pure noise at the chosen horizon — evidence for.**
20-day cross-sectional z-scored return on large caps from technical features. Test IC 0.05–0.07 on ~9 independent windows; live 20-day IC positive for three overlapping months, negative in the fourth; 1–5 day IC negative. The model's output is explained mostly by volatility and price-level features. Whatever is there is tiny, unstable, and nowhere near 27 bps per round trip.

**No reliable evaluation loop — strong evidence for.**
No backtester, no tests, no experiment log. 24 commits on May 19 alone changed the horizon, the target definition, model size, dropout, the ensemble normalization, and the exit rules, each justified by a single number from that run. The universe tripled and the sentiment pipeline was swapped in July; execution rules changed in August and September. The evaluation clock was reset once (Aug 11). Decisions were made on the fixed-window test IC or on 1–7 weeks of paper equity, both of which are noise at this edge size. The dead sentiment features went unnoticed for six months because nothing ever checked that a feature varied.

---

## 5. Why the retraining job broke (diagnosis only)

Verified from `logs/train_*.log`, the matching `logs/daily_*.log` files, Task Scheduler state, and the Windows event logs.

**What "working" looked like.** Jul 12 → Sep 27: 12 Sunday runs completed in 65–110 minutes each (fetch ~3 min, XGBoost ~1 s, TFT 15–18 epochs, evaluation). The Aug 30 run logged 15 epochs and then no evaluation or completion line; cause unknown.

**The failures.**

| Date | What the log shows | Reading |
|---|---|---|
| May 31, Jun 7 | PowerShell header, then nothing | `$ErrorActionPreference = "Stop"` turned Python's stderr warnings into fatal errors (fixed Jul 8, commit `d8e6ba1`) |
| Oct 4, Oct 18, Oct 25, Nov 1 | PowerShell header at 22:00:01, then exactly one Python line ("Log file: …") written in UTF-16, then nothing. No traceback, no "ERROR: fetch failed" line from the wrapper. | The *PowerShell process itself* was terminated within ~30 s of starting (Python was still importing torch/transformers). A Python crash would have produced the wrapper's error line. |
| Nov 8 | No train log, no daily log | Task never launched (machine off, or missed; Task Scheduler history before Nov 11 has rolled over) |
| Oct 12 (manual, Mon 10:53) | fetch OK, XGBoost saved 10:56, TFT epoch 3 at 11:06, then nothing | Killed 13 minutes in |
| Nov 15 | fetch OK, XGBoost saved 22:03, TFT epoch 2 at 22:13, then nothing. Task Scheduler: ran 22:00:00 → 22:17:07, **exit code 3221225786 = 0xC000013A = STATUS_CONTROL_C_EXIT** | The console received Ctrl+C or its window was closed |

Supporting facts: the task runs with an interactive logon and a visible console window (deliberately, per `setup_scheduler.ps1`). Power config has the sleep timer at 0 (never sleeps on AC). The System event log shows no shutdown, logoff, or power event between 21:55 and 22:30 on Nov 15. The `NativeCommandError` blocks in the logs are PowerShell 5.1 echoing benign Python stderr warnings (HF Hub token, triton not found); they are not failures.

**Diagnosis (confirmed by Jordan after the first draft).** The job never failed on code. Since early October the retrain's visible console window was closed by hand shortly after it appeared on Sunday nights (and once 17 minutes in, and once during the manual Oct 12 run). Jordan's account: fixes were needed and interest had run out, so the window got closed. The log evidence (process killed within seconds, no Python error, exit code 0xC000013A on Nov 15) matches that exactly. So "the retraining job broke" is really "the retraining job was stopped", and nothing in the pipeline needed repair for it to run.

**Consequences of the partial runs.** XGBoost is saved before the TFT phase, so the Oct 12 and Nov 15 kills still replaced the production XGBoost model (file times 10-12 10:56 and 11-15 22:03). `tft.train()` deletes all old checkpoints at start, so the TFT on disk is a 2-epoch artifact (harmless at weight 0). The post-training evaluation never ran, so **the model that has been trading since Oct 13 has no logged test IC at all.** And as covered in section 4, even a successful retrain only moves the training window forward by ~3.5 days of 2025 data.

**Two unrelated scheduler facts worth knowing.** The daily fetch failed on Oct 8 and Nov 12 (DNS resolution for `data.alpaca.markets` failed at 7:15), so no trading those days. Both task triggers store a fixed `-07:00` offset; `schtasks` now reports the next daily run at 6:15 AM and the next weekly at 9:00 PM, one hour earlier than this week's actual 7:15 AM and 10:00 PM runs. If the daily run really fires at 6:15 MST (8:15 ET), the market-closed guard will skip order execution every day. I could not confirm which will happen.

---

## 6. Lessons for a next attempt

**Build the evaluation before the model.**
1. A backtester that replays the exact decision rule on historical features: next-open fills, a cost per side (use 13 bps as the measured paper number until you have a better one), the actual sizing, stops if any, and the daily re-selection. It must output P&L, turnover, cost, and holding period. Any strategy change that cannot be run through it does not get deployed.
2. Walk-forward evaluation that trains up to date T and tests on T+1…T+h with **non-overlapping** windows, reporting a t-statistic on the number of independent windows, not on the number of rows. Compare every model to trivial baselines on the same metric: random ranking, 20-day reversal, 20-day momentum, low volatility.
3. Train the production model on all data up to now. The 70/15/15 split is for choosing a method, not for the model you deploy.
4. Data assertions in the pipeline: every feature has non-zero variance; joins report their match rate; the last feature date equals the last bar date; the universe and the model's feature list match. The sentiment bug would have failed on the first run.
5. Log every prediction and the realized return at the model's horizon, and compute live IC weekly against the backtest's expectation. The signal log existed here; it was never read.

**Do differently.**
6. Match the holding period to the prediction horizon. Either hold 20 days (rebalance a 1/20th slice of the book each day) or train a model for the 1–5 day horizon you actually trade. Do not re-rank a 20-day forecast every morning.
7. Select by rank with hysteresis (enter at top 20%, exit at top 35%), never by a threshold on a min-max scaled score.
8. Drop the 7% stop and 20% take-profit on a cross-sectional book, or make exits part of the backtested rule. Here they cost $40k net and changed nothing about what was held.
9. Normalize level features per stock (returns, z-scores, ratios to price) so the model learns behavior, not "which stock is this".
10. Fix the drawdown halt to peak-to-trough, and make the task runners non-interactive so a stray click cannot kill a training run.
11. Keep secrets out of the repository (`.env.example` currently is `.env`).

**Evidence that a strategy has no edge, before tuning anything.**
- Cross-sectional IC on non-overlapping horizon windows with |t| < 2 against a random-ranking baseline.
- Long-short basket spread at the *traded* horizon smaller than the round-trip cost (here: −0.46% per 5 days against a 0.27% cost).
- IC that is positive at one horizon and negative at another with no story for why.
- Predictions explained mostly by volatility or price level rather than by anything directional.
- Picks that turn over faster than the horizon they were made for.
- A shuffled-label or shuffled-date control that produces a similar IC.
If any of these hold, tuning hyperparameters is tuning noise.

---

## 7. Open questions and what would settle them

1. ~~Who or what closed the retrain window?~~ Resolved: Jordan closed it by hand each time. The only remaining design point is that an interactive task with a visible window can be killed by a click, which argues for a non-interactive logon on any future scheduled job.
2. **Is the 20-day long-short spread (+3–5% in Jul–Sep, −1% in Oct) real?** Four overlapping months cannot say. Settle by: a walk-forward backtest with non-overlapping 20-day windows over the full 5 years, with 27 bps round-trip costs and a 20-day hold; compare to a simple volatility-sorted baseline since the model mostly sorts by volatility.
3. **How much of the 13 bps per side is a paper-trading artifact?** Settle by: capturing NBBO quotes at order time for a week (read-only), or a tiny live test with limit-on-open orders.
4. **Will the scheduled tasks fire at 6:15 AM / 9:00 PM after the DST change, as `schtasks` claims, or at 7:15 / 10:00 as they did this week?** Settle by: checking `logs/scheduler_2026-11-19.log` tomorrow.
5. **Why did the Aug 30 run end without evaluation lines?** Not in any log. Probably the same kind of termination; unverifiable.
6. **The 6 fills that opened a long on a SHORT-signalled name:** I attributed them to stop-buy fills racing a close; confirming needs matching each fill's `order_id` to the parent stop order and the close order timestamps (data is in `postmortem/data/`).
7. **Short borrow costs and dividend charges** on shorts were effectively zero in paper ($55 net dividends). Live costs on the hard-to-borrow names the model favors (SMCI, COHR, MRNA) would be higher; unknown by how much.

---

## Appendix: files

Scripts (all read-only; `postmortem/`):
- `00_pull_alpaca.py`, `00b_pull_orders_full.py`, `00c_pull_bars.py`: account, positions, equity history, all orders, all fills, non-trade activities, SIP daily bars for the traded names and SPY → `postmortem/data/`.
- `01_signal_history_overview.py`: per-run BUY/SHORT counts and prediction distribution → `out/signal_runs_summary.csv`.
- `02_data_coverage.py`: bar history per symbol → `out/data_coverage.csv`.
- `03_trade_analysis.py`: ledger, round trips, exit reasons, execution cost, exposure, equity vs benchmarks, prediction-vs-realized, sign check, liquidation days → `out/summary_03.txt`, `out/episodes.csv`, `out/fill_legs.csv`, `out/exposure_daily.csv`, `out/period_stats.csv`.
- `04_model_inspection.py`: SHAP shares and feature correlations of the live model.
- `05_sentiment_join_check.py`: proof the sentiment join never matches.
- `06_robustness.py`: reconciliation, stop-and-re-enter counterfactual, 20-day basket robustness, turnover.
- `07_split_boundaries.py`: calendar dates of train/val/test using the project's own split code.

Key source files referenced: `models/dataset.py` (target, split), `models/ensemble.py` (score, selection), `execution/orders.py` and `execution/risk.py` (rebalance, sizing, stops, drawdown), `data/features.py` and `data/bars.py` (feature build, the join), `scripts/weekly_train.ps1` and `scripts/setup_scheduler.ps1` (the job).
