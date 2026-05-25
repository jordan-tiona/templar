"""
Templar - entry point.

Commands:
  python main.py fetch              -- download/refresh bar data
  python main.py train              -- train TFT + XGBoost models
  python main.py run                -- generate signals and execute orders (paper)
  python main.py run --dry          -- generate signals, log orders but don't submit
  python main.py tune [--trials N]  -- Optuna hyperparameter search for XGBoost
  python main.py screen             -- screen S&P 500 by volume
"""

import argparse
import logging
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")
from datetime import date
from pathlib import Path

_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / f"daily_{date.today()}.log"

_file_handler = logging.FileHandler(_LOG_FILE)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s |%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s |%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), _file_handler],
)
log = logging.getLogger("templar")
log.info("Log file: %s", _LOG_FILE)


def _evaluate_models(xgb_model, tft_model, train_df, val_df, test_df, logger):
    import pandas as pd
    from scipy.stats import spearmanr
    from models.dataset import TARGET

    # --- XGBoost: rolling per-date Spearman IC across the full test period ---
    xgb_m, scaler = xgb_model.load()
    xgb_preds = xgb_model.predict(xgb_m, scaler, test_df)
    ev = test_df[["time_idx", TARGET]].copy()
    ev["pred"] = xgb_preds.values
    xgb_ic = (
        ev.groupby("time_idx")
        .apply(lambda g: spearmanr(g["pred"], g[TARGET]).statistic if len(g) >= 5 else float("nan"))
        .dropna()
    )
    logger.info(
        "XGBoost test IC | mean: %.4f  ICIR: %.2f  hit: %.0f%%  (%d dates)",
        xgb_ic.mean(),
        xgb_ic.mean() / xgb_ic.std() if xgb_ic.std() > 0 else float("nan"),
        (xgb_ic > 0).mean() * 100,
        len(xgb_ic),
    )

    # --- TFT: cross-sectional IC at the test boundary ---
    # Build a combined train+val context per symbol (time_idx is already continuous).
    # predict_cross_section uses predict=True mode: one prediction per symbol from the
    # most recent encoder window, which is the 20-day forward return starting at the
    # last val date — matching target_return at the first row of each symbol's test split.
    try:
        train_val_df = (
            pd.concat([train_df, val_df])
            .sort_values(["symbol", "time_idx"])
            .reset_index(drop=True)
        )
        tft = tft_model.load()
        tft_preds = tft_model.predict_cross_section(tft, train_val_df, train_df)

        first_test_actuals = (
            test_df.loc[test_df.groupby("symbol")["time_idx"].idxmin(), ["symbol", TARGET]]
            .set_index("symbol")[TARGET]
        )
        common = tft_preds.dropna().index.intersection(first_test_actuals.dropna().index)

        if len(common) >= 5:
            ic = spearmanr(tft_preds[common], first_test_actuals[common]).statistic
            spread = tft_preds[common].std()
            logger.info(
                "TFT spot IC (test boundary) | IC: %.4f  pred-spread: %.4f  (%d symbols)",
                ic, spread, len(common),
            )
            if spread < 0.01:
                logger.warning("TFT pred spread is near zero — model may not be differentiating symbols")
        else:
            logger.warning("TFT spot IC: too few common symbols (%d) to evaluate", len(common))
    except Exception as e:
        logger.warning("TFT evaluation failed: %s", e)


def cmd_fetch(args):
    from data import fetch_bars, fetch_all_sentiment
    from data.screener import get_watchlist
    watchlist = get_watchlist()
    log.info("Fetching bars for %d symbols...", len(watchlist))
    fetch_bars(watchlist, force_refresh=args.refresh)
    log.info("Fetching sentiment...")
    fetch_all_sentiment(watchlist)
    log.info("Done.")


def cmd_train(args):
    import pandas as pd
    from data import fetch_bars, fetch_all_sentiment, build_features
    from data.screener import get_watchlist
    from models import tft as tft_model, xgb as xgb_model
    from models.dataset import prepare_combined, train_val_test_split

    watchlist = get_watchlist()

    log.info("--- PHASE 1/3: Building features ---")
    bars = fetch_bars(watchlist)
    feature_dfs = {}
    for i, (sym, df) in enumerate(bars.items(), 1):
        log.info("  [%d/%d] %s |%d bars", i, len(bars), sym, len(df))
        sentiment = fetch_all_sentiment([sym])[sym]
        feature_dfs[sym] = build_features(sym, df, sentiment)

    combined = prepare_combined(feature_dfs)
    train_df, val_df, test_df = train_val_test_split(combined)
    log.info(
        "Dataset split |train: %d rows, val: %d rows, test: %d rows",
        len(train_df), len(val_df), len(test_df),
    )

    log.info("--- PHASE 2/3: Training XGBoost ---")
    xgb_model.train(train_df, val_df)
    log.info("XGBoost training complete")

    log.info("--- PHASE 3/3: Training TFT ---")
    log.info("This will take 20-40 minutes on CPU |progress shown per epoch below")
    tft_model.train(train_df, val_df)

    log.info("--- Training complete ---")
    log.info("--- Evaluating on held-out test split ---")
    _evaluate_models(xgb_model, tft_model, train_df, val_df, test_df, log)


def cmd_run(args):
    from signals import generate_signals
    from execution import execute_signals
    from data import fetch_bars
    from data.screener import get_watchlist

    watchlist = get_watchlist()

    log.info("Generating signals...")
    signals = generate_signals(watchlist)

    # Latest close prices for position sizing
    bars = fetch_bars(watchlist)
    prices = {sym: float(df["close"].iloc[-1]) for sym, df in bars.items() if not df.empty}

    log.info("Executing orders (dry_run=%s)...", args.dry)
    orders = execute_signals(signals, prices, dry_run=args.dry)

    if orders:
        log.info("Orders placed:")
        for o in orders:
            log.info("  %s", o)
    else:
        log.info("No orders placed.")


def cmd_tune(args):
    import pandas as pd
    from data import fetch_bars, fetch_all_sentiment, build_features
    from data.screener import get_watchlist
    from models import xgb as xgb_model
    from models.dataset import prepare_combined, train_val_test_split

    watchlist = get_watchlist()

    log.info("--- PHASE 1/2: Building features ---")
    bars = fetch_bars(watchlist)
    feature_dfs = {}
    for i, (sym, df) in enumerate(bars.items(), 1):
        log.info("  [%d/%d] %s |%d bars", i, len(bars), sym, len(df))
        sentiment = fetch_all_sentiment([sym])[sym]
        feature_dfs[sym] = build_features(sym, df, sentiment)

    combined = prepare_combined(feature_dfs)
    train_df, val_df, _ = train_val_test_split(combined)
    log.info(
        "Dataset split |train: %d rows, val: %d rows",
        len(train_df), len(val_df),
    )

    log.info("--- PHASE 2/2: Optuna hyperparameter search (%d trials) ---", args.trials)
    best_params = xgb_model.tune(train_df, val_df, n_trials=args.trials)

    log.info("Best params found:")
    for k, v in best_params.items():
        log.info("  %s = %s", k, v)
    log.info("Re-run `python main.py train` to train XGBoost with the new params.")


def cmd_screen(args):
    from data.screener import fetch_sp500, screen_by_volume, save_watchlist

    log.info("Fetching S&P 500 ticker list...")
    sp500 = fetch_sp500()
    log.info("Found %d symbols in S&P 500", len(sp500))

    log.info("Screening by volume (min=%d, top_n=%d)...", args.min_volume, args.top)
    results = screen_by_volume(sp500, min_avg_volume=args.min_volume, top_n=args.top)

    print(f"\nTop {len(results)} symbols by average daily volume:")
    for i, sym in enumerate(results, 1):
        print(f"  {i:3d}. {sym}")

    if args.save:
        save_watchlist(results)
        log.info(
            "Watchlist saved |`fetch` and `train` commands will now use these %d symbols.",
            len(results),
        )


def main():
    parser = argparse.ArgumentParser(prog="templar")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Download/refresh market data")
    p_fetch.add_argument("--refresh", action="store_true", help="Force re-download")

    sub.add_parser("train", help="Train TFT + XGBoost models")

    p_run = sub.add_parser("run", help="Generate signals and trade")
    p_run.add_argument("--dry", action="store_true", help="Log orders without submitting")

    p_tune = sub.add_parser("tune", help="Optuna hyperparameter search for XGBoost")
    p_tune.add_argument(
        "--trials", type=int, default=50, metavar="N",
        help="Number of Optuna trials (default: 50)",
    )

    p_screen = sub.add_parser("screen", help="Screen S&P 500 by average daily volume")
    p_screen.add_argument(
        "--top", type=int, default=50, metavar="N",
        help="Return top N symbols (default: 50)",
    )
    p_screen.add_argument(
        "--min-volume", type=int, default=5_000_000, metavar="N",
        help="Minimum average daily volume (default: 5000000)",
    )
    p_screen.add_argument(
        "--save", action="store_true",
        help="Save results as new watchlist (fetch/train will use it)",
    )

    args = parser.parse_args()

    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "tune":
        cmd_tune(args)
    elif args.command == "screen":
        cmd_screen(args)


if __name__ == "__main__":
    main()
