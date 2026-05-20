"""
Templar — entry point.

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("templar")


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

    log.info("━━━ PHASE 1/3: Building features ━━━")
    bars = fetch_bars(watchlist)
    feature_dfs = {}
    for i, (sym, df) in enumerate(bars.items(), 1):
        log.info("  [%d/%d] %s — %d bars", i, len(bars), sym, len(df))
        sentiment = fetch_all_sentiment([sym])[sym]
        feature_dfs[sym] = build_features(sym, df, sentiment)

    combined = prepare_combined(feature_dfs)
    train_df, val_df, test_df = train_val_test_split(combined)
    log.info(
        "Dataset split — train: %d rows, val: %d rows, test: %d rows",
        len(train_df), len(val_df), len(test_df),
    )

    log.info("━━━ PHASE 2/3: Training XGBoost ━━━")
    xgb_model.train(train_df, val_df)
    log.info("XGBoost training complete")

    log.info("━━━ PHASE 3/3: Training TFT ━━━")
    log.info("This will take 20–40 minutes on CPU — progress shown per epoch below")
    tft_model.train(train_df, val_df)

    log.info("━━━ Training complete ━━━")


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

    log.info("━━━ PHASE 1/2: Building features ━━━")
    bars = fetch_bars(watchlist)
    feature_dfs = {}
    for i, (sym, df) in enumerate(bars.items(), 1):
        log.info("  [%d/%d] %s — %d bars", i, len(bars), sym, len(df))
        sentiment = fetch_all_sentiment([sym])[sym]
        feature_dfs[sym] = build_features(sym, df, sentiment)

    combined = prepare_combined(feature_dfs)
    train_df, val_df, _ = train_val_test_split(combined)
    log.info(
        "Dataset split — train: %d rows, val: %d rows",
        len(train_df), len(val_df),
    )

    log.info("━━━ PHASE 2/2: Optuna hyperparameter search (%d trials) ━━━", args.trials)
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
            "Watchlist saved — `fetch` and `train` commands will now use these %d symbols.",
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
