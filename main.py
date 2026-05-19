"""
Templar — entry point.

Commands:
  python main.py fetch      -- download/refresh bar data
  python main.py train      -- train TFT + XGBoost models
  python main.py run        -- generate signals and execute orders (paper)
  python main.py run --dry  -- generate signals, log orders but don't submit
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
    from config.settings import WATCHLIST
    log.info("Fetching bars...")
    fetch_bars(WATCHLIST, force_refresh=args.refresh)
    log.info("Fetching sentiment...")
    fetch_all_sentiment(WATCHLIST)
    log.info("Done.")


def cmd_train(args):
    import pandas as pd
    from data import fetch_bars, fetch_all_sentiment, build_features
    from models import tft as tft_model, xgb as xgb_model
    from models.dataset import prepare_combined, train_val_test_split
    from config.settings import WATCHLIST

    log.info("━━━ PHASE 1/3: Building features ━━━")
    bars = fetch_bars(WATCHLIST)
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
    from config.settings import WATCHLIST

    log.info("Generating signals...")
    signals = generate_signals(WATCHLIST)
    print(signals[["ensemble_score", "signal"]].to_string())

    # Latest close prices for position sizing
    bars = fetch_bars(WATCHLIST)
    prices = {sym: float(df["close"].iloc[-1]) for sym, df in bars.items() if not df.empty}

    log.info("Executing orders (dry_run=%s)...", args.dry)
    orders = execute_signals(signals, prices, dry_run=args.dry)

    if orders:
        log.info("Orders placed:")
        for o in orders:
            log.info("  %s", o)
    else:
        log.info("No orders placed.")


def main():
    parser = argparse.ArgumentParser(prog="templar")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Download/refresh market data")
    p_fetch.add_argument("--refresh", action="store_true", help="Force re-download")

    sub.add_parser("train", help="Train TFT + XGBoost models")

    p_run = sub.add_parser("run", help="Generate signals and trade")
    p_run.add_argument("--dry", action="store_true", help="Log orders without submitting")

    args = parser.parse_args()

    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
