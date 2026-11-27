"""Verify whether the sentiment features ever join onto the bars (index timestamp mismatch?)."""
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd, glob
ROOT = Path(__file__).resolve().parent.parent
raw = pd.read_parquet(ROOT / "data/raw/AAPL.parquet")
sent = pd.read_parquet(ROOT / "data/cache/sentiment_news_AAPL.parquet")
print("raw bars index sample:", raw.index[-3:].tolist(), "dtype", raw.index.dtype)
print("sentiment index sample:", sent.index[-3:].tolist(), "dtype", sent.index.dtype, "rows", len(sent))
print("sentiment date range:", sent.index.min(), "->", sent.index.max())
print("intersection of index values:", len(raw.index.intersection(sent.index)))
# what build_features would produce
from data.features import merge_sentiment
m = merge_sentiment(raw.copy(), sent)
print("merged sentiment_mean nonzero rows:", (m.sentiment_mean != 0).sum(), "of", len(m), "; count>0 rows:", (m.sentiment_count > 0).sum())
# and what the training pipeline actually saved (processed parquet written by build_features on the last retrain)
tot = nz = 0
for p in glob.glob(str(ROOT / "data/processed/*.parquet")):
    d = pd.read_parquet(p)
    if "sentiment_mean" in d: tot += len(d); nz += (d.sentiment_mean != 0).sum()
print(f"processed parquets (last training run): {nz} of {tot} rows have nonzero sentiment_mean")
print("raw index hour-of-day distribution:", pd.Series(raw.index.hour).value_counts().to_dict())
