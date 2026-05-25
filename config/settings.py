from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

ROOT = Path(__file__).parent.parent

# Paths
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_CACHE = ROOT / "data" / "cache"
MODEL_TFT = ROOT / "models" / "tft"
MODEL_XGB = ROOT / "models" / "xgboost"
LOGS = ROOT / "logs"

for p in [DATA_RAW, DATA_PROCESSED, DATA_CACHE, MODEL_TFT, MODEL_XGB, LOGS]:
    p.mkdir(parents=True, exist_ok=True)

# Alpaca
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Alpha Vantage
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Reddit
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "Templar/1.0")

# Universe — starting small, easy to expand
WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "JPM", "BAC", "GS",
    "XOM", "CVX",
    "JNJ", "UNH",
    "SPY",   # S&P 500 ETF — useful macro feature
]

# Data
HISTORY_YEARS = 5
DAILY_BAR_TIMEFRAME = "1Day"

# Model
SEQUENCE_LENGTH = 60       # lookback window in trading days (~3 months)
PREDICTION_HORIZON = 20    # days ahead to predict (4 weeks — more signal, less noise than 5d)
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
# test = remaining 0.15

# Watchlist file path (overrides WATCHLIST when present)
WATCHLIST_PATH = ROOT / "watchlist.json"

# Model — TFT
TFT_LEARNING_RATE = 3e-4
TFT_HIDDEN_SIZE = 32           # reduced from 96; model was too large for 40K rows
TFT_ATTENTION_HEADS = 2        # reduced from 4
TFT_DROPOUT = 0.5              # increased from 0.3

# Ensemble weights — adjust as model IC warrants
# TFT weight set to 0 until it demonstrates positive test IC
TFT_ENSEMBLE_WEIGHT = 0.0
XGB_ENSEMBLE_WEIGHT = 1.0

# Ollama plain-English summaries
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"   # change to e.g. "qwen2.5:7b" for higher quality

# Signal thresholds
BUY_THRESHOLD = 0.6        # ensemble confidence to trigger buy
SELL_THRESHOLD = 0.4       # below this triggers sell

# Risk
MAX_POSITION_PCT = 0.05    # max 5% of portfolio per position
MAX_DRAWDOWN_PCT = 0.15    # halt trading if portfolio drawdown exceeds 15%
STOP_LOSS_PCT    = 0.07    # close position if it moves 7% against entry
TAKE_PROFIT_PCT  = 0.20   # close position if it gains 20% from entry
