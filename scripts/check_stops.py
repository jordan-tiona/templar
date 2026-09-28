"""On-demand check: any open position currently missing a live stop-loss order?

Run anytime with: venv/Scripts/python.exe scripts/check_stops.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.risk import get_client, positions_missing_stop

missing = positions_missing_stop(get_client())
if missing:
    print(f"{len(missing)} position(s) with no live stop-loss order: {missing}")
    sys.exit(1)
print("All open positions have a live stop-loss order.")
