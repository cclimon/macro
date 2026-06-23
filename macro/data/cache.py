# data/cache.py
# Persist and reload EOD signals from parquet + timestamp files.

import pandas as pd
from pathlib import Path
from datetime import datetime
import pytz

CACHE_DIR = Path(__file__).parent / "cache"

_PARQUET_KEYS = ("technical", "carry", "macro", "spot")


def save_signals(signals: dict) -> None:
    """Save technical/carry/macro/spot DataFrames and as_of timestamp to CACHE_DIR."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for key in _PARQUET_KEYS:
        df = signals.get(key)
        if df is not None and not df.empty:
            df.to_parquet(CACHE_DIR / f"{key}.parquet")

    as_of: datetime = signals.get("as_of", datetime.utcnow())
    (CACHE_DIR / "last_updated.txt").write_text(as_of.isoformat())


def load_signals() -> dict:
    """Load signals from parquet cache. Raises FileNotFoundError if cache is missing."""
    required = ("technical", "carry", "macro")
    missing = [k for k in required if not (CACHE_DIR / f"{k}.parquet").exists()]
    if missing or not (CACHE_DIR / "last_updated.txt").exists():
        raise FileNotFoundError("Cache files not found. Run python main.py first.")

    technical = pd.read_parquet(CACHE_DIR / "technical.parquet")
    carry     = pd.read_parquet(CACHE_DIR / "carry.parquet")
    macro     = pd.read_parquet(CACHE_DIR / "macro.parquet")

    spot_path = CACHE_DIR / "spot.parquet"
    spot      = pd.read_parquet(spot_path) if spot_path.exists() else pd.DataFrame()

    ts_raw = (CACHE_DIR / "last_updated.txt").read_text().strip()
    as_of  = datetime.fromisoformat(ts_raw)

    return {
        "technical": technical,
        "carry":     carry,
        "macro":     macro,
        "spot":      spot,
        "as_of":     as_of,
    }
