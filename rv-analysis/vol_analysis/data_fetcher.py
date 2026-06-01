"""
data_fetcher.py — Bloomberg ticker construction, data pulls, and cleaning.

OHLC is fetched once and reused across all tenors.
Each IV ticker is fetched once per tenor.
"""

from __future__ import annotations

import datetime

import pandas as pd

from vol_analysis.bbg_connector import bdh
from vol_analysis.config import TENOR_MAP, FFILL_LIMIT


# ── Ticker builders ───────────────────────────────────────────────────────────

def ohlc_ticker(pair: str) -> str:
    """Return the Bloomberg OHLC ticker for *pair* (e.g. 'EURUSD Curncy')."""
    return f"{pair} Curncy"


def iv_ticker(pair: str, tenor: str) -> str:
    """Return the Bloomberg ATM implied-vol ticker for *pair* / *tenor*.

    Example: pair='EURUSD', tenor='3M' → 'EURUSDV3M Curncy'
    """
    template = TENOR_MAP[tenor]["iv_ticker"]
    return template.replace("{PAIR}", pair)


# ── Data fetchers ─────────────────────────────────────────────────────────────

def fetch_ohlc(
    pair: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch OHLC daily data from Bloomberg for *pair*.

    Returns a cleaned DataFrame with columns [PX_OPEN, PX_HIGH, PX_LOW, PX_LAST]
    and a DatetimeIndex.
    """
    ticker = ohlc_ticker(pair)
    fields = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST"]

    raw = bdh(ticker, fields, start_date, end_date)
    return _clean(raw, fields)


def fetch_iv(
    pair: str,
    tenor: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Fetch ATM implied-volatility time series from Bloomberg.

    Returns a cleaned pd.Series named 'IV' (values in %, e.g. 8.5).
    """
    ticker = iv_ticker(pair, tenor)
    field  = "PX_LAST"

    raw = bdh(ticker, [field], start_date, end_date)
    s   = _clean(raw, [field])[field].rename("IV")
    return s


# ── Cleaning ──────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    """Forward-fill up to FFILL_LIMIT days, then drop remaining NaNs.

    Also ensures the index is a DatetimeIndex sorted ascending.
    """
    df = df[required_cols].copy()
    df.sort_index(inplace=True)
    df.ffill(limit=FFILL_LIMIT, inplace=True)
    df.dropna(inplace=True)
    return df
