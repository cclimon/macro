"""
data/fred.py
------------
Pulls macro time series from the St. Louis FRED API.

Dependencies:
    pip install fredapi pandas

Usage:
    from data.fred import fetch_fred_series, fetch_all_macro

    df = fetch_all_macro()          # returns wide DataFrame, one col per series
    cpi = fetch_fred_series("CPIAUCSL", start="2018-01-01")
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_fred_client():
    """
    Instantiates a fredapi.Fred client using the key in config.
    Raises a clear error if the key is missing or fredapi is not installed.
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError(
            "fredapi is not installed. Run: pip install fredapi"
        )

    from config import FRED_API_KEY

    if FRED_API_KEY == "YOUR_FRED_API_KEY_HERE":
        raise ValueError(
            "FRED API key not set. Register at https://fred.stlouisfed.org "
            "and set the FRED_API_KEY environment variable."
        )

    return Fred(api_key=FRED_API_KEY)


def _validate_dates(
    start: Optional[str],
    end: Optional[str],
) -> tuple[str, str]:
    """Coerce date inputs to strings; default end to today."""
    from config import DEFAULT_START_DATE

    start = start or DEFAULT_START_DATE
    end   = end   or datetime.today().strftime("%Y-%m-%d")
    return start, end


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_fred_series(
    series_id: str,
    start: Optional[str] = None,
    end:   Optional[str] = None,
    frequency: Optional[str] = None,
    aggregation: str = "avg",
) -> pd.Series:
    """
    Fetch a single FRED series as a pandas Series with a DatetimeIndex.

    Parameters
    ----------
    series_id   : FRED series code, e.g. "CPIAUCSL"
    start       : start date string "YYYY-MM-DD" (defaults to config value)
    end         : end date string  "YYYY-MM-DD" (defaults to today)
    frequency   : optional frequency override, e.g. "d", "m", "q"
                  None = use native series frequency
    aggregation : aggregation method when downsampling ("avg", "sum", "eop")

    Returns
    -------
    pd.Series with name = series_id and DatetimeIndex
    """
    start, end = _validate_dates(start, end)
    fred = _get_fred_client()

    logger.info("Fetching FRED series: %s  [%s -> %s]", series_id, start, end)

    kwargs = dict(
        observation_start=start,
        observation_end=end,
    )
    if frequency:
        kwargs["frequency"]            = frequency
        kwargs["aggregation_method"]   = aggregation

    series = fred.get_series(series_id, **kwargs)
    series.name  = series_id
    series.index = pd.to_datetime(series.index)

    # Drop NaNs at the tails (FRED sometimes pads with NaN on release lag)
    series = series.dropna()

    logger.info(
        "  -> %d observations  (%s to %s)",
        len(series),
        series.index[0].date(),
        series.index[-1].date(),
    )

    return series


def fetch_all_macro(
    start: Optional[str] = None,
    end:   Optional[str] = None,
    resample_to_business_days: bool = True,
) -> pd.DataFrame:
    """
    Fetch all macro series defined in config.FRED_SERIES and return a
    wide DataFrame aligned to a common daily (business day) index.

    Mixed-frequency series (monthly CPI, daily FF rate) are forward-filled
    onto the daily spine — appropriate for signal construction where we want
    the most recently *known* value at each point in time.

    Parameters
    ----------
    start                    : start date (defaults to config value)
    end                      : end date   (defaults to today)
    resample_to_business_days: if True, reindex to business-day spine with ffill

    Returns
    -------
    pd.DataFrame, columns = FRED series names, DatetimeIndex (business days)
    """
    from config import FRED_SERIES, RESAMPLE_FREQ

    start, end = _validate_dates(start, end)
    frames: dict[str, pd.Series] = {}

    for name, series_id in FRED_SERIES.items():
        try:
            s = fetch_fred_series(series_id, start=start, end=end)
            s.name = name              # use human-readable column name
            frames[name] = s
        except Exception as exc:
            logger.warning("Failed to fetch %s (%s): %s", name, series_id, exc)

    if not frames:
        raise RuntimeError("No FRED series could be fetched. Check API key and connection.")

    df = pd.DataFrame(frames)

    if resample_to_business_days:
        bday_index = pd.bdate_range(start=start, end=end, freq=RESAMPLE_FREQ)
        df = df.reindex(bday_index)
        # Forward-fill: use last known value on non-release days
        df = df.ffill()
        # Back-fill a small window for series that start slightly after our start date
        df = df.bfill(limit=5)

    logger.info(
        "Macro DataFrame built: %d rows × %d columns",
        len(df), len(df.columns),
    )

    return df


def compute_yoy_changes(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Compute year-over-year percentage changes for level series (CPI, PCE etc.)
    and add them as new columns with suffix '_yoy_pct'.

    These are the *surprise* inputs for the OSINT event classifier.
    """
    for col in columns:
        if col in df.columns:
            df[f"{col}_yoy_pct"] = df[col].pct_change(periods=252) * 100
    return df


def compute_macro_surprise(
    actual: float,
    consensus: float,
    historical_std: float,
) -> float:
    """
    Normalise a data release surprise relative to its historical std deviation.
    Returns a z-score: positive = hawkish surprise, negative = dovish surprise.

    This scalar feeds directly into the OSINT impact magnitude for scheduled
    data release events (NFP, CPI, PCE).

    Parameters
    ----------
    actual         : the released figure
    consensus      : median analyst forecast (e.g. from Bloomberg survey)
    historical_std : rolling std of past surprises for this series

    Returns
    -------
    float: surprise z-score
    """
    if historical_std == 0:
        return 0.0
    return (actual - consensus) / historical_std


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def save_macro_cache(df: pd.DataFrame) -> None:
    """Persist macro DataFrame to parquet for fast local reload."""
    from config import CACHE_FRED_MACRO
    df.to_parquet(CACHE_FRED_MACRO)
    logger.info("Macro cache saved -> %s", CACHE_FRED_MACRO)


def load_macro_cache() -> pd.DataFrame:
    """Load macro DataFrame from parquet cache."""
    from config import CACHE_FRED_MACRO
    if not CACHE_FRED_MACRO.exists():
        raise FileNotFoundError(
            f"Macro cache not found at {CACHE_FRED_MACRO}. Run fetch_all_macro() first."
        )
    df = pd.read_parquet(CACHE_FRED_MACRO)
    logger.info("Macro cache loaded: %d rows × %d columns", len(df), len(df.columns))
    return df