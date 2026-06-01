"""
data/cme.py
-----------
Pulls SOFR / Fed Funds futures price data.

Two backends are supported:
  1. yfinance  — free, good enough for prototyping (default)
  2. CME DataMine — production-grade tick/daily data (requires paid account)

The public interface is identical regardless of backend, so switching later
requires no changes to signal or backtest code.

Dependencies:
    pip install yfinance pandas

Usage:
    from data.cme import fetch_continuous_front, fetch_implied_rate, load_futures_cache

    df = fetch_continuous_front()           # OHLCV + implied rate, daily
    df = fetch_implied_rate(df)             # adds implied_rate column if not present
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _futures_price_to_rate(price: pd.Series) -> pd.Series:
    """
    Convert CME-convention futures price (100 - rate) to implied rate (%).
    e.g. 95.50 -> 4.50%
    """
    from config import FUTURES_PRICE_OFFSET
    return (FUTURES_PRICE_OFFSET - price).round(4)


def _validate_dates(
    start: Optional[str],
    end: Optional[str],
) -> tuple[str, str]:
    from config import DEFAULT_START_DATE
    start = start or DEFAULT_START_DATE
    end   = end   or datetime.today().strftime("%Y-%m-%d")
    return start, end


# ---------------------------------------------------------------------------
# Backend 1: yfinance (prototype)
# ---------------------------------------------------------------------------

def _fetch_yfinance(
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Pull OHLCV from yfinance for a given futures ticker.

    Note: yfinance futures coverage is limited for SOFR (SR3=F).
    If SR3=F returns empty data, we fall back to Fed Funds (ZQ=F),
    which has near-identical signal properties for the front contract
    and a longer available history.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    logger.info("yfinance: downloading %s  [%s -> %s]", ticker, start, end)
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        logger.warning(
            "yfinance returned empty data for %s. "
            "Falling back to Fed Funds front (ZQ=F).",
            ticker,
        )
        from config import FF_FRONT_YFTICKER
        raw = yf.download(
            FF_FRONT_YFTICKER, start=start, end=end,
            auto_adjust=True, progress=False,
        )
        if raw.empty:
            raise RuntimeError(
                "Both SR3=F and ZQ=F returned empty data from yfinance. "
                "Check your internet connection or consider using CME DataMine."
            )

    # yfinance returns MultiIndex columns when downloading a single ticker
    # in newer versions — flatten if needed
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index   = pd.to_datetime(df.index)
    df.index.name = "date"

    # Drop any rows where close is NaN or zero (bad ticks from yfinance)
    df = df[(df["close"] > 0) & df["close"].notna()]

    logger.info(
        "  -> %d rows  (%s to %s)  close range [%.3f, %.3f]",
        len(df),
        df.index[0].date(),
        df.index[-1].date(),
        df["close"].min(),
        df["close"].max(),
    )

    return df


# ---------------------------------------------------------------------------
# Backend 2: CME DataMine stub (production)
# ---------------------------------------------------------------------------

def _fetch_cme_datamine(
    product_code: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Stub for CME DataMine API.
    Implement once CME_USERNAME / CME_PASSWORD are configured.

    CME DataMine provides clean, roll-adjusted continuous contracts
    and tick data. See: https://datamine.cmegroup.com

    When implemented, return a DataFrame with the same schema as
    _fetch_yfinance: columns [open, high, low, close, volume],
    DatetimeIndex named 'date'.
    """
    from config import CME_USERNAME, CME_PASSWORD

    if not CME_USERNAME or not CME_PASSWORD:
        raise NotImplementedError(
            "CME DataMine credentials not set. "
            "Set CME_USERNAME and CME_PASSWORD environment variables, "
            "or use the yfinance backend for prototyping."
        )

    # TODO: implement CME DataMine HTTP client
    # Reference: https://datamine.cmegroup.com/cme/api/v1/datamine/...
    raise NotImplementedError("CME DataMine backend not yet implemented.")


# ---------------------------------------------------------------------------
# Roll-adjustment logic
# ---------------------------------------------------------------------------

def build_continuous_series(
    contracts: dict[str, pd.DataFrame],
    method: str = "back_adjust",
) -> pd.DataFrame:
    """
    Stitch multiple contract DataFrames into a single continuous series.

    Parameters
    ----------
    contracts : dict mapping expiry code -> OHLCV DataFrame
                e.g. {"SR3H25": df_h25, "SR3M25": df_m25, ...}
                DataFrames must be sorted by date ascending.
    method    : "back_adjust"  — panama canal / backwards ratio adjustment
                                 (prices are consistent; returns are exact)
                "none"         — no adjustment, raw prices spliced at roll date

    Returns
    -------
    pd.DataFrame with same schema as individual contract DataFrames,
    plus a 'contract' column indicating which contract was front at each date.

    Note: for the prototype, yfinance's continuous contract (SR3=F / ZQ=F)
    is used directly and this function is not needed. It becomes important
    when stitching individual CME DataMine contract files.
    """
    if not contracts:
        raise ValueError("contracts dict is empty.")

    sorted_codes = sorted(contracts.keys())
    frames = []

    for i, code in enumerate(sorted_codes):
        df = contracts[code].copy()
        df["contract"] = code

        if i < len(sorted_codes) - 1:
            next_code  = sorted_codes[i + 1]
            next_start = contracts[next_code].index[0]
            # Hold front contract until the next one starts
            df = df[df.index < next_start]

        frames.append(df)

    continuous = pd.concat(frames).sort_index()

    if method == "back_adjust":
        # Compute backward ratio adjustment at each roll
        # (shift all prior prices so returns are continuous across rolls)
        roll_dates = [contracts[c].index[0] for c in sorted_codes[1:]]
        for roll_date in reversed(roll_dates):
            idx_before = continuous.index[continuous.index < roll_date]
            idx_after  = continuous.index[continuous.index >= roll_date]
            if len(idx_before) == 0 or len(idx_after) == 0:
                continue
            close_before = continuous.loc[idx_before[-1], "close"]
            close_after  = continuous.loc[idx_after[0],  "close"]
            ratio = close_after / close_before if close_before != 0 else 1.0
            for col in ["open", "high", "low", "close"]:
                continuous.loc[idx_before, col] *= ratio

    return continuous


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_continuous_front(
    start: Optional[str] = None,
    end:   Optional[str] = None,
    backend: str = "yfinance",
) -> pd.DataFrame:
    """
    Fetch the continuous front-month SOFR (or FF) futures contract.

    Adds derived columns:
        implied_rate  : 100 - close  (annualised %, CME convention)
        log_return    : log(close_t / close_{t-1})
        price_change  : close_t - close_{t-1}  (in price points)

    Parameters
    ----------
    start   : start date "YYYY-MM-DD"
    end     : end date   "YYYY-MM-DD"
    backend : "yfinance" (default, free) or "cme_datamine" (production)

    Returns
    -------
    pd.DataFrame with DatetimeIndex and columns:
        open, high, low, close, volume,
        implied_rate, log_return, price_change
    """
    import numpy as np

    start, end = _validate_dates(start, end)

    if backend == "yfinance":
        from config import SOFR_FRONT_YFTICKER
        df = _fetch_yfinance(SOFR_FRONT_YFTICKER, start, end)
    elif backend == "cme_datamine":
        from config import SOFR_QUARTERLY_CODES
        df = _fetch_cme_datamine(SOFR_QUARTERLY_CODES[0], start, end)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Choose 'yfinance' or 'cme_datamine'.")

    # Derived columns
    df["implied_rate"]  = _futures_price_to_rate(df["close"])
    df["log_return"]    = np.log(df["close"] / df["close"].shift(1))
    df["price_change"]  = df["close"].diff()

    df = df.dropna(subset=["log_return"])   # drop first row (no prior close)

    return df


def fetch_implied_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience: ensure a futures DataFrame has an implied_rate column.
    Adds it if missing; no-op if already present.
    """
    if "implied_rate" not in df.columns:
        df = df.copy()
        df["implied_rate"] = _futures_price_to_rate(df["close"])
    return df


def compute_realised_vol(
    df: pd.DataFrame,
    window: int = 20,
    annualise: bool = True,
) -> pd.Series:
    """
    Compute rolling realised volatility from log returns.

    Parameters
    ----------
    df        : DataFrame with log_return column (from fetch_continuous_front)
    window    : rolling window in business days (default 20 ≈ 1 month)
    annualise : if True, multiply by sqrt(252)

    Returns
    -------
    pd.Series named 'realised_vol'

    This is one of the two primary observable inputs to the regime
    classifier (alongside return autocorrelation).
    """
    import numpy as np

    if "log_return" not in df.columns:
        raise KeyError("DataFrame must have a 'log_return' column.")

    vol = df["log_return"].rolling(window).std()
    if annualise:
        vol = vol * np.sqrt(252)
    vol.name = f"realised_vol_{window}d"
    return vol


def compute_autocorrelation(
    df: pd.DataFrame,
    window: int = 20,
    lag: int = 1,
) -> pd.Series:
    """
    Compute rolling return autocorrelation at a given lag.
    Positive = momentum regime; negative = mean-reversion / crisis.

    Parameters
    ----------
    df     : DataFrame with log_return column
    window : rolling window in business days
    lag    : autocorrelation lag (default 1 day)

    Returns
    -------
    pd.Series named 'autocorr_{lag}d'

    This is the second primary observable input to the regime classifier.
    """
    if "log_return" not in df.columns:
        raise KeyError("DataFrame must have a 'log_return' column.")

    ac = df["log_return"].rolling(window).apply(
        lambda x: x.autocorr(lag=lag), raw=False
    )
    ac.name = f"autocorr_{lag}d"
    return ac


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def save_futures_cache(df: pd.DataFrame) -> None:
    """Persist futures DataFrame to parquet."""
    from config import CACHE_SOFR_FUTURES
    df.to_parquet(CACHE_SOFR_FUTURES)
    logger.info("Futures cache saved -> %s", CACHE_SOFR_FUTURES)


def load_futures_cache() -> pd.DataFrame:
    """Load futures DataFrame from parquet cache."""
    from config import CACHE_SOFR_FUTURES
    if not CACHE_SOFR_FUTURES.exists():
        raise FileNotFoundError(
            f"Futures cache not found at {CACHE_SOFR_FUTURES}. "
            "Run fetch_continuous_front() first."
        )
    df = pd.read_parquet(CACHE_SOFR_FUTURES)
    logger.info("Futures cache loaded: %d rows", len(df))
    return df