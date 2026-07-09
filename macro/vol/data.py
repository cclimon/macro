# vol/data.py
# Data acquisition layer — Bloomberg first (xbbg), synthetic mock fallback
# so the dashboard can be developed/tested away from the terminal.

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from config.vol_universe import (
    TENORS,
    UNIVERSE,
    implied_ticker,
    spot_ticker,
)

OHLC_FIELDS = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST"]
FIELD_MAP = {"PX_OPEN": "open", "PX_HIGH": "high", "PX_LOW": "low", "PX_LAST": "close"}


def _bbg_available() -> bool:
    try:
        import blpapi  # noqa: F401
        return True
    except ImportError:
        return False


# ── Bloomberg pulls ──────────────────────────────────────────────────────────

def fetch_implied_history(
    pairs: list[str],
    tenors: list[str],
    start: dt.date,
    end: dt.date | None = None,
) -> dict[str, pd.DataFrame]:
    """Implied ATM vol history per tenor.

    Returns {tenor: DataFrame[date x pair]} in vol points.
    """
    from xbbg import blp

    end = end or dt.date.today()
    out: dict[str, pd.DataFrame] = {}
    for tenor in tenors:
        tickers = [implied_ticker(p, tenor) for p in pairs]
        raw = blp.bdh(tickers, "PX_LAST", start, end)
        raw.columns = raw.columns.droplevel(1)  # drop field level
        # map ticker back to pair
        rename = {implied_ticker(p, tenor): p for p in pairs}
        out[tenor] = raw.rename(columns=rename).sort_index()
    return out


def fetch_ohlc_history(
    pairs: list[str],
    start: dt.date,
    end: dt.date | None = None,
) -> dict[str, pd.DataFrame]:
    """Daily OHLC per pair. Returns {pair: DataFrame[date x open/high/low/close]}."""
    from xbbg import blp

    end = end or dt.date.today()
    out: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        raw = blp.bdh(spot_ticker(pair), OHLC_FIELDS, start, end)
        raw.columns = raw.columns.droplevel(0)  # drop ticker level
        out[pair] = raw.rename(columns=FIELD_MAP).sort_index()
    return out


# ── Mock data (development without a terminal) ──────────────────────────────

def _mock_ohlc(seed: int, days: int, base_vol: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=dt.date.today(), periods=days)
    daily = base_vol / np.sqrt(252) / 100.0

    close = np.empty(days)
    open_ = np.empty(days)
    close[0] = 1.0
    open_[0] = 1.0
    # regime-switching vol to make z-scores meaningful
    regime = np.clip(np.cumsum(rng.normal(0, 0.03, days)), -0.8, 1.5)
    dv = daily * np.exp(regime)
    for t in range(1, days):
        open_[t] = close[t - 1] * np.exp(rng.normal(0, dv[t] * 0.35))  # gap
        close[t] = open_[t] * np.exp(rng.normal(0, dv[t]))
    intraday = np.abs(rng.normal(0, dv, days))
    high = np.maximum(open_, close) * np.exp(intraday * 0.6)
    low = np.minimum(open_, close) * np.exp(-intraday * 0.6)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close}, index=idx
    )


def mock_data(
    pairs: list[str] | None = None,
    tenors: list[str] | None = None,
    days: int = 1100,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Synthetic (implied_by_tenor, ohlc_by_pair) with realistic-ish dynamics."""
    pairs = pairs or UNIVERSE
    tenors = tenors or list(TENORS)
    base_vols = {p: 6.0 + 10.0 * (i % 5) / 4 for i, p in enumerate(pairs)}

    ohlc = {p: _mock_ohlc(seed=abs(hash(p)) % 2**31, days=days, base_vol=base_vols[p])
            for p in pairs}

    implied: dict[str, pd.DataFrame] = {}
    rng = np.random.default_rng(7)
    for j, tenor in enumerate(tenors):
        cols = {}
        for p in pairs:
            # implied = smoothed realized proxy + premium + noise
            ret = np.log(ohlc[p]["close"]).diff()
            rv = ret.rolling(21).std() * np.sqrt(252) * 100
            prem = 0.8 + 0.15 * j
            cols[p] = (rv.ewm(span=10).mean() + prem
                       + rng.normal(0, 0.3, len(rv)))
        implied[tenor] = pd.DataFrame(cols).dropna(how="all")
    return implied, ohlc


def load(
    pairs: list[str],
    tenors: list[str],
    history_days: int,
    use_mock: bool | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], str]:
    """Unified loader. Returns (implied_by_tenor, ohlc_by_pair, source_label)."""
    if use_mock is None:
        use_mock = not _bbg_available()
    if use_mock:
        implied, ohlc = mock_data(pairs, tenors, days=history_days)
        return implied, ohlc, "MOCK DATA — no Bloomberg session"
    start = dt.date.today() - dt.timedelta(days=int(history_days * 1.6))
    implied = fetch_implied_history(pairs, tenors, start)
    ohlc = fetch_ohlc_history(pairs, start)
    return implied, ohlc, "Bloomberg (xbbg)"
