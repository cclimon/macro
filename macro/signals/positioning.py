"""
positioning.py
FX Positioning tag for G10 dashboard — built EUR/USD/JPY first, extensible to G10/EM.
Pillars: Risk Reversal (Spectra-style, 40/60 1m/6m), Trend (RSI+MA dev),
         CFTC (hybrid level/change, BNP-style percentile base).
Final output normalized to -50 (max short) / +50 (max long).
Pattern mirrors pmi.py for repo coherence.
"""

import logging
import pandas as pd
import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# ============================================================
# CONFIG — extend these dicts as new currencies are added
# ============================================================

ACTIVE_CURRENCIES = ["EUR", "JPY"]  # USD derived, not listed here
HIST_YEARS = 6
ZSCORE_WINDOW = 180          # trading days, per spec
PCTILE_WINDOW = ZSCORE_WINDOW * 7  # ~5y equivalent for CFTC level percentile (1260d)
MIN_PERIODS_FRAC = 0.8        # require 80% of window before scoring

RR_WEIGHTS = {"1m": 0.4, "6m": 0.6}
CFTC_WEIGHTS = {"level": 0.7, "change": 0.3}
PILLAR_WEIGHTS = {"rr": 1 / 3, "trend": 1 / 3, "cftc": 1 / 3}  # equal-weight per spec

POSITIONING_TICKERS = {
    "spot": {
        "EUR": "EURUSD Curncy",
        "JPY": "USDJPY Curncy",
    },
    "risk_reversal_1m": {
        "EUR": "EURUSD25R1M Curncy",
        "JPY": "USDJPY25R1M Curncy",
    },
    "risk_reversal_6m": {
        "EUR": "EURUSD25R6M Curncy",
        "JPY": "USDJPY25R6M BGN Curncy",
    },
    "cftc_long": {
        "EUR": "IMMBENCL Index",
        "JPY": "IMM5JNCL Index",
    },
    "cftc_short": {
        "EUR": "IMMBENCS Index",
        "JPY": "IMM5JNCS Index",
    },
}

# parked — fill in and add ticker to ACTIVE_CURRENCIES when extending
PARKED_TICKERS = {
    "spot": {"GBP": "GBPUSD Curncy"},
    "risk_reversal_1m": {"GBP": "GBPUSD25R1M Curncy"},
    "risk_reversal_6m": {"GBP": "GBPUSD25R6M Curncy"},
    "cftc_long": {"GBP": "IMM5PNCL Index"},
    "cftc_short": {"GBP": "IMM5PNCS Index"},  # unverified
}


# ============================================================
# NaN HANDLING (best practice, per pillar data frequency)
# ============================================================

def _clean_daily_series(s: pd.Series, max_ffill_days: int = 3) -> pd.Series:
    """For naturally-daily series (spot, RR): ffill short gaps (holidays), don't mask real outages."""
    return s.ffill(limit=max_ffill_days)


def _clean_weekly_series(s: pd.Series, daily_index: pd.DatetimeIndex,
                          max_ffill_days: int = 9) -> pd.Series:
    """
    CFTC is weekly (Tue cut / Fri release per BNP methodology).
    Reindex to daily and ffill, capped at 9 days so a missed release doesn't
    silently carry forward indefinitely.
    """
    s = s.reindex(s.index.union(daily_index)).sort_index()
    s = s.ffill(limit=max_ffill_days)
    return s.reindex(daily_index)


# ============================================================
# PILLAR SCORING FUNCTIONS
# ============================================================

def _zscore(s: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    min_p = int(window * MIN_PERIODS_FRAC)
    mean = s.rolling(window, min_periods=min_p).mean()
    std = s.rolling(window, min_periods=min_p).std()
    return (s - mean) / std


def rr_score_spectra(rr_1m: pd.Series, rr_6m: pd.Series) -> pd.Series:
    z1m = _zscore(rr_1m, ZSCORE_WINDOW)
    z6m = _zscore(rr_6m, ZSCORE_WINDOW)
    return z1m * RR_WEIGHTS["1m"] + z6m * RR_WEIGHTS["6m"]


def trend_score(spot: pd.Series) -> pd.Series:
    delta = spot.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    dev_20 = spot - spot.rolling(20).mean()
    dev_100 = spot - spot.rolling(100).mean()

    rsi_z = _zscore(rsi, ZSCORE_WINDOW)
    dev20_z = _zscore(dev_20, ZSCORE_WINDOW)
    dev100_z = _zscore(dev_100, ZSCORE_WINDOW)

    return pd.concat([rsi_z, dev20_z, dev100_z], axis=1).mean(axis=1)


def cftc_score_hybrid(cftc_long: pd.Series, cftc_short: pd.Series,
                       daily_index: pd.DatetimeIndex) -> pd.Series:
    net_oi = (cftc_long - cftc_short)
    net_oi_daily = _clean_weekly_series(net_oi, daily_index)

    min_p = int(PCTILE_WINDOW * MIN_PERIODS_FRAC)
    level_pctile = net_oi_daily.rolling(PCTILE_WINDOW, min_periods=min_p).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    level_z_equiv = norm.ppf(level_pctile.clip(0.001, 0.999))  # pctile -> z-equivalent

    chg = net_oi_daily.diff(20)  # ~4 trading weeks
    chg_z = _zscore(chg, ZSCORE_WINDOW)

    return pd.Series(level_z_equiv, index=net_oi_daily.index) * CFTC_WEIGHTS["level"] \
        + chg_z * CFTC_WEIGHTS["change"]


def _zscore_to_pm50(combined_z: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    """
    Final normalization: rolling percentile of the composite z-score,
    mapped to -50/+50 (BNP convention). Keeps scale consistent across
    pillars regardless of how each pillar was internally scored.
    """
    min_p = int(window * MIN_PERIODS_FRAC)
    pctile = combined_z.rolling(window, min_periods=min_p).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    return (pctile - 0.5) * 100  # -> -50/+50


# ============================================================
# BUILD PER CURRENCY
# ============================================================

def build_positioning_for_currency(ccy: str, spot, rr_1m, rr_6m,
                                    cftc_long, cftc_short) -> pd.Series:
    daily_index = spot.index
    spot = _clean_daily_series(spot)
    rr_1m = _clean_daily_series(rr_1m)
    rr_6m = _clean_daily_series(rr_6m)

    rr = rr_score_spectra(rr_1m, rr_6m)
    trend = trend_score(spot)
    cftc = cftc_score_hybrid(cftc_long, cftc_short, daily_index)

    composite_z = (rr * PILLAR_WEIGHTS["rr"]
                   + trend * PILLAR_WEIGHTS["trend"]
                   + cftc * PILLAR_WEIGHTS["cftc"])

    score = _zscore_to_pm50(composite_z)
    score.name = f"{ccy}_positioning"
    return score


def build_usd_positioning(currency_scores: dict) -> pd.Series:
    """USD leg = inverse-weighted average of all other active legs (BNP/Spectra convention)."""
    df = pd.concat(currency_scores.values(), axis=1)
    usd_score = -1 * df.mean(axis=1)
    usd_score.name = "USD_positioning"
    return usd_score


# ============================================================
# DASHBOARD INTEGRATION
# ============================================================

def build_positioning_tag(data: dict) -> pd.DataFrame:
    """
    data: nested dict of pulled BBG series, structure:
      data[ticker_category][ccy] -> pd.Series, e.g. data["spot"]["EUR"]

    Returns a DataFrame, one column per currency (incl. derived USD),
    ready to merge into the G10 dashboard alongside the PMI tag.
    """
    scores = {}
    for ccy in ACTIVE_CURRENCIES:
        spot_index = data["spot"][ccy].index
        nan_series = pd.Series(float("nan"), index=spot_index)

        def _get(category, fallback=None, _ccy=ccy, _idx=spot_index):
            s = data.get(category, {}).get(_ccy)
            if s is None and fallback:
                s = data.get(fallback, {}).get(_ccy)
                if s is not None:
                    logger.warning("%s: %s missing, falling back to %s", _ccy, category, fallback)
            return s.reindex(_idx) if s is not None else nan_series

        scores[ccy] = build_positioning_for_currency(
            ccy,
            spot=data["spot"][ccy],
            rr_1m=_get("risk_reversal_1m"),
            rr_6m=_get("risk_reversal_6m", fallback="risk_reversal_1m"),
            cftc_long=_get("cftc_long"),
            cftc_short=_get("cftc_short"),
        )
    scores["USD"] = build_usd_positioning(scores)
    return pd.concat(scores.values(), axis=1)
