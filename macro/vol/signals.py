# vol/signals.py
# Core signal: z-score of ln(IV / RV_YZ) per pair x tenor.
#
#   ratio_t = ln( IV_t / YZ_t(window=tenor) )
#   z_t     = (ratio_t - mean(ratio, lookback)) / std(ratio, lookback)
#
# Positive z  -> implied rich vs its own history of premium over realized
# Negative z  -> implied cheap
#
# Percentile rank over the same lookback is provided as a fat-tail-honest
# companion to the z-score.

from __future__ import annotations

import numpy as np
import pandas as pd

from config.vol_universe import rv_window
from vol.estimators import yang_zhang


def realized_panel(
    ohlc_by_pair: dict[str, pd.DataFrame],
    tenor: str,
) -> pd.DataFrame:
    """Rolling YZ realized vol per pair, window matched to the tenor."""
    n = rv_window(tenor)
    cols = {p: yang_zhang(df, n) for p, df in ohlc_by_pair.items()}
    return pd.DataFrame(cols)


def log_ratio_panel(
    implied: pd.DataFrame,
    realized: pd.DataFrame,
) -> pd.DataFrame:
    """ln(IV/RV) aligned on common dates/pairs, ffill implied up to 3 days."""
    common = implied.columns.intersection(realized.columns)
    iv = implied[common].ffill(limit=3)
    rv = realized[common]
    idx = iv.index.intersection(rv.index)
    ratio = np.log(iv.loc[idx] / rv.loc[idx])
    return ratio.replace([np.inf, -np.inf], np.nan)


def zscore_latest(ratio: pd.DataFrame, lookback: int) -> pd.Series:
    """Z-score of the latest ln(IV/RV) vs its `lookback`-day history."""
    tail = ratio.tail(lookback)
    mu = tail.mean()
    sd = tail.std(ddof=1)
    return (ratio.iloc[-1] - mu) / sd


def percentile_latest(ratio: pd.DataFrame, lookback: int) -> pd.Series:
    """Percentile rank (0-100) of the latest value within the lookback."""
    tail = ratio.tail(lookback)
    return tail.rank(pct=True).iloc[-1] * 100.0


def zscore_series(ratio: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Full rolling z-score history (for sparklines / drill-down)."""
    mu = ratio.rolling(lookback).mean()
    sd = ratio.rolling(lookback).std(ddof=1)
    return (ratio - mu) / sd


def build_snapshot(
    implied_by_tenor: dict[str, pd.DataFrame],
    ohlc_by_pair: dict[str, pd.DataFrame],
    tenors: list[str],
    lookback: int,
) -> dict[str, pd.DataFrame]:
    """Assemble everything the heatmap needs.

    Returns dict of DataFrames [pair x tenor]:
      'z'          latest z-score of ln(IV/RV)
      'pct'        latest percentile of ln(IV/RV)
      'iv'         latest implied (vol pts)
      'rv'         latest YZ realized (vol pts)
      'ratio_hist' dict tenor -> ln(IV/RV) full panel (for drill-down)
    """
    z, pct, iv_now, rv_now = {}, {}, {}, {}
    ratio_hist: dict[str, pd.DataFrame] = {}

    for tenor in tenors:
        rv = realized_panel(ohlc_by_pair, tenor)
        ratio = log_ratio_panel(implied_by_tenor[tenor], rv)
        ratio_hist[tenor] = ratio

        z[tenor] = zscore_latest(ratio, lookback)
        pct[tenor] = percentile_latest(ratio, lookback)
        iv_now[tenor] = implied_by_tenor[tenor].ffill(limit=3).iloc[-1]
        rv_now[tenor] = rv.iloc[-1]

    return {
        "z": pd.DataFrame(z),
        "pct": pd.DataFrame(pct),
        "iv": pd.DataFrame(iv_now),
        "rv": pd.DataFrame(rv_now),
        "ratio_hist": ratio_hist,
    }
