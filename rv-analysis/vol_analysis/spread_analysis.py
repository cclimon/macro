"""
spread_analysis.py — IV-to-RV spreads and rolling percentile ranks.
"""

from __future__ import annotations

import pandas as pd

from vol_analysis.config import PERCENTILE_WINDOWS


# ── Spreads ───────────────────────────────────────────────────────────────────

def compute_spreads(iv: pd.Series, yz_rv: pd.Series, ewma_rv: pd.Series) -> pd.DataFrame:
    """Compute IV − YZ and IV − EWMA spreads on aligned dates.

    Parameters
    ----------
    iv:      Implied vol series (%).
    yz_rv:   Yang-Zhang RV series (%).
    ewma_rv: EWMA RV series (%).

    Returns
    -------
    DataFrame with columns [IV, YZ_RV, EWMA_RV, Spread_YZ, Spread_EWMA].
    """
    df = pd.concat([iv.rename("IV"), yz_rv, ewma_rv], axis=1).dropna()
    df["Spread_YZ"]   = df["IV"] - df["YZ_RV"]
    df["Spread_EWMA"] = df["IV"] - df["EWMA_RV"]
    return df


# ── Rolling percentile rank ───────────────────────────────────────────────────

def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Compute the rolling percentile rank of *series* over *window* observations.

    The rank is defined as the fraction of past values (within the window) that
    are strictly less than the current value, expressed as 0–100.

    Parameters
    ----------
    series: Input time series.
    window: Look-back window in trading days.

    Returns
    -------
    pd.Series of percentile ranks in [0, 100].
    """
    def _pct_rank(x: pd.Series) -> float:
        current = x.iloc[-1]
        past    = x.iloc[:-1]
        if len(past) == 0:
            return float("nan")
        return (past < current).sum() / len(past) * 100

    return series.rolling(window, min_periods=window // 2).apply(_pct_rank, raw=False)


def compute_percentile_ranks(spread_df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling 1Y and 2Y percentile rank columns for both spreads.

    New columns added:
      YZ_pct_1Y, YZ_pct_2Y, EWMA_pct_1Y, EWMA_pct_2Y

    Parameters
    ----------
    spread_df: DataFrame from compute_spreads().

    Returns
    -------
    The same DataFrame with four additional percentile-rank columns.
    """
    df = spread_df.copy()
    for label, window in PERCENTILE_WINDOWS.items():
        df[f"YZ_pct_{label}"]   = rolling_percentile_rank(df["Spread_YZ"],   window)
        df[f"EWMA_pct_{label}"] = rolling_percentile_rank(df["Spread_EWMA"], window)
    return df
