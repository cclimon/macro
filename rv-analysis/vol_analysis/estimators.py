"""
estimators.py — Yang-Zhang realized volatility and EWMA (RiskMetrics) volatility.

Both estimators return annualised percentage volatility series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vol_analysis.config import ANNUALISATION_FACTOR, EWMA_LAMBDA


# ── Yang-Zhang realized volatility ───────────────────────────────────────────

def yang_zhang_rv(ohlc: pd.DataFrame, window: int) -> pd.Series:
    """Full Yang-Zhang realized volatility estimator.

    Combines three variance components:
      - Overnight variance  (close-to-open)
      - Open-to-close variance (Rogers-Satchell component)
      - Weighted combination per the Yang-Zhang formula

    Parameters
    ----------
    ohlc:   DataFrame with columns PX_OPEN, PX_HIGH, PX_LOW, PX_LAST.
    window: Rolling window length (trading days).

    Returns
    -------
    pd.Series of annualised YZ volatility in % (e.g. 8.5 means 8.5%).
    """
    o = np.log(ohlc["PX_OPEN"])
    h = np.log(ohlc["PX_HIGH"])
    l = np.log(ohlc["PX_LOW"])
    c = np.log(ohlc["PX_LAST"])

    # Overnight return: log(Open_t / Close_{t-1})
    c_prev    = c.shift(1)
    overnight = o - c_prev

    # Open-to-close return
    oc_ret    = c - o

    # Rogers-Satchell variance (intra-day component, mean-zero by construction)
    rs_var = (h - o) * (h - c) + (l - o) * (l - c)

    # Rolling variances
    k = 0.34 / (1.34 + (window + 1) / (window - 1))   # Yang-Zhang weighting constant

    var_overnight = overnight.rolling(window).var(ddof=1)
    var_oc        = oc_ret.rolling(window).var(ddof=1)
    var_rs        = rs_var.rolling(window).mean()

    yz_var = var_overnight + k * var_oc + (1 - k) * var_rs

    # Annualise and convert to %
    yz_vol = np.sqrt(yz_var * ANNUALISATION_FACTOR) * 100
    yz_vol.name = "YZ_RV"
    return yz_vol


# ── EWMA (RiskMetrics) volatility ────────────────────────────────────────────

def ewma_rv(ohlc: pd.DataFrame) -> pd.Series:
    """Close-to-close EWMA volatility (RiskMetrics λ = 0.94).

    Uses log returns on PX_LAST.

    Parameters
    ----------
    ohlc: DataFrame with at minimum a PX_LAST column.

    Returns
    -------
    pd.Series of annualised EWMA volatility in % (e.g. 8.5 means 8.5%).
    """
    log_ret = np.log(ohlc["PX_LAST"] / ohlc["PX_LAST"].shift(1)).dropna()

    # Squared returns; initialise variance with first observation
    sq_ret   = log_ret ** 2
    var_vals = np.empty(len(sq_ret))
    var_vals[0] = sq_ret.iloc[0]

    lam = EWMA_LAMBDA
    for i in range(1, len(sq_ret)):
        var_vals[i] = lam * var_vals[i - 1] + (1 - lam) * sq_ret.iloc[i]

    ewma_var = pd.Series(var_vals, index=sq_ret.index)

    # Annualise and convert to %
    ewma_vol = np.sqrt(ewma_var * ANNUALISATION_FACTOR) * 100
    ewma_vol.name = "EWMA_RV"
    return ewma_vol
