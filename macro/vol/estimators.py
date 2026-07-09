# vol/estimators.py
# Realized volatility estimators from daily OHLC bars.
#
# Yang-Zhang (2000): minimum-variance, drift-independent combination of
#   - overnight (close-to-open) variance   -> captures gaps / weekend jumps
#   - open-to-close variance
#   - Rogers-Satchell intraday range term  -> drift-robust high/low information
#
# sigma^2_YZ = sigma^2_overnight + k * sigma^2_openclose + (1 - k) * sigma^2_RS
# k = 0.34 / (1.34 + (n + 1) / (n - 1))
#
# All functions return ANNUALIZED vol in PERCENTAGE points (e.g. 8.25),
# to match Bloomberg implied vol quoting.

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALIZATION = 252


def _log_components(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Log-return building blocks from an OHLC frame.

    Expects columns: open, high, low, close (case-insensitive).
    """
    df = ohlc.rename(columns=str.lower)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    out = pd.DataFrame(index=df.index)
    out["on"] = np.log(o / c.shift(1))   # overnight: prior close -> open
    out["oc"] = np.log(c / o)            # open -> close
    out["u"] = np.log(h / o)
    out["d"] = np.log(l / o)
    out["cc"] = np.log(c / c.shift(1))   # close-to-close (for reference est.)
    return out


def yang_zhang(
    ohlc: pd.DataFrame,
    window: int,
    annualization: int = ANNUALIZATION,
) -> pd.Series:
    """Rolling Yang-Zhang realized vol (annualized, in vol points).

    Parameters
    ----------
    ohlc : DataFrame with open/high/low/close columns, daily bars.
    window : rolling window in business days (match to the implied tenor).
    """
    x = _log_components(ohlc)
    n = window

    # Sample variances (ddof=1) of overnight and open-to-close returns
    var_on = x["on"].rolling(n).var(ddof=1)
    var_oc = x["oc"].rolling(n).var(ddof=1)

    # Rogers-Satchell term: mean of u(u-oc) + d(d-oc), drift-independent
    rs = x["u"] * (x["u"] - x["oc"]) + x["d"] * (x["d"] - x["oc"])
    var_rs = rs.rolling(n).mean()

    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var_yz = var_on + k * var_oc + (1 - k) * var_rs

    vol = np.sqrt(var_yz.clip(lower=0) * annualization) * 100.0
    vol.name = f"yz_{n}d"
    return vol


def close_to_close(
    ohlc: pd.DataFrame, window: int, annualization: int = ANNUALIZATION
) -> pd.Series:
    """Classic close-to-close realized vol — reference / sanity check."""
    x = _log_components(ohlc)
    vol = np.sqrt(x["cc"].rolling(window).var(ddof=1) * annualization) * 100.0
    vol.name = f"cc_{window}d"
    return vol


def parkinson(
    ohlc: pd.DataFrame, window: int, annualization: int = ANNUALIZATION
) -> pd.Series:
    """Parkinson high-low estimator (no drift, no jumps handled)."""
    df = ohlc.rename(columns=str.lower)
    hl2 = np.log(df["high"] / df["low"]) ** 2
    var = hl2.rolling(window).mean() / (4.0 * np.log(2.0))
    vol = np.sqrt(var * annualization) * 100.0
    vol.name = f"park_{window}d"
    return vol


def rogers_satchell(
    ohlc: pd.DataFrame, window: int, annualization: int = ANNUALIZATION
) -> pd.Series:
    """Rogers-Satchell estimator — drift-independent, ignores overnight."""
    x = _log_components(ohlc)
    rs = x["u"] * (x["u"] - x["oc"]) + x["d"] * (x["d"] - x["oc"])
    vol = np.sqrt(rs.rolling(window).mean().clip(lower=0) * annualization) * 100.0
    vol.name = f"rs_{window}d"
    return vol


def garman_klass(
    ohlc: pd.DataFrame, window: int, annualization: int = ANNUALIZATION
) -> pd.Series:
    """Garman-Klass estimator — efficient but assumes zero drift, no jumps."""
    x = _log_components(ohlc)
    gk = 0.5 * (x["u"] - x["d"]) ** 2 - (2.0 * np.log(2.0) - 1.0) * x["oc"] ** 2
    vol = np.sqrt(gk.rolling(window).mean().clip(lower=0) * annualization) * 100.0
    vol.name = f"gk_{window}d"
    return vol


ESTIMATORS = {
    "yang_zhang": yang_zhang,
    "close_to_close": close_to_close,
    "parkinson": parkinson,
    "rogers_satchell": rogers_satchell,
    "garman_klass": garman_klass,
}
