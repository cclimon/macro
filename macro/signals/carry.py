# signals/carry.py
# Carry signals — rate differentials, carry/vol, forward implied carry, real carry

import pandas as pd
import numpy as np
from typing import Dict
from config.pairs import G10_PAIRS, LOOKBACK


# ── Utility: extract base/quote currencies from pair string ───────────────────

def split_pair(pair: str):
    return pair[:3], pair[3:]


# ── Realised Volatility ───────────────────────────────────────────────────────

def realised_vol(prices: pd.Series, window: int, annualise: int = 252) -> float:
    log_ret = np.log(prices / prices.shift(1)).dropna()
    if len(log_ret) < window:
        return np.nan
    return log_ret.iloc[-window:].std() * np.sqrt(annualise) * 100   # in %


# ── Carry / Vol ratio ─────────────────────────────────────────────────────────

def carry_vol_ratio(carry_bps: float, vol_pct: float) -> float:
    """Risk-adjusted carry: carry (annualised bps) / realised vol (%)."""
    if pd.isna(vol_pct) or vol_pct == 0 or pd.isna(carry_bps):
        return np.nan
    return round(carry_bps / (vol_pct * 100), 4)   # bps / bps


# ── Vol regime label ──────────────────────────────────────────────────────────

def vol_regime(current_vol: float, hist_vol_series: pd.Series) -> str:
    """Classify current vol vs its 6m rolling average."""
    if pd.isna(current_vol) or len(hist_vol_series) < 21:
        return "N/A"
    avg_6m = hist_vol_series.rolling(126).mean().iloc[-1]
    if pd.isna(avg_6m):
        return "N/A"
    ratio = current_vol / avg_6m
    if ratio > 1.3:
        return "High"
    if ratio < 0.7:
        return "Low"
    return "Normal"


def rolling_vol_series(prices: pd.Series, window: int = 21) -> pd.Series:
    """Rolling annualised vol series (%) for regime detection."""
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252) * 100


# ── Forward-implied carry ─────────────────────────────────────────────────────

def forward_implied_carry(spot: float, fwd: float, tenor_months: int = 3) -> float:
    """
    Annualised carry implied by forward vs spot.
    carry (%) = (fwd/spot - 1) * (12 / tenor_months) * 100
    Positive = base currency at premium (earns carry if long base).
    """
    if pd.isna(spot) or pd.isna(fwd) or spot == 0:
        return np.nan
    return ((fwd / spot) - 1) * (12 / tenor_months) * 100


# ── Build carry signals ────────────────────────────────────────────────────────

def build_carry_signals(
    spot_df: pd.DataFrame,
    rates_3m: pd.Series,           # latest 3m rate per currency (%)
    cpi_latest: pd.Series,         # latest CPI YoY per currency (%)
    fwd_latest: pd.Series,         # latest 3m forward per pair (spot units)
    spot_latest: pd.Series,        # latest spot per pair
) -> pd.DataFrame:
    """
    Build all carry signals for each G10 pair.

    Parameters
    ----------
    spot_df     : historical spot prices (columns = pair names)
    rates_3m    : latest 3m rate, indexed by currency code (USD, EUR, …)
    cpi_latest  : latest CPI YoY, indexed by currency code
    fwd_latest  : latest 3m fwd price, indexed by pair name
    spot_latest : latest spot price, indexed by pair name
    """
    records = []

    for pair in G10_PAIRS:
        if pair not in spot_df.columns:
            continue

        base, quote = split_pair(pair)
        px = spot_df[pair].dropna()

        # ── 3m nominal rate differential ──────────────────────────────────────
        r_base = rates_3m.get(base, np.nan)
        r_quote = rates_3m.get(quote, np.nan)
        rate_diff_bps = (r_base - r_quote) * 100 if not (
            pd.isna(r_base) or pd.isna(r_quote)
        ) else np.nan
        # Positive = long base earns carry

        # ── CPI / real carry ──────────────────────────────────────────────────
        cpi_base = cpi_latest.get(base, np.nan)
        cpi_quote = cpi_latest.get(quote, np.nan)
        cpi_diff = (cpi_base - cpi_quote) if not (
            pd.isna(cpi_base) or pd.isna(cpi_quote)
        ) else np.nan
        real_carry_bps = (rate_diff_bps - cpi_diff * 100) if not (
            pd.isna(rate_diff_bps) or pd.isna(cpi_diff)
        ) else np.nan

        # ── Realised vol ──────────────────────────────────────────────────────
        rvol_1m = realised_vol(px, LOOKBACK["1m"])
        rvol_3m = realised_vol(px, LOOKBACK["3m"])

        # ── Carry / Vol ───────────────────────────────────────────────────────
        carry_vol_1m = carry_vol_ratio(rate_diff_bps, rvol_1m)
        carry_vol_3m = carry_vol_ratio(rate_diff_bps, rvol_3m)

        # ── Vol regime ────────────────────────────────────────────────────────
        rv_series = rolling_vol_series(px)
        regime = vol_regime(rvol_1m, rv_series)

        # ── Forward implied carry ─────────────────────────────────────────────
        spot_px = spot_latest.get(pair, np.nan)
        fwd_px = fwd_latest.get(pair, np.nan)
        fwd_carry = forward_implied_carry(spot_px, fwd_px, 3)

        records.append(
            {
                "pair": pair,
                "rate_base_3m": round(r_base, 3) if not pd.isna(r_base) else np.nan,
                "rate_quote_3m": round(r_quote, 3) if not pd.isna(r_quote) else np.nan,
                "rate_diff_bps": round(rate_diff_bps, 1) if not pd.isna(rate_diff_bps) else np.nan,
                "cpi_diff_pct": round(cpi_diff, 2) if not pd.isna(cpi_diff) else np.nan,
                "real_carry_bps": round(real_carry_bps, 1) if not pd.isna(real_carry_bps) else np.nan,
                "rvol_1m_pct": round(rvol_1m, 2) if not pd.isna(rvol_1m) else np.nan,
                "rvol_3m_pct": round(rvol_3m, 2) if not pd.isna(rvol_3m) else np.nan,
                "carry_vol_1m": round(carry_vol_1m, 3) if not pd.isna(carry_vol_1m) else np.nan,
                "carry_vol_3m": round(carry_vol_3m, 3) if not pd.isna(carry_vol_3m) else np.nan,
                "vol_regime": regime,
                "fwd_carry_ann_pct": round(fwd_carry, 3) if not pd.isna(fwd_carry) else np.nan,
            }
        )

    return pd.DataFrame(records).set_index("pair")
