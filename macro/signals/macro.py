# signals/macro.py
# Macro signals — PMI, CPI surprise, CB bias, current account, unemployment

import pandas as pd
import numpy as np
from config.pairs import ALL_PAIRS


def split_pair(pair: str):
    return pair[:3], pair[3:]


# ── PMI composite differential ────────────────────────────────────────────────

def pmi_signal(diff: float) -> str:
    """Direction label based on PMI composite differential."""
    if pd.isna(diff):
        return "N/A"
    if diff > 2:
        return "Strong Bull"
    if diff > 0:
        return "Mild Bull"
    if diff < -2:
        return "Strong Bear"
    if diff < 0:
        return "Mild Bear"
    return "Neutral"


# ── CPI surprise ──────────────────────────────────────────────────────────────

def cpi_surprise_diff(
    cpi_actual: pd.Series,
    cpi_consensus: pd.Series,
) -> pd.Series:
    """Latest CPI actual vs consensus — requires Bloomberg EST field."""
    return (cpi_actual - cpi_consensus).rename("cpi_surprise")


# ── CB policy bias score ──────────────────────────────────────────────────────

# Manually maintained bias table — update after each meeting
# +1 = Hawkish, 0 = Neutral, -1 = Dovish
CB_BIAS = {
    "USD": 0,    # Fed — on hold, data dependent
    "EUR": -1,   # ECB — cutting cycle
    "GBP": -1,   # BoE — cutting
    "JPY": 1,    # BoJ — gradual hiking
    "CHF": -1,   # SNB — cutting
    "CAD": -1,   # BoC — cutting
    "AUD": 0,    # RBA — on hold
    "NZD": -1,   # RBNZ — cutting
    "NOK": 0,    # Norges Bank — on hold
    "SEK": -1,   # Riksbank — cutting
    # EM — update after each meeting
    "BRL": 1,    # BCB — hiking cycle
    "MXN": -1,   # Banxico — cutting
    "CLP": 0,    # BCCh — on hold after cuts
    "HUF": -1,   # MNB — cutting
    "PLN": 0,    # NBP — on hold
    "CZK": -1,   # CNB — cutting
    "ZAR": 0,    # SARB — on hold
    "KRW": -1,   # BoK — cutting
    "IDR": 0,    # BI — on hold
}

CB_BIAS_LABEL = {1: "Hawkish 🦅", 0: "Neutral ⚖️", -1: "Dovish 🕊️"}


# ── Momentum of macro data (MoM change) ──────────────────────────────────────

def latest_momentum(series: pd.Series, periods: int = 1) -> float:
    """Last n-period change in a macro series (e.g. PMI MoM)."""
    s = series.dropna()
    if len(s) < periods + 1:
        return np.nan
    return s.iloc[-1] - s.iloc[-(periods + 1)]


# ── Build macro signals ───────────────────────────────────────────────────────

def build_macro_signals(
    pmi: pd.Series,            # latest composite PMI per currency
    cpi_latest: pd.Series,     # latest CPI YoY per currency
    policy_rate: pd.Series,    # latest policy rate per currency
    pmi_hist: pd.DataFrame = None,   # monthly history for momentum
    cpi_hist: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Build macro differential signals for each G10 pair.
    All input Series indexed by currency code (USD, EUR, GBP, …).
    """
    records = []

    for pair in ALL_PAIRS:
        base, quote = split_pair(pair)

        # ── PMI differential ──────────────────────────────────────────────────
        pmi_b = pmi.get(base, np.nan)
        pmi_q = pmi.get(quote, np.nan)
        pmi_diff = (pmi_b - pmi_q) if not (pd.isna(pmi_b) or pd.isna(pmi_q)) else np.nan

        # ── PMI momentum (MoM change) ─────────────────────────────────────────
        pmi_mom_b = np.nan
        pmi_mom_q = np.nan
        if pmi_hist is not None and base in pmi_hist.columns:
            pmi_mom_b = latest_momentum(pmi_hist[base])
        if pmi_hist is not None and quote in pmi_hist.columns:
            pmi_mom_q = latest_momentum(pmi_hist[quote])
        pmi_mom_diff = (
            (pmi_mom_b - pmi_mom_q)
            if not (pd.isna(pmi_mom_b) or pd.isna(pmi_mom_q))
            else np.nan
        )

        # ── CPI differential ──────────────────────────────────────────────────
        cpi_b = cpi_latest.get(base, np.nan)
        cpi_q = cpi_latest.get(quote, np.nan)
        cpi_diff = (cpi_b - cpi_q) if not (pd.isna(cpi_b) or pd.isna(cpi_q)) else np.nan

        # ── CPI momentum ─────────────────────────────────────────────────────
        cpi_mom_b, cpi_mom_q = np.nan, np.nan
        if cpi_hist is not None and base in cpi_hist.columns:
            cpi_mom_b = latest_momentum(cpi_hist[base])
        if cpi_hist is not None and quote in cpi_hist.columns:
            cpi_mom_q = latest_momentum(cpi_hist[quote])
        cpi_mom_diff = (
            (cpi_mom_b - cpi_mom_q)
            if not (pd.isna(cpi_mom_b) or pd.isna(cpi_mom_q))
            else np.nan
        )

        # ── CB policy bias differential ───────────────────────────────────────
        cb_b = CB_BIAS.get(base, 0)
        cb_q = CB_BIAS.get(quote, 0)
        cb_diff = cb_b - cb_q
        cb_label = (
            CB_BIAS_LABEL.get(cb_b, "N/A") + " vs " + CB_BIAS_LABEL.get(cb_q, "N/A")
        )

        # ── Policy rate differential ───────────────────────────────────────────
        pr_b = policy_rate.get(base, np.nan)
        pr_q = policy_rate.get(quote, np.nan)
        policy_diff_bps = (
            (pr_b - pr_q) * 100
            if not (pd.isna(pr_b) or pd.isna(pr_q))
            else np.nan
        )

        records.append(
            {
                "pair": pair,
                # PMI
                "pmi_base": round(pmi_b, 1) if not pd.isna(pmi_b) else np.nan,
                "pmi_quote": round(pmi_q, 1) if not pd.isna(pmi_q) else np.nan,
                "pmi_diff": round(pmi_diff, 1) if not pd.isna(pmi_diff) else np.nan,
                "pmi_signal": pmi_signal(pmi_diff),
                "pmi_mom_diff": round(pmi_mom_diff, 2) if not pd.isna(pmi_mom_diff) else np.nan,
                # CPI
                "cpi_base_yoy": round(cpi_b, 2) if not pd.isna(cpi_b) else np.nan,
                "cpi_quote_yoy": round(cpi_q, 2) if not pd.isna(cpi_q) else np.nan,
                "cpi_diff_pct": round(cpi_diff, 2) if not pd.isna(cpi_diff) else np.nan,
                "cpi_mom_diff": round(cpi_mom_diff, 2) if not pd.isna(cpi_mom_diff) else np.nan,
                # CB
                "cb_bias_base": CB_BIAS_LABEL.get(cb_b, "N/A"),
                "cb_bias_quote": CB_BIAS_LABEL.get(cb_q, "N/A"),
                "cb_diff_score": cb_diff,
                "cb_label": cb_label,
                # Policy rate diff
                "policy_rate_diff_bps": (
                    round(policy_diff_bps, 1) if not pd.isna(policy_diff_bps) else np.nan
                ),
            }
        )

    return pd.DataFrame(records).set_index("pair")
