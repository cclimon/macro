"""
fx_vol_bands.py
===============
Computes 1-week implied vol bands for FX pairs, adds Bollinger Band width as a
regime filter, and flags potential gamma squeeze setups.

Data sources (in priority order):
  1. Bloomberg BLPAPI  – requires a running Bloomberg terminal / B-PIPE session
  2. CSV file          – columns: date, spot, implied_vol_1w, risk_reversal_25d
  3. Built-in sample data generator (for testing / offline use)

Usage examples
--------------
# Bloomberg pull (1 year of EURUSD data):
    python fx_vol_bands.py --blp --ticker EURUSD

# CSV file:
    python fx_vol_bands.py --csv my_data.csv

# Offline sample data (testing):
    python fx_vol_bands.py --sample --pair EURUSD
"""

import argparse
import sys
import os

# Force UTF-8 output on Windows so box-drawing / arrow chars print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless-safe; switch to "TkAgg" for interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Bloomberg availability guard ──────────────────────────────────────────────
try:
    import blpapi
    BLPAPI_AVAILABLE = True
except ImportError:
    BLPAPI_AVAILABLE = False


# =============================================================================
# 1. DATA INGESTION
# =============================================================================

def generate_sample_data(
    pair: str = "EURUSD",
    n: int = 252,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic FX dataset for offline testing.

    Spot follows a GBM with a mild trending regime.
    Implied vol cycles between low and high regimes to ensure squeeze flags fire.
    """
    rng = np.random.default_rng(seed)
    end   = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    dates = pd.bdate_range(end=end, periods=n)

    # Spot – GBM
    S0    = 1.0850 if pair.upper().startswith("EUR") else 1.2700
    mu    = 0.0
    sigma = 0.065   # ~6.5% annual RV
    dt    = 1 / 252
    returns = rng.normal(mu * dt, sigma * np.sqrt(dt), n)
    spot  = S0 * np.exp(np.cumsum(returns))

    # Implied vol – oscillate to create squeeze-prone periods
    base_iv = 0.070
    iv_cycle = base_iv + 0.010 * np.sin(np.linspace(0, 4 * np.pi, n))
    iv_noise = rng.normal(0, 0.002, n)
    implied_vol_1w = np.clip(iv_cycle + iv_noise, 0.03, 0.25)

    # 25-delta risk reversal – slightly skewed in direction of move
    rr = 0.001 * returns / (sigma * np.sqrt(dt) + 1e-9)   # directional proxy
    rr = rr + rng.normal(0, 0.05, n)

    df = pd.DataFrame({
        "date":              dates,
        "spot":              spot,
        "implied_vol_1w":    implied_vol_1w,
        "risk_reversal_25d": rr,
    })
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def load_csv(path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Required column : date, spot
    Optional columns: implied_vol_1w, risk_reversal_25d
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    df.index = pd.to_datetime(df.index)

    if "spot" not in df.columns:
        raise ValueError("CSV must contain a 'spot' column.")

    # Fill optional columns with NaN so downstream code handles gracefully
    for col in ("implied_vol_1w", "risk_reversal_25d"):
        if col not in df.columns:
            df[col] = np.nan

    return df


def fetch_bloomberg_data(
    ticker: str,
    start_date: str | None = None,
    end_date:   str | None = None,
    iv_constant: float | None = None,
    rr_constant: float | None = None,
) -> pd.DataFrame:
    """
    Pull spot, 1W implied vol and 25d risk reversal from Bloomberg via BDH.

    ticker examples: "EURUSD", "USDJPY", "GBPUSD"

    Bloomberg field mapping
    -----------------------
    Spot              : <PAIR> Curncy  – PX_LAST
    1W implied vol    : <PAIR>V1W Curncy – PX_LAST  (e.g. EURUSDV1W)
    25d RR (1W)       : <PAIR>25R1W Curncy – PX_LAST (e.g. EURUSD25R1W)
    """
    if not BLPAPI_AVAILABLE:
        raise ImportError(
            "blpapi is not installed. Install it from the Bloomberg API SDK or "
            "run with --csv / --sample instead."
        )

    if end_date is None:
        end_date   = datetime.today().strftime("%Y%m%d")
    if start_date is None:
        start_date = (datetime.today() - timedelta(days=365)).strftime("%Y%m%d")

    pair = ticker.upper().replace(" ", "")

    # Build Bloomberg security / field lists
    spot_sec = f"{pair} Curncy"
    iv_sec   = f"{pair}V1W Curncy"
    rr_sec   = f"{pair}25R1W Curncy"

    session_options = blpapi.SessionOptions()
    session_options.setServerHost("localhost")
    session_options.setServerPort(8194)

    session = blpapi.Session(session_options)
    if not session.start():
        raise ConnectionError("Failed to start Bloomberg session. Check terminal is running.")
    if not session.openService("//blp/refdata"):
        raise ConnectionError("Failed to open //blp/refdata service.")

    refdata_service = session.getService("//blp/refdata")

    def _bdh(security: str, field: str) -> pd.Series:
        """Fire a single BDH request and return a named Series indexed by date."""
        request = refdata_service.createRequest("HistoricalDataRequest")
        request.getElement("securities").appendValue(security)
        request.getElement("fields").appendValue(field)
        request.set("startDate", start_date)
        request.set("endDate",   end_date)
        request.set("periodicityAdjustment", "ACTUAL")
        request.set("periodicitySelection",  "DAILY")

        session.sendRequest(request)
        data = {}
        while True:
            event = session.nextEvent(500)
            for msg in event:
                if msg.hasElement("securityData"):
                    sec_data      = msg.getElement("securityData")
                    field_data    = sec_data.getElement("fieldData")
                    for i in range(field_data.numValues()):
                        point = field_data.getValue(i)
                        raw_date = point.getElementAsDatetime("date")
                        # blpapi may return datetime.date or datetime.datetime
                        if hasattr(raw_date, "date"):
                            raw_date = raw_date.date()
                        val = point.getElementAsFloat(field)
                        data[pd.Timestamp(raw_date)] = val
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        return pd.Series(data, name=security)

    spot_series = _bdh(spot_sec, "PX_LAST")

    # IV from Bloomberg is in %, convert to decimal
    if iv_constant is not None:
        iv_series = pd.Series(iv_constant, index=spot_series.index, name="implied_vol_1w")
    else:
        iv_raw    = _bdh(iv_sec, "PX_LAST")
        iv_series = (iv_raw / 100.0).rename("implied_vol_1w")

    # RR from Bloomberg is in % (vol points), convert to decimal
    if rr_constant is not None:
        rr_series = pd.Series(rr_constant, index=spot_series.index, name="risk_reversal_25d")
    else:
        rr_raw    = _bdh(rr_sec, "PX_LAST")
        rr_series = (rr_raw / 100.0).rename("risk_reversal_25d")

    session.stop()

    df = pd.concat(
        [spot_series.rename("spot"), iv_series, rr_series],
        axis=1,
    ).dropna(subset=["spot"])
    df.index.name = "date"
    df.sort_index(inplace=True)
    return df


# =============================================================================
# 2. IMPLIED VOL BANDS
# =============================================================================

def compute_vol_bands(df: pd.DataFrame, iv_constant: float | None = None) -> pd.DataFrame:
    """
    Compute ±1σ, ±1.5σ, ±2σ 1-week implied vol bands.

    Formula: band = spot * exp(±k * σ_1w * sqrt(7/365))

    Also computes:
    - sigma_consumed : spot displacement expressed in σ units (signed)
    - abs_sigma      : absolute σ consumed
    """
    iv = df["implied_vol_1w"].copy()
    if iv_constant is not None:
        iv = iv.fillna(iv_constant)

    T = np.sqrt(7 / 365)

    df = df.copy()
    df["iv_used"] = iv

    for k in (1.0, 1.5, 2.0):
        label = str(k).replace(".", "")   # "10", "15", "20"
        df[f"band_up_{label}"]  = df["spot"] * np.exp( k * iv * T)
        df[f"band_dn_{label}"]  = df["spot"] * np.exp(-k * iv * T)

    # σ consumed: compare today's spot to yesterday's implied bands
    # (forward-looking bands are centred on today's spot by definition, so we
    #  use the previous day's centre-spot as the anchor)
    spot_lag   = df["spot"].shift(1)
    iv_lag     = df["iv_used"].shift(1)
    one_sigma  = spot_lag * (np.exp(iv_lag * T) - 1)   # ≈ spot_lag * iv * T

    df["sigma_consumed"] = (df["spot"] - spot_lag) / one_sigma.replace(0, np.nan)
    df["abs_sigma"]      = df["sigma_consumed"].abs()

    return df


# =============================================================================
# 3. BOLLINGER BANDS + BANDWIDTH REGIME
# =============================================================================

def compute_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    n_std: float = 2.0,
    pct_threshold: float = 20.0,
) -> pd.DataFrame:
    """
    Standard 20-period Bollinger Bands on spot.

    Adds:
    - bb_mid, bb_upper, bb_lower
    - bb_bandwidth : (upper - lower) / mid
    - bb_bandwidth_pct : rolling percentile rank of bandwidth in its own history
    - bb_squeeze : True when bandwidth is in the lowest <pct_threshold>th percentile
    """
    df = df.copy()
    roll = df["spot"].rolling(window)

    df["bb_mid"]    = roll.mean()
    std             = roll.std(ddof=1)
    df["bb_upper"]  = df["bb_mid"] + n_std * std
    df["bb_lower"]  = df["bb_mid"] - n_std * std
    df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # Percentile rank within all available history at each point
    def _pct_rank(series: pd.Series) -> pd.Series:
        return series.expanding().rank(pct=True) * 100

    df["bb_bandwidth_pct"] = _pct_rank(df["bb_bandwidth"])
    df["bb_squeeze"]       = df["bb_bandwidth_pct"] <= pct_threshold

    return df


# =============================================================================
# 4. IV / RV RATIO
# =============================================================================

def compute_iv_rv_ratio(
    df: pd.DataFrame,
    rv_window: int = 20,
    iv_rv_threshold: float = 1.2,
) -> pd.DataFrame:
    """
    Compute 20-day close-to-close realised vol (annualised) and IV/RV ratio.

    Adds:
    - log_return
    - rv_20d      : 20-day realised vol (annualised, decimal)
    - iv_rv_ratio : implied_vol_1w / rv_20d
    - iv_rv_flag  : True when ratio > iv_rv_threshold
    """
    df = df.copy()
    df["log_return"] = np.log(df["spot"] / df["spot"].shift(1))

    df["rv_20d"] = (
        df["log_return"]
          .rolling(rv_window)
          .std(ddof=1)
        * np.sqrt(252)
    )

    df["iv_rv_ratio"] = df["iv_used"] / df["rv_20d"].replace(0, np.nan)
    df["iv_rv_flag"]  = df["iv_rv_ratio"] > iv_rv_threshold

    return df


# =============================================================================
# 5. GAMMA SQUEEZE FLAGS
# =============================================================================

def compute_squeeze_flags(
    df: pd.DataFrame,
    sigma_threshold: float = 1.2,
    iv_rv_threshold: float = 1.2,
) -> pd.DataFrame:
    """
    Set squeeze_flag = True when ALL of the following are met:

      1. abs(sigma_consumed) > sigma_threshold   (spot outside 1.2σ)
      2. bb_squeeze == True                       (BB bandwidth in bottom 20th pct)
      3. iv_rv_flag == True                       (IV/RV > 1.2)
      4. If risk_reversal_25d is available:
         RR sign matches direction of spot displacement
         (positive RR -> calls expensive -> spot rallying = consistent)

    Also sets:
    - squeeze_direction : "up" / "down" / NaN
    """
    df = df.copy()

    cond_sigma = df["abs_sigma"] > sigma_threshold
    cond_bb    = df["bb_squeeze"].fillna(False)
    cond_iv_rv = df["iv_rv_flag"].fillna(False)

    has_rr = df["risk_reversal_25d"].notna().any()
    if has_rr:
        # RR > 0 means calls bid -> spot squeeze candidate is upward
        # RR < 0 means puts bid  -> spot squeeze candidate is downward
        direction = np.sign(df["sigma_consumed"])   # +1 up, -1 down
        rr_sign   = np.sign(df["risk_reversal_25d"])
        cond_rr   = direction == rr_sign
    else:
        cond_rr = pd.Series(True, index=df.index)

    df["squeeze_flag"] = cond_sigma & cond_bb & cond_iv_rv & cond_rr

    df["squeeze_direction"] = np.where(
        df["squeeze_flag"] & (df["sigma_consumed"] > 0), "up",
        np.where(
            df["squeeze_flag"] & (df["sigma_consumed"] < 0), "down",
            None,
        ),
    )

    return df


# =============================================================================
# 6. OUTPUT – CSV
# =============================================================================

def save_results(df: pd.DataFrame, out_path: str) -> None:
    """Save the full enriched DataFrame to CSV."""
    df.to_csv(out_path)
    print(f"[OUTPUT] Results saved -> {out_path}")


# =============================================================================
# 6. OUTPUT – CHART
# =============================================================================

def _build_summary_panel(ax: plt.Axes, df: pd.DataFrame, pair: str) -> None:
    """
    Render the bottom summary panel: left = panel descriptions, right = conditions table.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    last = df.iloc[-1]
    n_squeeze = int(df["squeeze_flag"].sum())
    as_of = df.index[-1].strftime("%d %b %Y")

    # ── Left column : panel descriptions ─────────────────────────────────────
    descriptions = [
        ("Panel 1 – Spot & Implied Vol Bands",
         "Spot price overlaid with ±1\u03c3 / ±1.5\u03c3 / ±2\u03c3 1-week implied vol bands centred on the prior close\n"
         "(formula: S\u2099\u208b\u2081 \u00d7 exp(\u00b1k \u00d7 \u03c3\u2081w \u00d7 \u221a(7/365))). Purple dashed lines are 20-period Bollinger Bands.\n"
         "Red dots mark gamma squeeze flags. Green/orange/red shading widens when IV is elevated."),
        ("Panel 2 – Bollinger Band Bandwidth",
         "BB bandwidth = (upper \u2212 lower) / mid, expressed as % of spot. A falling line signals realised\n"
         "vol contraction. Pink shading = bandwidth in the lowest 20th percentile of its own history\n"
         "(squeeze regime). Expansions following these contractions often produce sharp directional moves."),
        ("Panel 3 – IV / RV Ratio",
         "1-week implied vol divided by 20-day close-to-close realised vol (annualised). A ratio above 1.2\u00d7\n"
         "(pink shading) means the market is paying a significant premium for options vs recent realised.\n"
         "When combined with BB contraction and spot displacement, this signals elevated squeeze risk."),
    ]

    y_positions = [0.95, 0.60, 0.25]
    for (title, body), y in zip(descriptions, y_positions):
        ax.text(0.01, y, title, fontsize=8, fontweight="bold", va="top",
                transform=ax.transAxes, color="#1a1a2e")
        ax.text(0.01, y - 0.05, body, fontsize=7, va="top",
                transform=ax.transAxes, color="#333333",
                linespacing=1.5)

    # Vertical divider
    ax.axvline(0.52, color="#cccccc", lw=1.0)

    # ── Right column : conditions table ──────────────────────────────────────
    def _val(col, fmt=None, pct=False):
        v = last.get(col, np.nan)
        if pd.isna(v):
            return "n/a"
        if pct:
            return f"{v:.2%}"
        if fmt:
            return fmt.format(v)
        return str(v)

    def _met(flag_col):
        v = last.get(flag_col, False)
        if pd.isna(v):
            return "n/a", "#888888"
        return ("\u2713 MET", "#2ca02c") if bool(v) else ("\u2717 NOT MET", "#d62728")

    has_rr = df["risk_reversal_25d"].notna().any()

    # Squeeze condition rows
    sigma_val   = _val("abs_sigma",        "{:.2f}\u03c3")
    sigma_flag  = bool(last.get("abs_sigma", 0) > 1.2) if pd.notna(last.get("abs_sigma")) else False
    sigma_mark, sigma_col = ("\u2713 MET", "#2ca02c") if sigma_flag else ("\u2717 NOT MET", "#d62728")

    bb_mark,    bb_col    = _met("bb_squeeze")
    ivrv_mark,  ivrv_col  = _met("iv_rv_flag")

    if has_rr:
        direction = np.sign(last.get("sigma_consumed", 0))
        rr_sign   = np.sign(last.get("risk_reversal_25d", 0))
        rr_aligned = (direction == rr_sign) and (direction != 0)
        rr_val    = _val("risk_reversal_25d", "{:+.3f}")
        rr_mark, rr_col = ("\u2713 MET", "#2ca02c") if rr_aligned else ("\u2717 NOT MET", "#d62728")
    else:
        rr_val, rr_mark, rr_col = "n/a", "N/A", "#888888"

    squeeze_on    = bool(last.get("squeeze_flag", False))
    sq_mark, sq_col = ("\u2713 ACTIVE", "#d62728") if squeeze_on else ("\u2717 INACTIVE", "#555555")

    # Header
    hx = 0.54
    ax.text(hx, 0.97,
            f"Gamma Squeeze Conditions  –  {pair}  (as of {as_of})",
            fontsize=8.5, fontweight="bold", va="top", transform=ax.transAxes, color="#1a1a2e")
    ax.text(hx, 0.90,
            f"Squeeze flags in dataset: {n_squeeze}  |  "
            f"Direction: {last.get('squeeze_direction') or 'none'}",
            fontsize=7.5, va="top", transform=ax.transAxes, color="#555555")

    # Table data
    col_headers = ["Condition", "Value", "Threshold", "Status"]
    rows = [
        ["\u03c3 consumed (abs)",
         sigma_val,
         "> 1.2\u03c3",
         (sigma_mark, sigma_col)],
        ["BB bandwidth %ile",
         f"{_val('bb_bandwidth_pct', '{:.1f}th')}",
         "\u2264 20th pct",
         (bb_mark, bb_col)],
        ["IV / RV ratio",
         _val("iv_rv_ratio", "{:.2f}\u00d7"),
         "> 1.2\u00d7",
         (ivrv_mark, ivrv_col)],
        ["RR direction aligned",
         rr_val,
         "RR sign = spot dir",
         (rr_mark, rr_col)],
    ]

    # Supplementary metrics
    metrics = [
        ("Spot",          _val("spot",    "{:.5f}")),
        ("IV 1W",         _val("iv_used", None, pct=True)),
        ("RV 20d",        _val("rv_20d",  None, pct=True)),
        ("BB width",      f"{_val('bb_bandwidth', '{:.4f}')}"),
        ("\u03c3 consumed (signed)", _val("sigma_consumed", "{:+.2f}\u03c3")),
    ]

    # Draw condition rows
    row_h   = 0.10
    y_start = 0.80
    col_x   = [hx, hx + 0.17, hx + 0.30, hx + 0.41]

    # Column headers
    for cx, ch in zip(col_x, col_headers):
        ax.text(cx, y_start, ch, fontsize=7, fontweight="bold", va="top",
                transform=ax.transAxes, color="#1a1a2e")

    # Separator line
    ax.annotate("", xy=(1.0, y_start - 0.025), xytext=(hx, y_start - 0.025),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.8))

    for i, (cond, val, thresh, (mark, mcol)) in enumerate(rows):
        y = y_start - 0.035 - i * row_h
        bg_col = "#fff8f8" if mcol == "#d62728" else "#f8fff8" if mcol == "#2ca02c" else "#f8f8f8"
        rect = plt.Rectangle((hx - 0.005, y - 0.065), 0.465, row_h - 0.01,
                              transform=ax.transAxes, color=bg_col, zorder=0, clip_on=False)
        ax.add_patch(rect)
        for cx, txt in zip(col_x, [cond, val, thresh]):
            ax.text(cx, y, txt, fontsize=7.5, va="top", transform=ax.transAxes, color="#222222")
        ax.text(col_x[3], y, mark, fontsize=7.5, fontweight="bold", va="top",
                transform=ax.transAxes, color=mcol)

    # Squeeze flag summary row
    y_sq = y_start - 0.035 - 4 * row_h
    ax.annotate("", xy=(1.0, y_sq + 0.02), xytext=(hx, y_sq + 0.02),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8))
    ax.text(hx, y_sq, "SQUEEZE FLAG (ALL conditions)", fontsize=8,
            fontweight="bold", va="top", transform=ax.transAxes, color="#1a1a2e")
    ax.text(col_x[3], y_sq, sq_mark, fontsize=8, fontweight="bold", va="top",
            transform=ax.transAxes, color=sq_col)

    # Supplementary metrics (small, below the table)
    y_met = y_sq - 0.14
    ax.text(hx, y_met, "Key metrics:", fontsize=7, fontweight="bold", va="top",
            transform=ax.transAxes, color="#555555")
    for j, (mlabel, mval) in enumerate(metrics):
        col_idx = j % 3
        row_idx = j // 3
        mx = hx + col_idx * 0.155
        my = y_met - 0.07 - row_idx * 0.08
        ax.text(mx, my, f"{mlabel}: {mval}", fontsize=7, va="top",
                transform=ax.transAxes, color="#333333")


def plot_results(
    df: pd.DataFrame,
    pair: str = "FX",
    out_path: str = "fx_vol_bands_chart.png",
) -> None:
    """
    Four-panel chart:
      Panel 1 : Spot + implied vol bands (±1σ, ±1.5σ, ±2σ) + BB bands
                Gamma squeeze flags marked as red dots
      Panel 2 : BB bandwidth with 20th-percentile threshold line
      Panel 3 : IV/RV ratio with 1.2 threshold line
      Panel 4 : Panel descriptions + squeeze conditions table (latest bar)
    """
    fig = plt.figure(figsize=(18, 16))
    gs  = fig.add_gridspec(
        4, 1,
        height_ratios=[4, 1.5, 1.5, 2.8],
        hspace=0.35,
    )
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3])

    fig.suptitle(
        f"{pair}  –  Implied Vol Bands & Gamma Squeeze Monitor",
        fontsize=14, fontweight="bold", y=0.99,
    )

    dates = df.index

    # ── Panel 1 : Spot + bands ────────────────────────────────────────────────
    band_configs = [
        ("band_up_20", "band_dn_20", "#d62728", 0.08, "\u00b12\u03c3 IV band"),
        ("band_up_15", "band_dn_15", "#ff7f0e", 0.12, "\u00b11.5\u03c3 IV band"),
        ("band_up_10", "band_dn_10", "#2ca02c", 0.18, "\u00b11\u03c3 IV band"),
    ]
    for up_col, dn_col, colour, alpha, label in band_configs:
        if up_col in df.columns and dn_col in df.columns:
            ax1.fill_between(dates, df[dn_col], df[up_col],
                             alpha=alpha, color=colour, label=label)

    if "bb_upper" in df.columns:
        ax1.plot(dates, df["bb_upper"], "--", color="#9467bd", lw=1.0, label="BB upper/lower")
        ax1.plot(dates, df["bb_lower"], "--", color="#9467bd", lw=1.0)
        ax1.plot(dates, df["bb_mid"],   ":",  color="#9467bd", lw=0.8, label="BB mid")

    ax1.plot(dates, df["spot"], color="black", lw=1.4, label="Spot", zorder=5)

    squeeze_rows = df[df["squeeze_flag"]]
    if not squeeze_rows.empty:
        ax1.scatter(
            squeeze_rows.index,
            squeeze_rows["spot"],
            color="red", s=60, zorder=10, label=f"Squeeze flag ({len(squeeze_rows)})",
        )

    ax1.set_ylabel("Spot", fontsize=10)
    ax1.legend(loc="upper left", fontsize=7, ncol=3)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Panel 2 : BB Bandwidth ────────────────────────────────────────────────
    if "bb_bandwidth" in df.columns:
        ax2.plot(dates, df["bb_bandwidth"] * 100, color="#9467bd", lw=1.0, label="BB bandwidth (%)")
        threshold_val = np.nanpercentile(df["bb_bandwidth"].dropna(), 20) * 100
        ax2.axhline(threshold_val, color="red", lw=0.9, linestyle="--",
                    label=f"20th pct = {threshold_val:.2f}%")
        squeeze_mask = df["bb_squeeze"].fillna(False)
        ax2.fill_between(dates, 0,
                         np.where(squeeze_mask, df["bb_bandwidth"] * 100, np.nan),
                         color="red", alpha=0.25, label="BB squeeze zone")

    ax2.set_ylabel("BB Width (%)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=7)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # ── Panel 3 : IV/RV Ratio ─────────────────────────────────────────────────
    if "iv_rv_ratio" in df.columns:
        ax3.plot(dates, df["iv_rv_ratio"], color="#1f77b4", lw=1.0, label="IV/RV ratio")
        ax3.axhline(1.2, color="red", lw=0.9, linestyle="--", label="1.2\u00d7 threshold")
        ax3.axhline(1.0, color="grey", lw=0.6, linestyle=":")
        flag_mask = df["iv_rv_flag"].fillna(False)
        ax3.fill_between(dates, 1.0,
                         np.where(flag_mask, df["iv_rv_ratio"].clip(upper=3.0), np.nan),
                         color="red", alpha=0.2, label="IV/RV > 1.2")

    ax3.set_ylabel("IV/RV", fontsize=10)
    ax3.legend(loc="upper right", fontsize=7)
    ax3.grid(axis="y", linestyle=":", alpha=0.4)
    ax3.set_xlabel("Date", fontsize=10)

    # ── Panel 4 : Summary ─────────────────────────────────────────────────────
    _build_summary_panel(ax4, df, pair)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OUTPUT] Chart saved -> {out_path}")


# =============================================================================
# 6. SUMMARY PRINT
# =============================================================================

def print_summary(df: pd.DataFrame, pair: str = "FX") -> None:
    """Print a human-readable summary of squeeze flags to stdout."""
    squeeze_rows = df[df["squeeze_flag"]].copy()

    print("\n" + "=" * 72)
    print(f"  GAMMA SQUEEZE SUMMARY – {pair}")
    print("=" * 72)
    print(f"  Period        : {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  Total bars    : {len(df)}")
    print(f"  Squeeze flags : {len(squeeze_rows)}")

    if squeeze_rows.empty:
        print("  No squeeze flags identified in this dataset.")
    else:
        cols_to_show = [
            c for c in
            ["spot", "iv_used", "rv_20d", "iv_rv_ratio",
             "abs_sigma", "bb_bandwidth_pct", "squeeze_direction"]
            if c in squeeze_rows.columns
        ]
        squeeze_rows_display = squeeze_rows[cols_to_show].copy()

        # Format floats
        float_fmt = {
            "spot":             "{:.5f}",
            "iv_used":          "{:.2%}",
            "rv_20d":           "{:.2%}",
            "iv_rv_ratio":      "{:.2f}",
            "abs_sigma":        "{:.2f}",
            "bb_bandwidth_pct": "{:.1f}",
        }
        for col, fmt in float_fmt.items():
            if col in squeeze_rows_display.columns:
                squeeze_rows_display[col] = squeeze_rows_display[col].apply(
                    lambda x: fmt.format(x) if pd.notna(x) else "n/a"
                )

        print()
        print(squeeze_rows_display.to_string())

    # Most recent bar snapshot
    last = df.iloc[-1]
    print("\n  ── Latest bar snapshot ──────────────────────────────────────────")
    snapshot_fields = {
        "Spot":              ("spot",             "{:.5f}"),
        "IV (1W)":           ("iv_used",          "{:.2%}"),
        "RV (20d)":          ("rv_20d",           "{:.2%}"),
        "IV/RV":             ("iv_rv_ratio",      "{:.2f}x"),
        "σ consumed":        ("sigma_consumed",   "{:+.2f}σ"),
        "BB width %ile":     ("bb_bandwidth_pct", "{:.1f}th"),
        "BB squeeze":        ("bb_squeeze",       "{}"),
        "Squeeze flag":      ("squeeze_flag",     "{}"),
    }
    for label, (col, fmt) in snapshot_fields.items():
        if col in last.index and pd.notna(last[col]):
            print(f"  {label:<20}: {fmt.format(last[col])}")
    print("=" * 72 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FX 1-week implied vol bands + gamma squeeze detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Data source ───────────────────────────────────────────────────────────
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--blp",    action="store_true", help="Fetch from Bloomberg (requires BLPAPI)")
    source.add_argument("--csv",    metavar="FILE",       help="Load from CSV file")
    source.add_argument("--sample", action="store_true", help="Generate synthetic sample data")

    # ── Bloomberg options ─────────────────────────────────────────────────────
    parser.add_argument("--ticker",     default="EURUSD",  help="FX pair ticker (e.g. EURUSD, USDJPY)")
    parser.add_argument("--start-date", default=None,      help="Start date YYYYMMDD (Bloomberg)")
    parser.add_argument("--end-date",   default=None,      help="End date YYYYMMDD (Bloomberg)")

    # ── Override / constants ──────────────────────────────────────────────────
    parser.add_argument("--iv-constant",  type=float, default=None,
                        help="Override IV with a fixed value (decimal, e.g. 0.07 = 7%%)")
    parser.add_argument("--rr-constant",  type=float, default=None,
                        help="Override RR with a fixed value (decimal)")
    parser.add_argument("--pair",         default=None,
                        help="Pair label for chart/summary (defaults to ticker or SAMPLE)")

    # ── Parameters ────────────────────────────────────────────────────────────
    parser.add_argument("--bb-window",       type=int,   default=20,   help="Bollinger Band window (default 20)")
    parser.add_argument("--bb-pct",          type=float, default=20.0, help="BB width squeeze percentile (default 20)")
    parser.add_argument("--rv-window",       type=int,   default=20,   help="Realised vol window (default 20)")
    parser.add_argument("--sigma-threshold", type=float, default=1.2,  help="σ-consumed threshold (default 1.2)")
    parser.add_argument("--iv-rv-threshold", type=float, default=1.2,  help="IV/RV threshold (default 1.2)")

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--out-csv",   default=None, help="Output CSV path (default: output/PAIR_YYYYMMDD.csv)")
    parser.add_argument("--out-chart", default=None, help="Output chart path (default: output/PAIR_YYYYMMDD.png)")
    parser.add_argument("--out-dir",   default="output", help="Output directory (default: output/)")
    parser.add_argument("--no-chart",  action="store_true", help="Skip chart generation")

    args = parser.parse_args()

    # ── Resolve pair label ────────────────────────────────────────────────────
    if args.pair:
        pair = args.pair
    elif args.blp or args.csv:
        pair = args.ticker if args.blp else os.path.splitext(os.path.basename(args.csv or ""))[0]
    else:
        pair = "SAMPLE"

    # ── Auto-generate output paths if not provided ────────────────────────────
    run_date = datetime.today().strftime("%Y%m%d")
    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"fx_vol_bands_{pair}_{run_date}"
    out_csv   = args.out_csv   or os.path.join(args.out_dir, f"{stem}.csv")
    out_chart = args.out_chart or os.path.join(args.out_dir, f"{stem}.png")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"[INFO] Loading data …")
    if args.blp:
        if not BLPAPI_AVAILABLE:
            sys.exit("[ERROR] blpapi not installed. Use --csv or --sample instead.")
        df = fetch_bloomberg_data(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            iv_constant=args.iv_constant,
            rr_constant=args.rr_constant,
        )
        pair = args.ticker
        print(f"[INFO] Bloomberg data loaded: {len(df)} rows for {pair}")

    elif args.csv:
        df = load_csv(args.csv)
        if args.iv_constant is not None:
            df["implied_vol_1w"] = df["implied_vol_1w"].fillna(args.iv_constant)
        if args.rr_constant is not None:
            df["risk_reversal_25d"] = df["risk_reversal_25d"].fillna(args.rr_constant)
        print(f"[INFO] CSV loaded: {len(df)} rows from {args.csv}")

    else:  # --sample
        df = generate_sample_data(pair=pair if pair != "SAMPLE" else "EURUSD")
        print(f"[INFO] Sample data generated: {len(df)} rows")

    # ── Compute ───────────────────────────────────────────────────────────────
    print("[INFO] Computing implied vol bands …")
    df = compute_vol_bands(df, iv_constant=args.iv_constant)

    print("[INFO] Computing Bollinger Bands …")
    df = compute_bollinger_bands(df, window=args.bb_window, pct_threshold=args.bb_pct)

    print("[INFO] Computing IV/RV ratio …")
    df = compute_iv_rv_ratio(df, rv_window=args.rv_window, iv_rv_threshold=args.iv_rv_threshold)

    print("[INFO] Computing squeeze flags …")
    df = compute_squeeze_flags(df, sigma_threshold=args.sigma_threshold, iv_rv_threshold=args.iv_rv_threshold)

    # ── Output ────────────────────────────────────────────────────────────────
    save_results(df, out_csv)

    if not args.no_chart:
        plot_results(df, pair=pair, out_path=out_chart)

    print_summary(df, pair=pair)


if __name__ == "__main__":
    main()
