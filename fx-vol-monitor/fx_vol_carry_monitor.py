#!/usr/bin/env python3
"""
FX Volatility & Carry Monitor
Pulls 1-year daily data from Bloomberg (falls back to synthetic data).
Modules: vol bands, HAR-RV forecast, carry/vol ratios, squeeze flags.
Outputs per-pair charts and a summary heatmap to ./output/.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "fx-vol-monitor"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TODAY = datetime.today()
START_DATE = TODAY - timedelta(days=365)
START_STR = START_DATE.strftime("%Y%m%d")
END_STR = TODAY.strftime("%Y%m%d")

G10_MAJORS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
    "AUDUSD", "NZDUSD", "EURGBP", "EURCHF", "EURJPY",
]
CROSSES = [
    "GBPJPY", "AUDJPY", "CADJPY", "GBPAUD", "GBPCAD",
    "EURAUD", "EURCAD", "EURNZD",
]
ALL_PAIRS = G10_MAJORS + CROSSES
CURRENCIES = ["EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]

# ── Chart theme ───────────────────────────────────────────────────────────────
T = {
    "bg": "white",    "fg": "#111111", "grid": "#dddddd",
    "band1": "#aaaaaa", "band15": "#cc7722", "band2": "#cc2222",
    "inv": "#fce4e4",
    "spot": "#1a6fbd", "rv": "#217a3c", "har": "#c94a2a",
    "iv_rv": "#1a6fbd", "carry": "#7b2fa8", "squeeze": "#e00000",
    "attr": "#e8f5e9", "unattr": "#fffde7", "avoid": "#fce4e4",
}


# ─────────────────────────────────────────────────────────────────────────────
# Bloomberg
# ─────────────────────────────────────────────────────────────────────────────

def connect_bloomberg():
    """Open a Bloomberg session; returns None on any failure."""
    try:
        import blpapi
    except ImportError:
        print("[Bloomberg] blpapi not installed -> synthetic data.")
        return None

    opts = blpapi.SessionOptions()
    opts.setServerHost("localhost")
    opts.setServerPort(8194)
    session = blpapi.Session(opts)
    try:
        if not session.start():
            raise RuntimeError("Session.start() returned False")
        if not session.openService("//blp/refdata"):
            raise RuntimeError("openService('//blp/refdata') failed")
        print("[Bloomberg] Connected.")
        return session
    except Exception as exc:
        print(f"[Bloomberg] {exc} -> synthetic data.")
        try:
            session.stop()
        except Exception:
            pass
        return None


def _blp_hist(session, tickers: list[str]) -> pd.DataFrame:
    """Fetch daily PX_LAST for *tickers* over the global date range."""
    import blpapi

    svc = session.getService("//blp/refdata")
    req = svc.createRequest("HistoricalDataRequest")
    for t in tickers:
        req.append("securities", t)
    req.append("fields", "PX_LAST")
    req.set("startDate", START_STR)
    req.set("endDate", END_STR)
    req.set("periodicitySelection", "DAILY")
    session.sendRequest(req)

    raw: dict[str, dict[str, float]] = {}
    while True:
        ev = session.nextEvent(3000)
        for msg in ev:
            if not msg.hasElement("securityData"):
                continue
            sd = msg.getElement("securityData")
            sec = sd.getElementAsString("security")
            fd = sd.getElement("fieldData")
            rows: dict[str, float] = {}
            for i in range(fd.numValues()):
                row = fd.getValueAsElement(i)
                dt = row.getElementAsDatetime("date")
                key = f"{dt.year}-{dt.month:02d}-{dt.day:02d}"
                try:
                    rows[key] = row.getElementAsFloat("PX_LAST")
                except Exception:
                    rows[key] = np.nan
            raw[sec] = rows
        if ev.eventType() == blpapi.Event.RESPONSE:
            break

    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def fetch_spot(session, pairs: list[str]) -> pd.DataFrame:
    tickers = [f"{p} BGN Curncy" for p in pairs]
    df = _blp_hist(session, tickers)
    df.columns = [c.split()[0] for c in df.columns]
    return df


def fetch_vol_surface(session, pairs: list[str]):
    """Returns (vol_1w_df, vol_1m_df, rr_1w_df)."""
    d1w = _blp_hist(session, [f"{p}V1W BGN Curncy" for p in pairs])
    d1m = _blp_hist(session, [f"{p}V1M BGN Curncy" for p in pairs])
    drr = _blp_hist(session, [f"{p}1WRR BGN Curncy" for p in pairs])
    d1w.columns = [c.split("V1W")[0] for c in d1w.columns]
    d1m.columns = [c.split("V1M")[0] for c in d1m.columns]
    drr.columns = [c.split("1WRR")[0] for c in drr.columns]
    return d1w, d1m, drr


def fetch_carry_yields(session, currencies: list[str]) -> pd.DataFrame:
    ccy_tickers = [f"{c}I3M Curncy" for c in currencies]
    df = _blp_hist(session, ["USSOC Curncy"] + ccy_tickers)
    rename = {"USSOC Curncy": "USD"}
    rename.update({t: c for t, c in zip(ccy_tickers, currencies)})
    return df.rename(columns=rename)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback
# ─────────────────────────────────────────────────────────────────────────────

# (spot0, ann_vol, iv_1w_base, iv_1m_base, rr_base)
_SYNTH: dict[str, tuple] = {
    "EURUSD": (1.085, 0.065, 6.5,  6.8, -0.30),
    "GBPUSD": (1.270, 0.080, 7.5,  7.8, -0.40),
    "USDJPY": (150.0, 0.090, 8.5,  9.0,  0.50),
    "USDCHF": (0.900, 0.070, 6.0,  6.3,  0.20),
    "USDCAD": (1.360, 0.060, 5.8,  6.0, -0.20),
    "AUDUSD": (0.650, 0.100, 9.0,  9.5, -0.80),
    "NZDUSD": (0.600, 0.105, 9.5, 10.0, -0.90),
    "EURGBP": (0.854, 0.055, 5.2,  5.5, -0.10),
    "EURCHF": (0.958, 0.050, 4.8,  5.0,  0.10),
    "EURJPY": (162.0, 0.085, 8.0,  8.5,  0.30),
    "GBPJPY": (190.5, 0.095, 9.0,  9.5,  0.40),
    "AUDJPY": (97.50, 0.110, 10.0, 10.5, -0.50),
    "CADJPY": (110.3, 0.080, 7.5,  8.0,  0.20),
    "GBPAUD": (1.954, 0.110, 10.5, 11.0, -0.60),
    "GBPCAD": (1.714, 0.090, 8.5,  9.0, -0.30),
    "EURAUD": (1.668, 0.100, 9.5, 10.0, -0.70),
    "EURCAD": (1.474, 0.075, 7.0,  7.5, -0.20),
    "EURNZD": (1.808, 0.105, 10.0, 10.5, -0.80),
}
_YIELDS = {
    "USD": 5.30, "EUR": 3.60, "GBP": 5.00, "JPY": 0.10,
    "CHF": 1.50, "CAD": 4.75, "AUD": 4.35, "NZD": 5.00,
}


def generate_synthetic_data(pairs: list[str]):
    """Returns (spot_df, vol_1w_df, vol_1m_df, rr_1w_df, yield_df)."""
    print("[Synthetic] Generating fallback data...")
    dates = pd.bdate_range(start=START_DATE, end=TODAY)
    n = len(dates)
    rng = np.random.default_rng(42)

    spot_d, v1w_d, v1m_d, rr_d = {}, {}, {}, {}
    for pair in pairs:
        s0, ann_vol, iv1w0, iv1m0, rr0 = _SYNTH.get(pair, (1.0, 0.08, 8.0, 8.5, 0.0))
        dt = 1 / 252
        lr = -0.5 * ann_vol ** 2 * dt + ann_vol * np.sqrt(dt) * rng.standard_normal(n)
        spot_d[pair] = s0 * np.exp(np.cumsum(lr))

        # Mean-reverting implied vol
        iv1w = np.empty(n)
        iv1m = np.empty(n)
        iv1w[0], iv1m[0] = iv1w0, iv1m0
        for t in range(1, n):
            iv1w[t] = iv1w[t-1] + 0.06 * (iv1w0 - iv1w[t-1]) + 0.35 * rng.standard_normal()
            iv1m[t] = iv1m[t-1] + 0.04 * (iv1m0 - iv1m[t-1]) + 0.20 * rng.standard_normal()
        v1w_d[pair] = np.clip(iv1w, iv1w0 * 0.4, iv1w0 * 3.0)
        v1m_d[pair] = np.clip(iv1m, iv1m0 * 0.4, iv1m0 * 3.0)
        rr_d[pair] = rr0 + 0.25 * rng.standard_normal(n)

    yield_d = {
        ccy: _YIELDS[ccy] + 0.08 * rng.standard_normal(n)
        for ccy in ["USD"] + CURRENCIES
    }

    return (
        pd.DataFrame(spot_d, index=dates),
        pd.DataFrame(v1w_d, index=dates),
        pd.DataFrame(v1m_d, index=dates),
        pd.DataFrame(rr_d, index=dates),
        pd.DataFrame(yield_d, index=dates),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Module 1 – Realised volatility
# ─────────────────────────────────────────────────────────────────────────────

def compute_realised_vol(spot_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling 20-day annualised realised vol (%)."""
    return np.log(spot_df / spot_df.shift(1)).rolling(window).std() * np.sqrt(252) * 100


# ─────────────────────────────────────────────────────────────────────────────
# Module 2 – HAR-RV forecast
# ─────────────────────────────────────────────────────────────────────────────

def fit_har_rv(spot_series: pd.Series) -> tuple[pd.Series, float]:
    """
    Fit HAR-RV on daily log returns of spot_series.

    Returns:
      rolling_forecast : Series aligned to spot_series.index (annualised %)
      current_forecast : scalar, latest 1-step-ahead forecast (annualised %)
    """
    log_r = np.log(spot_series / spot_series.shift(1)).dropna()
    rv_d = log_r ** 2          # daily variance
    rv_w = rv_d.rolling(5).mean()
    rv_m = rv_d.rolling(22).mean()

    df = pd.concat([rv_d, rv_w, rv_m], axis=1, keys=["d", "w", "m"]).dropna()
    y = rv_d.shift(-1).reindex(df.index).dropna()
    X = df.loc[y.index]

    rolling_forecast = pd.Series(np.nan, index=spot_series.index, dtype=float)
    current_forecast = np.nan

    if len(y) < 30:
        return rolling_forecast, current_forecast

    A = np.column_stack([np.ones(len(X)), X["d"].values, X["w"].values, X["m"].values])
    coeffs, *_ = np.linalg.lstsq(A, y.values, rcond=None)

    A_full = np.column_stack([np.ones(len(df)), df["d"].values, df["w"].values, df["m"].values])
    fitted_var = np.maximum(A_full @ coeffs, 0.0)
    forecast_ann = np.sqrt(fitted_var) * np.sqrt(252) * 100

    forecast_s = pd.Series(forecast_ann, index=df.index)
    rolling_forecast.update(forecast_s)

    last = df.iloc[-1]
    cv = max(coeffs[0] + coeffs[1] * last["d"] + coeffs[2] * last["w"] + coeffs[3] * last["m"], 0.0)
    current_forecast = np.sqrt(cv) * np.sqrt(252) * 100

    return rolling_forecast, current_forecast


# ─────────────────────────────────────────────────────────────────────────────
# Module 3 – Carry / vol
# ─────────────────────────────────────────────────────────────────────────────

def _carry_diff_series(pair: str, yield_df: pd.DataFrame) -> pd.Series:
    nan_s = pd.Series(np.nan, index=yield_df.index)
    if "USD" in pair:
        ccy = pair.replace("USD", "")
        return yield_df.get(ccy, nan_s) - yield_df.get("USD", nan_s)
    leg1, leg2 = pair[:3], pair[3:]
    return yield_df.get(leg1, nan_s) - yield_df.get(leg2, nan_s)


def compute_carry_vol(
    yield_df: pd.DataFrame,
    vol_1w_df: pd.DataFrame,
    pairs: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    carry_diff = pd.DataFrame({p: _carry_diff_series(p, yield_df) for p in pairs})
    iv = vol_1w_df.reindex(columns=pairs).replace(0.0, np.nan)
    carry_ratio = carry_diff / iv
    return carry_diff, carry_ratio


def _carry_label(ratio: float) -> str:
    if np.isnan(ratio):
        return "N/A"
    if ratio < 0:
        return "AVOID"
    if ratio < 0.5:
        return "UNATTRACTIVE"
    if ratio < 1.5:
        return "NEUTRAL"
    return "ATTRACTIVE"


# ─────────────────────────────────────────────────────────────────────────────
# Summary monitor table
# ─────────────────────────────────────────────────────────────────────────────

def compute_vol_monitor(
    spot_df: pd.DataFrame,
    vol_1w_df: pd.DataFrame,
    vol_1m_df: pd.DataFrame,
    rv_df: pd.DataFrame,
    har_forecasts: dict[str, tuple],
    carry_diff_df: pd.DataFrame,
    carry_ratio_df: pd.DataFrame,
) -> pd.DataFrame:

    def _last(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns:
            return np.nan
        s = df[col].dropna()
        return float(s.iloc[-1]) if len(s) else np.nan

    rows = []
    for pair in ALL_PAIRS:
        spot   = _last(spot_df, pair)
        iv_1w  = _last(vol_1w_df, pair)
        iv_1m  = _last(vol_1m_df, pair)
        rv20   = _last(rv_df, pair)
        _, har_cur = har_forecasts.get(pair, (None, np.nan))
        carry_diff  = _last(carry_diff_df, pair)
        carry_ratio = _last(carry_ratio_df, pair)

        ts_ratio = (iv_1w / iv_1m) if (not np.isnan(iv_1m) and iv_1m > 0) else np.nan
        iv_rv    = (iv_1w / har_cur) if (not np.isnan(har_cur) and har_cur > 0) else np.nan

        # 3-month spot percentile rank (≈63 business days)
        spot_pctile = np.nan
        if pair in spot_df.columns:
            s = spot_df[pair].dropna()
            if len(s) >= 5:
                window = s.iloc[-63:] if len(s) >= 63 else s
                lo, hi = window.min(), window.max()
                spot_pctile = (spot - lo) / (hi - lo) * 100 if hi > lo else 50.0

        # Squeeze: last-day spot move > 1.2σ_1w AND IV/RV > 1.2 AND TS inverted
        squeeze = False
        if pair in spot_df.columns and not np.isnan(iv_1w):
            s = spot_df[pair].dropna()
            if len(s) >= 2:
                move     = abs(s.iloc[-1] - s.iloc[-2])
                sig_1w   = s.iloc[-1] * (iv_1w / 100) * np.sqrt(7 / 365)
                sc       = move / sig_1w if sig_1w > 0 else 0.0
                squeeze  = (
                    sc > 1.2
                    and not np.isnan(iv_rv) and iv_rv > 1.2
                    and not np.isnan(ts_ratio) and ts_ratio > 1.0
                )

        rows.append({
            "Pair": pair, "Spot": spot, "Spot_Pctile": spot_pctile,
            "1W_IV": iv_1w, "1M_IV": iv_1m,
            "RV_20d": rv20, "HAR_Forecast": har_cur,
            "IV_RV_Ratio": iv_rv, "TS_Ratio": ts_ratio,
            "Carry_Diff": carry_diff, "Carry_Vol_Ratio": carry_ratio,
            "Carry_Label": _carry_label(carry_ratio),
            "Squeeze": squeeze,
        })

    return pd.DataFrame(rows).set_index("Pair")


def flag_squeeze(monitor_df: pd.DataFrame) -> pd.DataFrame:
    """Passthrough – squeeze already computed in compute_vol_monitor."""
    return monitor_df


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_theme(fig, axes):
    fig.patch.set_facecolor(T["bg"])
    for ax in axes:
        ax.set_facecolor(T["bg"])
        ax.tick_params(colors=T["fg"], labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(T["grid"])
        ax.grid(True, color=T["grid"], lw=0.4, alpha=0.6)
        ax.yaxis.label.set_color(T["fg"])
        ax.xaxis.label.set_color(T["fg"])
        ax.title.set_color(T["fg"])


# ─────────────────────────────────────────────────────────────────────────────
# Chart A – per-pair 4-panel
# ─────────────────────────────────────────────────────────────────────────────

def plot_pair(
    pair: str,
    spot_df: pd.DataFrame,
    vol_1w_df: pd.DataFrame,
    vol_1m_df: pd.DataFrame,
    rv_df: pd.DataFrame,
    har_forecasts: dict,
    carry_ratio_df: pd.DataFrame,
):
    if pair not in spot_df.columns:
        return

    spot = spot_df[pair].dropna()
    dates = spot.index

    def _align(df):
        return df[pair].reindex(dates) if pair in df.columns else pd.Series(np.nan, index=dates)

    iv_1w = _align(vol_1w_df)
    iv_1m = _align(vol_1m_df)
    rv20  = _align(rv_df)
    har_s, _ = har_forecasts.get(pair, (pd.Series(np.nan, index=dates), np.nan))
    har_s     = har_s.reindex(dates)
    carry_r   = _align(carry_ratio_df)

    fig = plt.figure(figsize=(14, 12))
    gs  = GridSpec(4, 1, figure=fig, hspace=0.06, top=0.94, bottom=0.06)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    _apply_theme(fig, [ax1, ax2, ax3, ax4])

    # ── Panel 1: Spot + IV bands + squeeze dots ───────────────────────────
    ax1.plot(dates, spot, color=T["spot"], lw=1.0, label="Spot", zorder=3)

    sigma_width = (iv_1w / 100) * np.sqrt(7 / 365)
    for sig, col, lbl in [(2.0, T["band2"], "±2σ"),
                          (1.5, T["band15"], "±1.5σ"),
                          (1.0, T["band1"], "±1σ")]:
        up = spot * np.exp( sig * sigma_width)
        dn = spot * np.exp(-sig * sigma_width)
        ax1.fill_between(dates, dn, up, color=col, alpha=0.12)
        ax1.plot(dates, up, color=col, lw=0.6, alpha=0.7, label=lbl)
        ax1.plot(dates, dn, color=col, lw=0.6, alpha=0.7)

    # Squeeze markers: spot move > 1.2σ_1w AND IV/HAR-RV > 1.2 AND TS inverted
    ts_s       = iv_1w / iv_1m.replace(0, np.nan)
    iv_rv_s    = iv_1w / har_s.replace(0, np.nan)
    prev_band  = spot.shift(1) * sigma_width.shift(1).replace(0, np.nan)
    sigma_cons = np.log(spot / spot.shift(1)).abs() / prev_band
    sq_mask    = (sigma_cons > 1.2) & (iv_rv_s > 1.2) & (ts_s > 1.0)
    sq_idx     = sq_mask[sq_mask].index
    if len(sq_idx):
        ax1.scatter(sq_idx, spot.loc[sq_idx],
                    color=T["squeeze"], s=30, zorder=5, marker="o", label="Squeeze")

    ax1.set_title(f"{pair}  –  FX Vol & Carry Monitor", fontsize=11, loc="left", pad=5)
    ax1.set_ylabel("Spot", fontsize=8)
    ax1.legend(loc="upper left", fontsize=6, ncol=5, framealpha=0.25,
               labelcolor=T["fg"], facecolor=T["bg"])
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Panel 2: Term-structure ratio ─────────────────────────────────────
    ts = iv_1w / iv_1m.replace(0, np.nan)
    ax2.plot(dates, ts, color=T["iv_rv"], lw=0.9, label="1W / 1M")
    ax2.axhline(1.0, color="#e05252", lw=0.8, ls="--", alpha=0.8, label="1.0")
    ax2.fill_between(dates, ts, 1.0,
                     where=ts.gt(1.0).fillna(False),
                     color=T["inv"], alpha=0.7, label="Inverted")
    ax2.set_ylabel("TS Ratio", fontsize=8)
    ax2.legend(loc="upper left", fontsize=6, ncol=3, framealpha=0.25,
               labelcolor=T["fg"], facecolor=T["bg"])
    plt.setp(ax2.get_xticklabels(), visible=False)

    # ── Panel 3: IV/HAR-RV + twin-axis 20d RV vs HAR-RV ──────────────────
    iv_rv_plot = iv_1w / har_s.replace(0, np.nan)
    ax3.plot(dates, iv_rv_plot, color=T["iv_rv"], lw=0.9, label="IV / HAR-RV")
    ax3.axhline(1.2, color="#e05252", lw=0.8, ls="--", alpha=0.8, label="1.2×")
    ax3.set_ylabel("IV/RV Ratio", fontsize=8)

    ax3r = ax3.twinx()
    ax3r.set_facecolor(T["bg"])
    ax3r.tick_params(colors=T["fg"], labelsize=7)
    for sp in ax3r.spines.values():
        sp.set_edgecolor(T["grid"])
    ax3r.plot(dates, rv20,  color=T["rv"],  lw=0.75, alpha=0.8, label="20d RV")
    ax3r.plot(dates, har_s, color=T["har"], lw=0.75, alpha=0.8, ls="--", label="HAR-RV")
    ax3r.set_ylabel("RV / HAR (%)", fontsize=7, color=T["fg"])

    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3r.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6, ncol=4,
               framealpha=0.25, labelcolor=T["fg"], facecolor=T["bg"])
    plt.setp(ax3.get_xticklabels(), visible=False)

    # ── Panel 4: Carry / vol ──────────────────────────────────────────────
    cr   = carry_r.reindex(dates)
    yhi  = max(cr.max(skipna=True) * 1.15, 2.5)  if not cr.isna().all() else  2.5
    ylo  = min(cr.min(skipna=True) * 1.15, -1.5) if not cr.isna().all() else -1.5

    ax4.fill_between(dates, ylo,  0.0, color=T["avoid"],  alpha=0.55)
    ax4.fill_between(dates, 0.0,  0.5, color=T["unattr"], alpha=0.55)
    ax4.fill_between(dates, 1.5,  yhi, color=T["attr"],   alpha=0.55)
    ax4.axhline(1.5, color="#3fb950", lw=0.7, ls=":", alpha=0.9)
    ax4.axhline(0.5, color="#e3b341", lw=0.7, ls=":", alpha=0.9)
    ax4.axhline(0.0, color="#f85149", lw=0.7, ls=":", alpha=0.9)
    ax4.plot(dates, cr, color=T["carry"], lw=1.0, label="Carry/Vol", zorder=3)
    ax4.set_ylim(ylo, yhi)
    ax4.set_ylabel("Carry/Vol", fontsize=8)
    ax4.legend(loc="upper left", fontsize=6, framealpha=0.25,
               labelcolor=T["fg"], facecolor=T["bg"])
    ax4.xaxis.set_tick_params(rotation=25, labelsize=7)

    out = OUTPUT_DIR / f"{pair}_monitor.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=T["bg"])
    plt.close(fig)
    print(f"  {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart B – summary heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_heatmap(summary_df: pd.DataFrame):
    col_keys = ["Spot", "1W_IV", "IV_RV_Ratio", "TS_Ratio",
                "HAR_Forecast", "Carry_Vol_Ratio", "Squeeze"]
    col_hdrs = ["Spot", "1W IV (%)", "IV/RV", "TS Ratio",
                "HAR-RV (%)", "Carry/Vol", "Squeeze"]

    pairs = list(summary_df.index)
    n_r, n_c = len(pairs), len(col_keys)

    # Build float matrix — Spot column uses Spot_Pctile for colour, raw Spot for display
    color_keys = ["Spot_Pctile" if k == "Spot" else k for k in col_keys]
    mat = np.full((n_r, n_c), np.nan)
    for j, k in enumerate(color_keys):
        col = summary_df[k].values
        mat[:, j] = col.astype(float) if k == "Squeeze" else pd.to_numeric(col, errors="coerce")

    # Per-column colormap + norm
    cmaps, norms = [], []
    for j, k in enumerate(color_keys):
        valid = mat[:, j][~np.isnan(mat[:, j])]
        if k == "Squeeze":
            cmaps.append(mcolors.ListedColormap(["#e8e8e8", "#cc0000"]))
            norms.append(mcolors.BoundaryNorm([0, 0.5, 1], 2))
        elif k == "Spot_Pctile":
            # Red = near 3-month high, green = near 3-month low, white = mid-range
            cmaps.append(plt.cm.RdYlGn_r)
            norms.append(mcolors.Normalize(vmin=0, vmax=100))
        elif k in ("IV_RV_Ratio", "TS_Ratio"):
            cmaps.append(plt.cm.RdYlGn_r)
            norms.append(mcolors.Normalize(vmin=0.7, vmax=2.0))
        elif k == "Carry_Vol_Ratio":
            cmaps.append(plt.cm.RdYlGn)
            norms.append(mcolors.Normalize(vmin=-1.0, vmax=2.0))
        elif k in ("1W_IV", "HAR_Forecast"):
            cmaps.append(plt.cm.RdYlGn_r)
            lo = np.nanpercentile(valid, 10) if len(valid) else 0
            hi = np.nanpercentile(valid, 90) if len(valid) else 10
            norms.append(mcolors.Normalize(vmin=lo, vmax=hi))
        else:
            cmaps.append(plt.cm.Blues)
            lo = np.nanmin(valid) if len(valid) else 0
            hi = np.nanmax(valid) if len(valid) else 1
            norms.append(mcolors.Normalize(vmin=lo, vmax=hi))

    fig, ax = plt.subplots(figsize=(15, 0.55 * n_r + 3.5))
    fig.patch.set_facecolor(T["bg"])
    ax.set_facecolor(T["bg"])
    ax.axis("off")

    CW, CH, PAD = 1.0, 0.80, 0.04

    for i, pair in enumerate(pairs):
        row_y = n_r - i - 1
        ax.text(-0.55, row_y + CH / 2, pair, ha="right", va="center",
                fontsize=8, color=T["fg"], fontweight="bold")

        for j in range(n_c):
            val = mat[i, j]
            rgba = (0.08, 0.08, 0.12, 1.0) if np.isnan(val) else cmaps[j](norms[j](val))
            r, g, b, _ = rgba
            txt_col = "#0d1117" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.45 else T["fg"]

            rect = mpatches.FancyBboxPatch(
                (j * CW + PAD, row_y + PAD),
                CW - 2 * PAD, CH - 2 * PAD,
                boxstyle="round,pad=0.02",
                facecolor=rgba, edgecolor=T["grid"], lw=0.3,
            )
            ax.add_patch(rect)

            k = col_keys[j]
            if k == "Squeeze":
                txt = "YES" if val == 1.0 else "–"
            elif np.isnan(val):
                txt = "–"
            elif k == "Spot":
                txt = f"{val:.2f}" if val >= 10 else f"{val:.4f}"
            else:
                txt = f"{val:.2f}"

            ax.text(j * CW + CW / 2, row_y + CH / 2, txt,
                    ha="center", va="center",
                    fontsize=7.5, color=txt_col, fontweight="bold")

        # Carry label right column
        label = summary_df.loc[pair, "Carry_Label"]
        lc = {"ATTRACTIVE": "#3fb950", "UNATTRACTIVE": "#e3b341",
              "AVOID": "#f85149", "NEUTRAL": "#79c0ff", "N/A": T["grid"]}
        ax.text(n_c * CW + 0.15, row_y + CH / 2, label,
                ha="left", va="center", fontsize=7,
                color=lc.get(label, T["fg"]), fontweight="bold")

    # Column headers
    for j, hdr in enumerate(col_hdrs):
        ax.text(j * CW + CW / 2, n_r + 0.05, hdr,
                ha="center", va="bottom", fontsize=8,
                color=T["fg"], fontweight="bold")
    ax.text(n_c * CW + 0.15, n_r + 0.05, "Carry Label",
            ha="left", va="bottom", fontsize=8, color=T["fg"], fontweight="bold")

    # ── Colour-scale legend ───────────────────────────────────────────────
    SEP_Y   = -0.72
    BAR_Y   = -1.50
    BAR_H   = 0.32
    LBL_Y   = -1.92
    N_STEPS = 30

    ax.plot([0, n_c * CW], [SEP_Y, SEP_Y], color=T["grid"], lw=0.7)
    ax.text(-0.55, (SEP_Y + BAR_Y + BAR_H) / 2, "Colour\nkey:",
            ha="right", va="center", fontsize=6.5, color="#555555",
            style="italic", linespacing=1.4)

    leg_lo = ["3M Low",  "Low",  "0.7",  "0.7",  "Low",  "AVOID",       ""]
    leg_hi = ["3M High", "High", "2.0",  "2.0",  "High", "ATTRACTIVE",  ""]

    for j in range(n_c):
        x0 = j * CW + PAD
        bw = CW - 2 * PAD

        if col_keys[j] == "Squeeze":
            sw = bw / 2 - 0.04
            for sx, fc, lbl, tc in [
                (x0,             "#e8e8e8", "No",  T["fg"]),
                (x0 + sw + 0.08, "#cc0000", "YES", "white"),
            ]:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (sx, BAR_Y), sw, BAR_H,
                    boxstyle="round,pad=0.01",
                    facecolor=fc, edgecolor=T["grid"], lw=0.5))
                ax.text(sx + sw / 2, BAR_Y + BAR_H / 2, lbl,
                        ha="center", va="center",
                        fontsize=6.5, color=tc, fontweight="bold")
        else:
            step_w = bw / N_STEPS
            v0, v1 = norms[j].vmin, norms[j].vmax
            for k in range(N_STEPS):
                v = v0 + (v1 - v0) * (k + 0.5) / N_STEPS
                ax.add_patch(mpatches.Rectangle(
                    (x0 + k * step_w, BAR_Y), step_w, BAR_H,
                    facecolor=cmaps[j](norms[j](v)), edgecolor="none"))
            ax.add_patch(mpatches.Rectangle(
                (x0, BAR_Y), bw, BAR_H,
                facecolor="none", edgecolor=T["grid"], lw=0.5))
            ax.text(x0,      LBL_Y, leg_lo[j],
                    ha="left",  va="top", fontsize=6, color="#444444")
            ax.text(x0 + bw, LBL_Y, leg_hi[j],
                    ha="right", va="top", fontsize=6, color="#444444")

    ax.set_xlim(-1.2, n_c * CW + 1.8)
    ax.set_ylim(-2.5, n_r + 0.8)
    ax.set_title(
        f"FX Volatility & Carry Monitor  ·  {TODAY.strftime('%Y-%m-%d')}",
        color=T["fg"], fontsize=11, loc="left", pad=8, fontweight="bold",
    )

    out = OUTPUT_DIR / "fx_summary_heatmap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=T["bg"])
    plt.close(fig)
    print(f"  {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Legend text file
# ─────────────────────────────────────────────────────────────────────────────

def write_legend_txt():
    """Write a standalone colour-key reference to output/heatmap_legend.txt."""
    txt = f"""\
FX Volatility & Carry Monitor  --  Heatmap Colour Key
Generated: {TODAY.strftime('%Y-%m-%d')}
======================================================

COLUMN          COLOUR SCALE        LOW END (green)          HIGH END (red)
---------------------------------------------------------------------------
Spot            RdYlGn_r 0-100%     3M Low  (0th %ile)       3M High (100th %ile)
                Spot is coloured by where it sits within its 3-month
                high/low range. Red = near 3M high; green = near 3M low.

1W IV (%)       RdYlGn_r            Low IV (10th %ile         High IV (90th %ile
                cross-section)       of today's universe)      of today's universe)
                1-week ATM implied volatility level relative to peers.

IV / RV Ratio   RdYlGn_r 0.7-2.0   0.7  (vol cheap vs RV)   2.0  (vol rich vs RV)
                1W ATM IV divided by the HAR-RV 1-step forecast.
                Values > 1.2 suggest expensive vol; < 0.8 suggests cheap vol.

TS Ratio        RdYlGn_r 0.7-2.0   0.7  (steep contango)    2.0  (deeply inverted)
(Term Struct.)  1W ATM IV divided by 1M ATM IV.
                Values > 1.0 = term structure inverted (short-dated vol
                elevated vs medium-dated). Flag used in Squeeze trigger.

HAR-RV (%)      RdYlGn_r            Low forecast RV           High forecast RV
                cross-section        (10th %ile)               (90th %ile)
                5-day forward realised volatility forecast (annualised %)
                from the HAR-RV model (daily / weekly / monthly components).

Carry / Vol     RdYlGn -1.0 to 2.0  AVOID (<0)               ATTRACTIVE (>1.5)
                Carry differential (3M implied yields) divided by 1W ATM IV.
                  > 1.5  ATTRACTIVE   carry well compensated by vol
                  0.5-1.5 NEUTRAL
                  0-0.5  UNATTRACTIVE carry thin relative to vol cost
                  < 0    AVOID        negative carry

Squeeze Flag    Categorical          Light grey = No           Red = YES
                Triggered when ALL THREE conditions hold simultaneously:
                  1. Last-day spot move > 1.2x the 1W implied sigma band
                  2. IV / HAR-RV ratio > 1.2
                  3. Term structure inverted (1W/1M > 1.0)

CARRY LABEL COLUMN (right of grid)
  ATTRACTIVE    green   Carry/Vol ratio > 1.5
  NEUTRAL       blue    Carry/Vol ratio 0.5 to 1.5
  UNATTRACTIVE  yellow  Carry/Vol ratio 0.0 to 0.5
  AVOID         red     Carry/Vol ratio < 0  (negative carry)
"""
    out = OUTPUT_DIR / "heatmap_legend.txt"
    out.write_text(txt, encoding="utf-8")
    print(f"  {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FX Volatility & Carry Monitor")
    print(f"Range : {START_DATE.strftime('%Y-%m-%d')} to {TODAY.strftime('%Y-%m-%d')}")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print("=" * 60)

    # 1. Data acquisition
    session = connect_bloomberg()
    if session is not None:
        print("[Data] Fetching spot...")
        spot_df = fetch_spot(session, ALL_PAIRS)
        print("[Data] Fetching vol surface...")
        vol_1w_df, vol_1m_df, rr_1w_df = fetch_vol_surface(session, ALL_PAIRS)
        print("[Data] Fetching carry yields...")
        yield_df = fetch_carry_yields(session, CURRENCIES)
        session.stop()
    else:
        spot_df, vol_1w_df, vol_1m_df, rr_1w_df, yield_df = generate_synthetic_data(ALL_PAIRS)

    # 2. Realised vol
    print("[Compute] 20-day realised vol...")
    rv_df = compute_realised_vol(spot_df)

    # 3. HAR-RV (one fit per pair, result reused by table + charts)
    print("[Compute] HAR-RV forecasts...")
    har_forecasts: dict[str, tuple] = {}
    for pair in ALL_PAIRS:
        if pair in spot_df.columns:
            har_forecasts[pair] = fit_har_rv(spot_df[pair])
        else:
            har_forecasts[pair] = (pd.Series(dtype=float), np.nan)

    # 4. Carry ratios
    print("[Compute] Carry / vol ratios...")
    carry_diff_df, carry_ratio_df = compute_carry_vol(yield_df, vol_1w_df, ALL_PAIRS)

    # 5. Summary table
    print("[Compute] Summary monitor table...")
    monitor_df = compute_vol_monitor(
        spot_df, vol_1w_df, vol_1m_df, rv_df,
        har_forecasts, carry_diff_df, carry_ratio_df,
    )
    monitor_df = flag_squeeze(monitor_df)

    csv_path = OUTPUT_DIR / "fx_vol_carry_summary.csv"
    try:
        monitor_df.to_csv(csv_path)
        print(f"\n[CSV] {csv_path}")
    except PermissionError:
        print(f"\n[CSV] WARNING: {csv_path} is locked (close it in Excel). Skipping CSV write.")

    display_cols = ["Spot", "1W_IV", "IV_RV_Ratio", "TS_Ratio",
                    "HAR_Forecast", "Carry_Vol_Ratio", "Carry_Label", "Squeeze"]
    print(monitor_df[display_cols].to_string(float_format=lambda x: f"{x:.3f}"))

    # 6. Per-pair charts
    print(f"\n[Charts] Per-pair ({len(ALL_PAIRS)} pairs)...")
    for pair in ALL_PAIRS:
        plot_pair(pair, spot_df, vol_1w_df, vol_1m_df, rv_df,
                  har_forecasts, carry_ratio_df)

    # 7. Summary heatmap + legend key
    print("\n[Charts] Summary heatmap + legend key...")
    plot_summary_heatmap(monitor_df)
    write_legend_txt()

    print(f"\n[Done] {len(ALL_PAIRS)} pairs processed. Outputs in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
