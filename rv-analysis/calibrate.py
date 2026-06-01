"""
calibrate.py - Compare Yang-Zhang and EWMA daily RV estimators against
               30-min high-frequency realized volatility.

The 30-min HF-RV is used as the benchmark (sum of squared log-returns per
day, annualised). Yang-Zhang and EWMA are computed from daily OHLC over the
same period using a 2-week (10-day) rolling window.

Usage:
    python calibrate.py --pair EURUSD
    python calibrate.py --pair USDMXN --days 90 --output-dir results/
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from vol_analysis.bbg_connector import intraday_bars, stop_session
from vol_analysis.config import ANNUALISATION_FACTOR
from vol_analysis.data_fetcher import fetch_ohlc
from vol_analysis.estimators import ewma_rv, yang_zhang_rv

ROLL = 10  # 2-week rolling window in trading days


# -- HF-RV from 30-min bars ----------------------------------------------------

def compute_hf_rv(bars: pd.DataFrame) -> pd.Series:
    """Aggregate 30-min bars into daily realized volatility.

    Sums squared log-returns within each UTC calendar day, then annualises.
    Returns annualised vol in % on the same scale as YZ and EWMA.
    """
    log_ret = np.log(bars["close"] / bars["close"].shift(1)).dropna()
    sq_ret  = log_ret ** 2

    daily_var = sq_ret.groupby(sq_ret.index.normalize()).sum()
    hf_rv     = np.sqrt(daily_var * ANNUALISATION_FACTOR) * 100
    hf_rv.name = "HF_RV"
    return hf_rv


# -- Rolling metrics -----------------------------------------------------------

def _rolling_rmse(actual: pd.Series, predicted: pd.Series, window: int) -> pd.Series:
    return np.sqrt(((predicted - actual) ** 2).rolling(window, min_periods=window).mean())


def _rolling_mae(actual: pd.Series, predicted: pd.Series, window: int) -> pd.Series:
    return (predicted - actual).abs().rolling(window, min_periods=window).mean()


def _rolling_corr(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
    return s1.rolling(window, min_periods=window).corr(s2)


# -- Console summary -----------------------------------------------------------

def _print_summary(pair: str, df: pd.DataFrame) -> None:
    yz_err   = df["YZ_RV"]   - df["HF_RV"]
    ewma_err = df["EWMA_RV"] - df["HF_RV"]

    rmse_yz   = float(np.sqrt((yz_err ** 2).mean()))
    rmse_ewma = float(np.sqrt((ewma_err ** 2).mean()))
    winner    = "Yang-Zhang" if rmse_yz < rmse_ewma else "EWMA"

    print(f"\n  CALIBRATION SUMMARY - {pair}")
    print(f"  Period : {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} days)")
    print(f"  Window : {ROLL}-day rolling\n")
    print(f"  {'Metric':<26} {'Yang-Zhang':>12} {'EWMA':>12}")
    print(f"  {'-' * 52}")
    print(f"  {'RMSE  (vol pts)':<26} {rmse_yz:>12.3f} {rmse_ewma:>12.3f}")
    print(f"  {'MAE   (vol pts)':<26} {yz_err.abs().mean():>12.3f} {ewma_err.abs().mean():>12.3f}")
    print(f"  {'Mean bias (vol pts)':<26} {yz_err.mean():>12.3f} {ewma_err.mean():>12.3f}")
    print(f"  {'Correlation vs HF-RV':<26} {df['YZ_RV'].corr(df['HF_RV']):>12.3f} {df['EWMA_RV'].corr(df['HF_RV']):>12.3f}")
    print(f"  {'-' * 52}")
    print(f"  Lower RMSE -> {winner}\n")


# -- Chart ---------------------------------------------------------------------

def _plot(pair: str, df: pd.DataFrame, output_dir: str) -> str:
    fig, axes = plt.subplots(
        3, 1, figsize=(13, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2]},
    )
    fig.suptitle(
        f"{pair}  -  RV Calibration vs 30-min HF-RV  |  {ROLL}-day rolling window\n"
        f"{df.index[0].date()} -> {df.index[-1].date()}",
        fontsize=12, fontweight="bold",
    )

    # -- Panel 1: vol series
    ax = axes[0]
    ax.plot(df.index, df["HF_RV"],   color="black",       lw=1.4, label="30-min HF-RV  (benchmark)")
    ax.plot(df.index, df["YZ_RV"],   color="steelblue",   lw=1.0, label="Yang-Zhang  (daily)")
    ax.plot(df.index, df["EWMA_RV"], color="darkorange",  lw=1.0, label="EWMA  (daily, λ=0.94)")
    ax.set_ylabel("Annualised Vol (%)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, color="#cccccc", alpha=0.4)

    # -- Panel 2: rolling RMSE
    ax = axes[1]
    ax.plot(df.index, df["RMSE_YZ"],   color="steelblue",  lw=1.1, label=f"RMSE  YZ   (roll {ROLL}d)")
    ax.plot(df.index, df["RMSE_EWMA"], color="darkorange", lw=1.1, label=f"RMSE  EWMA (roll {ROLL}d)")
    ax.fill_between(
        df.index,
        df["RMSE_YZ"], df["RMSE_EWMA"],
        where=(df["RMSE_YZ"] < df["RMSE_EWMA"]),
        alpha=0.12, color="steelblue", label="YZ better",
    )
    ax.fill_between(
        df.index,
        df["RMSE_YZ"], df["RMSE_EWMA"],
        where=(df["RMSE_EWMA"] < df["RMSE_YZ"]),
        alpha=0.12, color="darkorange", label="EWMA better",
    )
    ax.set_ylabel("RMSE (vol pts)")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, color="#cccccc", alpha=0.4)

    # -- Panel 3: rolling correlation
    ax = axes[2]
    ax.plot(df.index, df["Corr_YZ"],   color="steelblue",  lw=1.1, label=f"Corr  YZ   (roll {ROLL}d)")
    ax.plot(df.index, df["Corr_EWMA"], color="darkorange", lw=1.1, label=f"Corr  EWMA (roll {ROLL}d)")
    ax.axhline(1.0, color="#999999", lw=0.6, ls="--")
    ax.set_ylabel("Correlation vs HF-RV")
    ax.set_ylim(bottom=0, top=1.05)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, color="#cccccc", alpha=0.4)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    fig.tight_layout()
    path = f"{output_dir}/{pair}_calibration.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# -- Entry point ---------------------------------------------------------------

def calibrate(pair: str, days: int = 120, output_dir: str = ".") -> None:
    today    = datetime.date.today()
    start_d  = today - datetime.timedelta(days=days)
    start_str = start_d.isoformat()
    end_str   = today.isoformat()

    print(f"\n[Calibrate] {pair}  |  {start_str} -> {end_str}  |  rolling {ROLL}d window")

    # 30-min intraday bars -> daily HF-RV
    print("  Fetching 30-min intraday bars from Bloomberg ...")
    start_dt = datetime.datetime(start_d.year, start_d.month, start_d.day, 0, 0)
    end_dt   = datetime.datetime(today.year,   today.month,   today.day,  23, 59)
    bars     = intraday_bars(f"{pair} Curncy", start_dt, end_dt, interval=30)
    print(f"  Intraday bars: {len(bars)} rows  ({bars.index[0]} -> {bars.index[-1]})")

    hf_rv = compute_hf_rv(bars)
    print(f"  HF-RV: {len(hf_rv)} daily observations")

    # Daily OHLC -> YZ + EWMA
    print("  Fetching daily OHLC from Bloomberg ...")
    ohlc = fetch_ohlc(pair, start_str, end_str)
    print(f"  Daily OHLC: {len(ohlc)} rows")

    yz   = yang_zhang_rv(ohlc, window=ROLL)
    ewma = ewma_rv(ohlc)

    # Align on common trading dates
    df = pd.concat([hf_rv, yz, ewma], axis=1, join="inner").dropna()
    print(f"  Aligned: {len(df)} common trading days")

    if len(df) < ROLL + 1:
        raise ValueError(
            f"Only {len(df)} aligned days after join - need at least {ROLL + 1}. "
            "Try increasing --days."
        )

    # Rolling metrics
    df["RMSE_YZ"]   = _rolling_rmse(df["HF_RV"], df["YZ_RV"],   ROLL)
    df["RMSE_EWMA"] = _rolling_rmse(df["HF_RV"], df["EWMA_RV"], ROLL)
    df["MAE_YZ"]    = _rolling_mae( df["HF_RV"], df["YZ_RV"],   ROLL)
    df["MAE_EWMA"]  = _rolling_mae( df["HF_RV"], df["EWMA_RV"], ROLL)
    df["Corr_YZ"]   = _rolling_corr(df["HF_RV"], df["YZ_RV"],   ROLL)
    df["Corr_EWMA"] = _rolling_corr(df["HF_RV"], df["EWMA_RV"], ROLL)

    summary_df = df.dropna()
    _print_summary(pair, summary_df)

    path = _plot(pair, summary_df, output_dir)
    print(f"  Chart saved -> {path}")

    stop_session()
    print("[Calibrate] Done.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate YZ and EWMA daily RV against 30-min HF realized volatility"
    )
    parser.add_argument("--pair",       required=True, type=str.upper,
                        help="FX currency pair, e.g. EURUSD")
    parser.add_argument("--days",       default=120, type=int,
                        help="Lookback in calendar days (default: 120)")
    parser.add_argument("--output-dir",
                        default=str(Path(__file__).resolve().parent.parent / "output" / "rv-analysis"),
                        help="Directory to save the PNG chart")
    args = parser.parse_args()

    try:
        calibrate(pair=args.pair, days=args.days, output_dir=args.output_dir)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        stop_session()
        sys.exit(1)


if __name__ == "__main__":
    main()
