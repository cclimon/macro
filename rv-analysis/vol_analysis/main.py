"""
main.py — CLI entry point for realized-volatility analysis.

Usage:
    python -m vol_analysis.main --pair EURUSD
    python -m vol_analysis.main --pair EURUSD --start 2020-01-01 --end 2024-12-31
    python main.py --pair EURUSD              (when run from vol_analysis/)
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

import numpy as np

from vol_analysis.bbg_connector import stop_session
from vol_analysis.config import TENOR_MAP, SIGNAL_RICH_THRESHOLD, SIGNAL_CHEAP_THRESHOLD
from vol_analysis.data_fetcher import fetch_ohlc, fetch_iv
from vol_analysis.estimators import yang_zhang_rv, ewma_rv
from vol_analysis.spread_analysis import compute_spreads, compute_percentile_ranks
from vol_analysis.plotter import plot_dashboard


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    today       = datetime.date.today()
    three_years = (today.replace(year=today.year - 3)).isoformat()

    parser = argparse.ArgumentParser(
        description="Realized vs Implied Volatility Analysis (Bloomberg data)"
    )
    parser.add_argument(
        "--pair", required=True, type=str.upper,
        help="FX currency pair (e.g. EURUSD)",
    )
    parser.add_argument(
        "--start", default=three_years,
        help="Start date YYYY-MM-DD (default: 3 years ago)",
    )
    parser.add_argument(
        "--end", default=today.isoformat(),
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent.parent / "output" / "rv-analysis"),
        help="Directory to save PNG dashboards",
    )
    return parser.parse_args()


# ── Snapshot printer ──────────────────────────────────────────────────────────

def _print_snapshot(pair: str, tenor: str, data) -> None:
    """Print the text snapshot for a tenor to stdout."""
    last      = data.iloc[-1]
    last_date = data.index[-1].strftime("%Y-%m-%d")

    def _signal(pct: float) -> str:
        if np.isnan(pct):
            return "NEUTRAL"
        return (
            "IV RICH"  if pct > SIGNAL_RICH_THRESHOLD  else
            "IV CHEAP" if pct < SIGNAL_CHEAP_THRESHOLD else
            "NEUTRAL"
        )

    sep = "-" * 45
    pct_yz_2y   = last["YZ_pct_2Y"]
    pct_ewma_2y = last["EWMA_pct_2Y"]

    pct_yz_str   = f"{pct_yz_2y:.0f}th"   if not np.isnan(pct_yz_2y)   else "N/A"
    pct_ewma_str = f"{pct_ewma_2y:.0f}th" if not np.isnan(pct_ewma_2y) else "N/A"

    print(f"\n  SNAPSHOT - {pair} {tenor}   as of {last_date}")
    print(f"  {sep}")
    print(f"  Implied Vol        :   {last['IV']:>6.2f}%")
    print(f"  Yang-Zhang RV      :   {last['YZ_RV']:>6.2f}%")
    print(f"  EWMA RV            :   {last['EWMA_RV']:>6.2f}%")
    print(f"  Spread YZ          :   {last['Spread_YZ']:>+6.2f} vol pts   [{pct_yz_str} pct / 2Y]")
    print(f"  Spread EWMA        :   {last['Spread_EWMA']:>+6.2f} vol pts   [{pct_ewma_str} pct / 2Y]")
    print(f"  Signal YZ          :   {_signal(pct_yz_2y)}")
    print(f"  Signal EWMA        :   {_signal(pct_ewma_2y)}")
    print(f"  {sep}\n")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(pair: str, start_date: str, end_date: str, output_dir: str) -> None:
    """Fetch data and produce dashboards for all five tenors."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[RV Analysis] pair={pair}  {start_date} -> {end_date}")
    print("Fetching OHLC data from Bloomberg ...")
    ohlc = fetch_ohlc(pair, start_date, end_date)
    print(f"  OHLC loaded: {len(ohlc)} rows  ({ohlc.index[0].date()} -> {ohlc.index[-1].date()})")

    for tenor in TENOR_MAP:
        window = TENOR_MAP[tenor]["window"]
        print(f"\n-- {tenor} (window={window}d) --------------------------")

        print("  Fetching implied vol …")
        iv = fetch_iv(pair, tenor, start_date, end_date)

        print("  Computing Yang-Zhang RV …")
        yz = yang_zhang_rv(ohlc, window)

        print("  Computing EWMA RV …")
        ew = ewma_rv(ohlc)

        print("  Computing spreads and percentile ranks …")
        spread_df = compute_spreads(iv, yz, ew)
        result_df = compute_percentile_ranks(spread_df)

        if result_df.empty:
            print("  WARNING: no data after alignment — skipping tenor.")
            continue

        _print_snapshot(pair, tenor, result_df)

        print("  Saving dashboard PNG …")
        path = plot_dashboard(pair, tenor, result_df, output_dir=output_dir)
        print(f"  Saved -> {path}")

    stop_session()
    print("\n[RV Analysis] Done.")


def main() -> None:
    args = _parse_args()
    try:
        run(
            pair=args.pair,
            start_date=args.start,
            end_date=args.end,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        stop_session()
        sys.exit(1)


if __name__ == "__main__":
    main()
