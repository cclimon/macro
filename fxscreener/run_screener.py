"""
FX Screener — runner.

    python run_screener.py                    # full run, prints ranks
    python run_screener.py --start 2015-01-01 --out results
    python run_screener.py --top 8            # show top/bottom 8 only

Outputs (to --out):
    ranks_outright.csv    full cross-section with pillar decomposition
    ranks_crosses.csv     RV scores for the configured cross list
    trade_notes.txt       one line per actionable name
    score_history.csv     composite time series, for the IC study
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

import expression
import score as scoring
from config import PILLAR_WEIGHTS
from pillars import MarketData

log = logging.getLogger("screener")


def run(start: str, end: str | None, outdir: Path, top: int) -> None:
    md = MarketData(start=start, end=end)

    # Print this FIRST and read it before anything else. If a carry number
    # looks wrong versus the policy differential you know, the divisor is
    # wrong and every downstream rank for that currency is wrong with it.
    if not md.fwd_diagnostic.empty:
        print("\n=== FORWARD CONVENTION / CARRY CHECK ===")
        # Select only the columns that exist, so a schema change in the
        # diagnostic cannot take down the whole run.
        want = ["ticker", "mode", "divisor", "raw_pts", "spot",
                "outright", "carry_pct"]
        have = [c for c in want if c in md.fwd_diagnostic.columns]
        print(md.fwd_diagnostic[have].to_string() if have
              else md.fwd_diagnostic.to_string())
        try:
            md.sanity_check_carry(raise_on_fail=True)
            print("\n  all carry values inside the plausibility band")
        except ValueError as e:
            raise SystemExit(
                f"\n{e}\n\nFix the divisor in config.FWD_CONVENTIONS, then "
                f"re-run. Nothing downstream is trustworthy until this passes."
            ) from None
    else:
        print("\n!! No forward diagnostic — carry and carry-adjusted risk "
              "will be empty. Run `python check_data.py --section fwd`.")

    pillars = scoring.build_pillars(md)
    if not pillars:
        raise SystemExit("No pillars resolved. Run `python bbg.py --validate` first.")

    total = scoring.composite(pillars)
    conv = scoring.conviction(pillars, total)
    snap = scoring.snapshot(md, pillars, total, conv)

    ivrv = expression.ivrv_z(md.atm_vol, md.rv_3m)
    skw = expression.skew_z(md.rr25, md.pairs)
    snap = expression.build(snap, ivrv, skw)

    crosses = scoring.cross_scores(total)
    cross_snap = (crosses.ffill().iloc[-1]
                  .rename("score").sort_values(ascending=False).to_frame())
    cross_snap["abs"] = cross_snap["score"].abs()
    cross_snap = cross_snap.sort_values("abs", ascending=False).drop(columns="abs")

    outdir.mkdir(parents=True, exist_ok=True)
    snap.to_csv(outdir / "ranks_outright.csv")
    cross_snap.to_csv(outdir / "ranks_crosses.csv")
    total.to_csv(outdir / "score_history.csv")

    notes = []
    for ccy, row in snap.iterrows():
        if row["structure"] != "no trade":
            notes.append(expression.format_note(row, ccy))
    (outdir / "trade_notes.txt").write_text("\n".join(notes), encoding="utf-8")

    # ---- console report
    cols = ["bloc", "score"] + [c for c in PILLAR_WEIGHTS if c in snap.columns] + \
           ["agreement", "n_pillars", "carry_pct", "iv_3m", "ivrv_z", "structure"]
    cols = [c for c in cols if c in snap.columns]
    pd.set_option("display.width", 220, "display.max_columns", 40)

    print(f"\n=== LONGS (top {top}) ===")
    print(snap.head(top)[cols].round(2).to_string())
    print(f"\n=== SHORTS (bottom {top}) ===")
    print(snap.tail(top)[cols].round(2).to_string())
    print(f"\n=== CROSSES (top {top} by conviction) ===")
    print(cross_snap.head(top).round(2).to_string())
    print("\n=== TRADE NOTES ===")
    print("\n".join(notes) if notes else "nothing clears the score threshold today")

    if getattr(__import__("bbg"), "FAILED", None):
        import bbg
        print(f"\n{len(bbg.FAILED)} tickers unresolved — pillars renormalised over "
              f"survivors. Run `python bbg.py --validate` for the list.")
    print(f"\nFiles -> {outdir.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default="output")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    run(a.start, a.end, Path(a.out), a.top)
