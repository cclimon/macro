"""
Data integrity check.

Run this before every screener run and after any config change. It verifies
that what comes off the terminal is what the model thinks it is, and it fails
loudly rather than producing a number.

    python check_data.py                 # full check
    python check_data.py --section fwd   # one section
    python check_data.py --out check.csv

Sections
--------
  spot   spot levels against reference, staleness
  fwd    forward reconstruction: raw -> points -> outright -> carry
  carry  carry cross-checked against the 2y rate differential
  vol    ATM vol, risk reversal and butterfly plausibility
  cds    CDS spreads: levels in bp, ordering across sovereigns
  cover  per-currency and per-pillar input coverage

Every check emits PASS, WARN or FAIL. FAIL means the number is wrong, not
merely surprising: a FAIL should stop you running the screener.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import numpy as np
import pandas as pd

import bbg
from config import (
    CARRY_SANITY_MAX, CARRY_SANITY_MIN, CARRY_TENOR_DAYS, FWD_CONVENTIONS,
    UNIVERSE, VOL_TENOR,
)

PAIRS = {p.ccy: p for p in UNIVERSE}
START = "2024-01-01"

# --------------------------------------------------------------------------
# Reference values, from the convention table supplied by the desk.
# Used to catch a ticker returning the wrong instrument entirely — a spot
# series that resolves but sits 30% away from reference is a different
# security, not a market move.
# --------------------------------------------------------------------------
REF_SPOT = {
    "EUR": 1.154, "JPY": 159.24, "GBP": 1.351, "CHF": 0.812, "CAD": 1.392,
    "AUD": 0.707, "NZD": 0.586, "NOK": 9.475, "SEK": 9.554,
    "BRL": 5.164, "MXN": 17.068, "CLP": 911.8, "COP": 3125.46,
    "PLN": 3.731, "HUF": 315.8, "CZK": 21.003, "ZAR": 16.142,
    "TRY": 47.757, "ILS": 2.981,
    "KRW": 1416.90, "CNH": 6.746, "TWD": 32.252, "THB": 33.16,
}
SPOT_TOL = 0.30            # 30% band; catches wrong instrument, not market moves

# Expected sign of annualised carry for being LONG the non-USD currency.
# Derived from where policy rates sit versus the US, not from the data, so it
# is an independent check rather than a restatement.
EXPECTED_CARRY_SIGN = {
    "EUR": -1, "JPY": -1, "GBP": 0, "CHF": -1, "CAD": -1, "AUD": -1,
    "NZD": -1, "NOK": 0, "SEK": -1,
    "BRL": +1, "MXN": +1, "CLP": 0, "COP": +1, "PEN": 0,
    "PLN": +1, "HUF": +1, "CZK": 0, "ZAR": +1, "TRY": +1, "ILS": 0,
    "KRW": -1, "CNH": -1, "TWD": -1, "THB": -1, "INR": +1, "IDR": +1,
}   # 0 = either sign acceptable

STALE_LIMIT_DAYS = 7
# Below this, carry is indistinguishable from zero. Legitimate for a tightly
# aligned pair, but far more often a transcription or divisor problem.
NEAR_ZERO_CARRY = 0.15
VOL_MIN, VOL_MAX = 1.5, 80.0
CDS_MIN, CDS_MAX = 3.0, 3000.0

RESULTS: list[dict] = []


def record(section, item, status, detail=""):
    RESULTS.append({"section": section, "item": item, "status": status,
                    "detail": detail})
    colour = {"PASS": "  ", "WARN": "! ", "FAIL": "**"}[status]
    print(f"{colour}{status:<5} {section:<6} {item:<26} {detail}", flush=True)


def _age(s: pd.Series) -> int | None:
    if s.empty:
        return None
    return (dt.date.today() - s.index[-1].date()).days


# ==========================================================================
# spot
# ==========================================================================

def check_spot() -> dict[str, pd.Series]:
    print("\n--- SPOT ---")
    out = {}
    for ccy, p in PAIRS.items():
        s = bbg.history(p.ticker, start=START)
        if s.empty:
            record("spot", ccy, "FAIL", f"{p.ticker} returned nothing")
            continue
        out[ccy] = s
        last, age = float(s.iloc[-1]), _age(s)

        ref = REF_SPOT.get(ccy)
        if ref is None:
            record("spot", ccy, "WARN", f"{last:.4f}, no reference to compare")
        elif abs(last / ref - 1.0) > SPOT_TOL:
            record("spot", ccy, "FAIL",
                   f"{last:.4f} vs reference {ref:.4f} — likely wrong instrument")
        elif age is not None and age > STALE_LIMIT_DAYS:
            record("spot", ccy, "FAIL", f"{last:.4f} but stale, last {s.index[-1].date()}")
        else:
            record("spot", ccy, "PASS", f"{last:.4f}  ({age}d old)")
    return out


# ==========================================================================
# forwards — the core reconstruction
# ==========================================================================

def check_fwd(spots: dict[str, pd.Series]) -> pd.DataFrame:
    """Rebuild every forward as SPOT + POINTS and show the full arithmetic.

    The table it prints is the artefact to eyeball: raw quote, divisor, real
    points, outright, implied carry. If a divisor is wrong the outright will
    look wrong to anyone who knows the pair, which is the point of printing it.
    """
    print("\n--- FORWARDS (FWD = SPOT + POINTS) ---")
    rows = []
    for ccy, (tk, div, mode) in FWD_CONVENTIONS.items():
        p = PAIRS.get(ccy)
        if p is None or ccy not in spots:
            record("fwd", ccy, "FAIL", "no spot to reconstruct against")
            continue
        raw = bbg.history(tk, start=START)
        if raw.empty:
            record("fwd", ccy, "FAIL", f"{tk} returned nothing")
            continue

        spot = spots[ccy].reindex(raw.index).ffill()
        if mode == "outright":
            # Back out points so every currency is expressed the same way.
            pts = raw - spot
            note = "outright ticker, points derived"
        else:
            pts = raw / div
            note = ""
        fwd = spot + pts

        s0, p0, f0 = float(spot.iloc[-1]), float(pts.iloc[-1]), float(fwd.iloc[-1])
        carry = (f0 / s0 - 1.0) * (365.0 / CARRY_TENOR_DAYS) * 100.0 * (-p.sign)
        age = _age(raw)

        rows.append({
            "ccy": ccy, "ticker": tk, "mode": mode, "divisor": div,
            "raw": round(float(raw.iloc[-1]), 4), "spot": round(s0, 4),
            "real_points": round(p0, 6), "outright": round(f0, 4),
            "carry_pct": round(carry, 2), "days_old": age,
        })

        # --- verdicts
        if age is not None and age > STALE_LIMIT_DAYS:
            record("fwd", ccy, "FAIL", f"stale, last {raw.index[-1].date()}")
        elif abs(carry) > CARRY_SANITY_MAX:
            record("fwd", ccy, "FAIL",
                   f"carry {carry:+.2f}% far too large — divisor {div:,.0f} "
                   f"is wrong by a factor of ten or more")
        elif abs(carry) < NEAR_ZERO_CARRY:
            record("fwd", ccy, "WARN",
                   f"carry {carry:+.2f}% is near zero — raw quote {float(raw.iloc[-1]):.2f} "
                   f"may be missing a digit, or the divisor is too large")
        else:
            exp = EXPECTED_CARRY_SIGN.get(ccy, 0)
            if exp and np.sign(carry) != exp:
                # WARN not FAIL: the expectation is a prior about where policy
                # rates sit, and the forward is live market data. If they
                # disagree the prior is the more likely thing to be stale.
                record("fwd", ccy, "WARN",
                       f"carry {carry:+.2f}%, expected "
                       f"{'positive' if exp > 0 else 'negative'} — verify, but "
                       f"the market may simply have moved past the assumption")
            else:
                record("fwd", ccy, "PASS",
                       f"pts {p0:+.6f} -> outright {f0:.4f}, carry {carry:+.2f}% {note}")

    df = pd.DataFrame(rows).set_index("ccy") if rows else pd.DataFrame()
    if not df.empty:
        print("\nReconstruction table — check the outright column against your screens:")
        print(df[["ticker", "divisor", "raw", "spot", "real_points",
                  "outright", "carry_pct"]].to_string())
    return df


# ==========================================================================
# carry cross-check against rate differentials
# ==========================================================================

def check_carry_vs_rates(fwd: pd.DataFrame) -> None:
    """Independent check: forward-implied carry versus the policy differential.

    Covered interest parity says these should agree to within the basis. A
    large gap means either a divisor error the plausibility band was too loose
    to catch, or a genuine CIP dislocation — both worth knowing about, and the
    check cannot tell you which, so it warns rather than fails.
    """
    print("\n--- CARRY vs RATE DIFFERENTIAL (CIP cross-check) ---")
    if fwd.empty:
        record("carry", "all", "FAIL", "no forward table")
        return
    usd = bbg.history("FEDL01 Index", start=START)
    if usd.empty:
        record("carry", "usd_leg", "WARN", "no USD policy rate; skipping")
        return
    u = float(usd.iloc[-1])
    record("carry", "usd_policy", "PASS", f"{u:.2f}%")

    for ccy, r in fwd.iterrows():
        p = PAIRS[ccy]
        pol = bbg.history(p.swap2y, start=START) if p.swap2y else pd.Series(dtype=float)
        if pol.empty:
            record("carry", ccy, "WARN", f"carry {r.carry_pct:+.2f}%, no rate to compare")
            continue
        diff = float(pol.iloc[-1]) - u
        gap = r.carry_pct - diff
        if abs(gap) > 5.0:
            record("carry", ccy, "WARN",
                   f"carry {r.carry_pct:+.2f}% vs rate diff {diff:+.2f}% "
                   f"(gap {gap:+.2f}) — divisor or large basis")
        else:
            record("carry", ccy, "PASS",
                   f"carry {r.carry_pct:+.2f}% vs rate diff {diff:+.2f}%")


# ==========================================================================
# vol
# ==========================================================================

def check_vol() -> None:
    print("\n--- VOL SURFACE ---")
    for ccy, p in PAIRS.items():
        tks = bbg.vol_tickers(p.vol_root, VOL_TENOR)
        atm = bbg.history(tks["atm"], start=START)
        if atm.empty:
            record("vol", ccy, "FAIL", f"{tks['atm']} returned nothing")
            continue
        v = float(atm.iloc[-1])
        age = _age(atm)
        if not (VOL_MIN <= v <= VOL_MAX):
            record("vol", ccy, "FAIL", f"ATM {v:.2f} outside [{VOL_MIN}, {VOL_MAX}]")
        elif age is not None and age > STALE_LIMIT_DAYS:
            record("vol", ccy, "FAIL", f"ATM {v:.2f} stale, last {atm.index[-1].date()}")
        else:
            rr = bbg.history(tks["rr25"], start=START)
            bf = bbg.history(tks["bf25"], start=START)
            extra = ""
            if not rr.empty:
                extra += f" RR {float(rr.iloc[-1]):+.2f}"
            if not bf.empty:
                b = float(bf.iloc[-1])
                extra += f" BF {b:+.2f}"
                if b < 0:
                    extra += " (negative BF is unusual)"
            record("vol", ccy, "PASS", f"ATM {v:.2f}{extra}")


# ==========================================================================
# cds
# ==========================================================================

def check_cds() -> None:
    print("\n--- CDS ---")
    levels = {}
    for ccy, p in PAIRS.items():
        if not p.cds5y:
            continue
        s = bbg.history(p.cds5y, start=START)
        if s.empty:
            record("cds", ccy, "FAIL", f"{p.cds5y} returned nothing — check CDS_SUFFIX")
            continue
        v = float(s.iloc[-1])
        levels[ccy] = v
        if not (CDS_MIN <= v <= CDS_MAX):
            record("cds", ccy, "FAIL",
                   f"{v:.1f} outside [{CDS_MIN}, {CDS_MAX}] — price rather than spread?")
        else:
            record("cds", ccy, "PASS", f"{v:.1f} bp")

    # Ordering check: a spread series that resolves but ranks Turkey inside
    # Czech is returning something other than a sovereign credit spread.
    if {"TRY", "CZK"} <= levels.keys() and levels["TRY"] < levels["CZK"]:
        record("cds", "ordering", "FAIL",
               f"TRY {levels['TRY']:.0f} < CZK {levels['CZK']:.0f} — implausible")
    elif len(levels) >= 5:
        record("cds", "ordering", "PASS", f"{len(levels)} sovereigns, ordering sane")


# ==========================================================================
# coverage
# ==========================================================================

def check_coverage(spots, fwd) -> None:
    print("\n--- PILLAR INPUT COVERAGE ---")
    rows = []
    for ccy, p in PAIRS.items():
        have = {
            "spot": ccy in spots,
            "fwd": ccy in fwd.index if not fwd.empty else False,
            "vol": not bbg.history(bbg.vol_tickers(p.vol_root, VOL_TENOR)["atm"],
                                   start=START).empty,
            "rr": not bbg.history(bbg.vol_tickers(p.vol_root, VOL_TENOR)["rr25"],
                                  start=START).empty,
            "cds": bool(p.cds5y) and not bbg.history(p.cds5y, start=START).empty,
            "cpi": bool(p.cpi_yoy) and not bbg.history(p.cpi_yoy, start=START).empty,
            "cesi": bool(p.cesi) and not bbg.history(p.cesi, start=START).empty,
            "reer": bool(p.reer) and not bbg.history(p.reer, start=START).empty,
        }
        rows.append({"ccy": ccy, **{k: ("Y" if v else "-") for k, v in have.items()},
                     "n": sum(have.values())})
    df = pd.DataFrame(rows).set_index("ccy")
    print(df.to_string())
    thin = df[df["n"] < 4]
    for ccy in thin.index:
        record("cover", ccy, "WARN", f"only {int(df.loc[ccy,'n'])}/8 inputs")
    if thin.empty:
        record("cover", "all", "PASS", "every currency has 4+ inputs")


# ==========================================================================

SECTIONS = ("spot", "fwd", "carry", "vol", "cds", "cover")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=SECTIONS, default=None)
    ap.add_argument("--out", default="data_check.csv")
    a = ap.parse_args()

    if not bbg.preflight():
        raise SystemExit("Bloomberg unreachable.")

    want = {a.section} if a.section else set(SECTIONS)
    spots, fwd = {}, pd.DataFrame()

    if want & {"spot", "fwd", "carry", "cover"}:
        spots = check_spot()
    if want & {"fwd", "carry", "cover"}:
        fwd = check_fwd(spots)
    if "carry" in want:
        check_carry_vs_rates(fwd)
    if "vol" in want:
        check_vol()
    if "cds" in want:
        check_cds()
    if "cover" in want:
        check_coverage(spots, fwd)

    res = pd.DataFrame(RESULTS)
    res.to_csv(a.out, index=False)
    counts = res["status"].value_counts()
    print("\n" + "=" * 62)
    print("  ".join(f"{k}: {v}" for k, v in counts.items()))
    fails = res[res.status == "FAIL"]
    if not fails.empty:
        print(f"\n{len(fails)} FAILURES — do not run the screener until these are fixed:")
        print(fails[["section", "item", "detail"]].to_string(index=False))
    print(f"\nFull report -> {a.out}")
    sys.exit(1 if not fails.empty else 0)
