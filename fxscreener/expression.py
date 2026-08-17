"""
Trade expression.

The score says WHAT to be long. This module says HOW to express it, using the
two surface diagnostics deliberately excluded from the directional score:

  IV/RV z    z-score of ln(implied / realised) — is optionality cheap or rich
  RR25 z     25d risk reversal — is the market already paying up for your side

The logic is a lookup table, not a model. Four cases, and the interesting one
is cheap vol with a strong score: that is where a directional view is worth
owning as convexity rather than as spot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from normalize import ts_zscore

SCORE_STRONG = 0.75      # sigma; below this the name is a watch, not a trade
IVRV_RICH = 0.75
IVRV_CHEAP = -0.50
SKEW_EXTREME = 1.0


def ivrv_z(atm_vol: pd.DataFrame, rv: pd.DataFrame) -> pd.DataFrame:
    """Time-series z of log(implied / realised)."""
    if atm_vol.empty or rv.empty:
        return pd.DataFrame()
    ratio = np.log(atm_vol / rv.reindex_like(atm_vol).replace(0, np.nan))
    return ratio.apply(ts_zscore)


def skew_z(rr25: pd.DataFrame, pairs: dict) -> pd.DataFrame:
    """Skew z oriented so positive = market pays up for the currency's upside."""
    if rr25.empty:
        return pd.DataFrame()
    sgn = pd.Series({c: -pairs[c].sign for c in rr25.columns})
    return rr25.mul(sgn, axis=1).apply(ts_zscore)


def _structure(score: float, ivrv: float, skew: float, deliverable: bool) -> tuple[str, str]:
    """Return (structure, rationale)."""
    if not np.isfinite(score) or abs(score) < SCORE_STRONG:
        return "no trade", "score inside the noise band"

    long_ccy = score > 0
    d = "long" if long_ccy else "short"
    fwd = "NDF" if not deliverable else "forward"

    # Skew working against you = your side of the wing is being given away.
    skew_favours = (skew < -SKEW_EXTREME) if long_ccy else (skew > SKEW_EXTREME)

    if not np.isfinite(ivrv):
        return f"{d} 3m {fwd}", "no vol surface; express in the forward"

    if ivrv <= IVRV_CHEAP:
        if skew_favours:
            return (f"{d} 3m 25d vanilla ({'call' if long_ccy else 'put'})",
                    "implied cheap to realised and skew is on the wrong side for "
                    "the crowd — own the convexity outright")
        return (f"{d} 3m call spread" if long_ccy else f"{d} 3m put spread",
                "implied cheap to realised — buy the direction, cap the premium")

    if ivrv >= IVRV_RICH:
        if skew_favours:
            return ("risk reversal, financed",
                    "implied rich and skew pays you to take the position — sell the "
                    "far wing to fund your strike")
        return ("ERKO / reverse knock-out" if not deliverable else "seagull",
                "implied rich — monetise the premium, accept the barrier or the "
                "capped tail rather than paying up for vanilla convexity")

    return (f"{d} 3m {fwd}", "vol fairly priced — no edge in the option, take the carry")


def build(snap: pd.DataFrame, ivrv: pd.DataFrame, skw: pd.DataFrame,
          asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Attach expression columns to a snapshot table."""
    out = snap.copy()
    if not ivrv.empty:
        asof = asof or ivrv.dropna(how="all").index[-1]
        out["ivrv_z"] = ivrv.reindex(columns=out.index).ffill().loc[:asof].iloc[-1]
    else:
        out["ivrv_z"] = np.nan
    if not skw.empty:
        out["skew_z"] = skw.reindex(columns=out.index).ffill().loc[:asof].iloc[-1]
    else:
        out["skew_z"] = np.nan

    structures, rationales = [], []
    for c, r in out.iterrows():
        s, why = _structure(r["score"], r.get("ivrv_z", np.nan),
                            r.get("skew_z", np.nan), bool(r.get("deliverable", True)))
        structures.append(s)
        rationales.append(why)
    out["structure"] = structures
    out["expression_note"] = rationales

    # Vol-scaled sizing: equal risk contribution, normalised to the median name.
    if "iv_3m" in out.columns:
        inv = 1.0 / out["iv_3m"].replace(0, np.nan)
        out["size_weight"] = (inv / inv.median()).round(2)
    return out


def format_note(row: pd.Series, ccy: str) -> str:
    """One-line trade note in desk register."""
    side = "long" if row["score"] > 0 else "short"
    return (
        f"{ccy} {side} | score {row['score']:+.2f} (rank {int(row['rank_all'])}) | "
        f"carry {row.get('carry_pct', float('nan')):+.2f}% | "
        f"3m IV {row.get('iv_3m', float('nan')):.1f} "
        f"(IV/RV z {row.get('ivrv_z', float('nan')):+.1f}) | "
        f"agreement {row.get('agreement', float('nan')):.0%} | "
        f"{row['structure']} — {row['expression_note']}"
    )
