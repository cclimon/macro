"""
Score aggregation.

Equal-weight composite across the seven pillars, plus cross (RV) scores and
conviction diagnostics.

Conviction is REPORTED, not applied. A dispersion penalty or an agreement
multiplier would be a free parameter fitted on the same history used to build
the pillars, which is exactly how a screener starts looking clever in sample
and stops working in production. Read the agreement column, do not let it
silently rescale the rank.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import CROSSES, PILLAR_WEIGHTS, UNIVERSE
from pillars import BLOCS, MarketData, PILLAR_FUNCS

log = logging.getLogger(__name__)

# Days a pillar score may be carried forward across a market holiday.
PILLAR_FFILL_LIMIT = 5


def build_pillars(md: MarketData) -> dict[str, pd.DataFrame]:
    out = {}
    for name, fn in PILLAR_FUNCS.items():
        try:
            panel = fn(md)
            if panel is None or panel.empty:
                log.warning("pillar %s produced no data", name)
                continue
            out[name] = panel
            log.info("pillar %-16s %d dates x %d ccy", name, *panel.shape)
        except Exception as e:                                # noqa: BLE001
            log.error("pillar %s failed: %s", name, e)
    return out


def composite(pillars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight composite, renormalised over pillars that resolved.

    Pillars are forward-filled by a few days first. Without it, a currency
    whose market was shut has fewer pillars contributing on that date, and
    because the weights renormalise over what survives, its score is quietly
    computed on a different basis from everyone else's — a composite of three
    pillars is not comparable to a composite of seven.
    """
    if not pillars:
        return pd.DataFrame()
    idx = sorted(set().union(*[p.index for p in pillars.values()]))
    cols = sorted(set().union(*[p.columns for p in pillars.values()]))

    num = pd.DataFrame(0.0, index=idx, columns=cols)
    den = pd.DataFrame(0.0, index=idx, columns=cols)
    for name, p in pillars.items():
        w = PILLAR_WEIGHTS.get(name, 0.0)
        a = p.reindex(index=idx, columns=cols).ffill(limit=PILLAR_FFILL_LIMIT)
        num += a.fillna(0.0) * w
        den += a.notna().astype(float) * w
    return num / den.replace(0, np.nan)


def conviction(pillars: dict[str, pd.DataFrame], total: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Agreement = share of live pillars sharing the composite's sign.
    Dispersion = cross-pillar standard deviation."""
    if not pillars or total.empty:
        return {}
    # Must apply the SAME forward-fill as composite(), or the diagnostics
    # describe a different cross-section from the one that was actually scored.
    stack = {k: v.reindex_like(total).ffill(limit=PILLAR_FFILL_LIMIT)
             for k, v in pillars.items()}
    arr = np.dstack([v.values for v in stack.values()])

    sgn_total = np.sign(total.values)[:, :, None]
    agree = np.nansum(np.sign(arr) == sgn_total, axis=2)
    live = np.sum(~np.isnan(arr), axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        agreement = np.where(live > 0, agree / live, np.nan)

    live_df = pd.DataFrame(live, index=total.index, columns=total.columns)
    last = live_df.ffill().iloc[-1]
    thin = last[last < last.max() - 1]
    if not thin.empty:
        log.warning("scored on fewer pillars than peers: %s",
                    ", ".join(f"{c}={int(n)}/{int(last.max())}" for c, n in thin.items()))

    with np.errstate(invalid="ignore"):
        disp = np.nanstd(arr, axis=2)

    return {
        "agreement": pd.DataFrame(agreement, index=total.index, columns=total.columns),
        "dispersion": pd.DataFrame(disp, index=total.index, columns=total.columns),
        "n_pillars": pd.DataFrame(live, index=total.index, columns=total.columns),
    }


def cross_scores(total: pd.DataFrame) -> pd.DataFrame:
    """Cross score = score(base) - score(term).

    Derived from the same USD-leg scores as the outrights, so an RV
    recommendation can never contradict the outright ranking.
    """
    out = {}
    for base, term in CROSSES:
        if base in total.columns and term in total.columns:
            out[f"{base}{term}"] = total[base] - total[term]
    return pd.DataFrame(out)


def _asof_row(panel: pd.DataFrame, cols, asof) -> pd.Series:
    """Latest available value per currency at or before `asof`.

    Always forward-fills. Selecting the raw row at `asof` drops any currency
    whose market was shut that day, which silently removes the EM names from
    the cross-section on their own local holidays.
    """
    if panel is None or panel.empty:
        return pd.Series(np.nan, index=cols)
    sub = panel.reindex(columns=cols).ffill()
    sub = sub.loc[:asof]
    if sub.empty:
        return pd.Series(np.nan, index=cols)
    return sub.iloc[-1]


def snapshot(
    md: MarketData,
    pillars: dict[str, pd.DataFrame],
    total: pd.DataFrame,
    conv: dict[str, pd.DataFrame],
    asof: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Latest cross-section as a ranked table."""
    if total.empty:
        return pd.DataFrame()
    asof = asof or total.dropna(how="all").index[-1]
    row = total.ffill().loc[asof]

    df = pd.DataFrame({"score": row})
    df["bloc"] = [BLOCS.get(c) for c in df.index]
    for name in PILLAR_WEIGHTS:
        if name in pillars:
            df[name] = _asof_row(pillars[name], df.index, asof)
    for k, v in conv.items():
        df[k] = _asof_row(v, df.index, asof)

    # Raw context columns for the trade note
    for label, panel in [("carry_pct", md.carry), ("iv_3m", md.atm_vol),
                         ("rv_3m", md.rv_3m), ("rr25", md.rr25)]:
        df[label] = _asof_row(panel, df.index, asof)

    df["rank_all"] = df["score"].rank(ascending=False)
    df["rank_bloc"] = df.groupby("bloc")["score"].rank(ascending=False)
    df["deliverable"] = [next(p.deliverable for p in UNIVERSE if p.ccy == c) for c in df.index]
    df.index.name = "ccy"
    return df.sort_values("score", ascending=False)
