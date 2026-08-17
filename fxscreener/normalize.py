"""
Normalisation layer.

Every raw input is mapped to a comparable score in the same units (sigma) via:

    winsorise -> time-series z (rolling)  \
                                           >-- blend -> pillar input
                 cross-sectional z (bloc) /

Why both:
* Time-series z answers "is this rich/cheap versus its own history" — the right
  question for carry, vol and valuation, which have currency-specific levels.
* Cross-sectional z answers "is this the best available today" — the right
  question for a screener that must rank, and it is immune to regime drift in
  the level of the whole complex.
Blending at 50/50 avoids committing to either failure mode. Set XS_TS_BLEND to
0 or 1 in config if you want a pure version.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import TS_Z_MIN_OBS, TS_Z_WINDOW, WINSOR_SIGMA, XS_TS_BLEND


def winsorise(s: pd.Series, sigma: float = WINSOR_SIGMA) -> pd.Series:
    """Clip to +/- sigma standard deviations of the full sample."""
    if s.dropna().empty:
        return s
    mu, sd = s.mean(), s.std()
    if not np.isfinite(sd) or sd == 0:
        return s
    return s.clip(mu - sigma * sd, mu + sigma * sd)


def ts_zscore(
    s: pd.Series,
    window: int = TS_Z_WINDOW,
    min_obs: int = TS_Z_MIN_OBS,
) -> pd.Series:
    """Rolling z-score. Strictly backward-looking — no lookahead."""
    if s.dropna().empty:
        return pd.Series(np.nan, index=s.index, name=s.name)
    mu = s.rolling(window, min_periods=min_obs).mean()
    sd = s.rolling(window, min_periods=min_obs).std()
    # NOTE: min_periods here is already explicit. The failure mode this guards
    # against is pandas' default (min_periods = window), which returns NaN if a
    # single observation in the window is missing — fatal for EM series whose
    # local holidays differ from the panel's union calendar.
    z = (s - mu) / sd.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


def xs_zscore(df: pd.DataFrame, blocs: pd.Series) -> pd.DataFrame:
    """Cross-sectional z within each bloc, computed row by row.

    df    : index = dates, columns = currency codes
    blocs : Series mapping currency -> bloc
    """
    if df.empty:
        return df
    out = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    for bloc in blocs.unique():
        cols = [c for c in df.columns if blocs.get(c) == bloc]
        if len(cols) < 3:          # too few names for a meaningful XS z
            continue
        sub = df[cols]
        mu = sub.mean(axis=1)
        sd = sub.std(axis=1).replace(0, np.nan)
        out[cols] = sub.sub(mu, axis=0).div(sd, axis=0)
    return out.replace([np.inf, -np.inf], np.nan)


def normalise_panel(
    raw: pd.DataFrame,
    blocs: pd.Series,
    blend: float = XS_TS_BLEND,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Full pipeline for one input across the universe.

    raw : index = dates, columns = currency codes, values = raw metric
          already expressed in "long the non-USD currency" orientation.
    """
    if raw.empty:
        return raw
    w = raw.apply(winsorise)
    ts = w.apply(ts_zscore)
    xs = xs_zscore(w, blocs)

    # Where one leg is missing, fall back to the other rather than dropping
    # the observation entirely.
    combined = blend * xs + (1 - blend) * ts
    combined = combined.where(combined.notna(), xs.where(xs.notna(), ts))

    if not higher_is_better:
        combined = -combined
    return combined.clip(-WINSOR_SIGMA, WINSOR_SIGMA)


def combine(
    components: dict[str, pd.DataFrame],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Weighted mean of component panels, renormalised over non-missing inputs.

    This is the mechanism that makes a dead ticker harmless: if a currency is
    missing 2 of 5 inputs, its pillar score is the weighted mean of the 3 that
    resolved, not a NaN and not a silent zero.
    """
    live = {k: v for k, v in components.items() if v is not None and not v.empty}
    if not live:
        return pd.DataFrame()
    weights = weights or {k: 1.0 for k in live}

    idx = sorted(set().union(*[v.index for v in live.values()]))
    cols = sorted(set().union(*[v.columns for v in live.values()]))

    num = pd.DataFrame(0.0, index=idx, columns=cols)
    den = pd.DataFrame(0.0, index=idx, columns=cols)
    for k, v in live.items():
        w = float(weights.get(k, 1.0))
        aligned = v.reindex(index=idx, columns=cols)
        mask = aligned.notna()
        num = num.add(aligned.fillna(0.0) * w, fill_value=0.0)
        den = den.add(mask.astype(float) * w, fill_value=0.0)

    return (num / den.replace(0, np.nan))


def pct_rank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank, for display alongside z-scores."""
    return s.rank(pct=True) * 100
