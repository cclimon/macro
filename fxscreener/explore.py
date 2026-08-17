"""
FX Screener — interactive exploration module.

The single entry-point for drilling into the screener's internals without
re-running the full pipeline. Every number produced here is guaranteed to
match run_screener.py: the same MarketData, the same normalisation pipeline,
the same pillar functions, the same combine() logic.

===============================================================================
QUICK START
===============================================================================

    import explore as ex

    # 1. Load once — reads from the parquet cache, computes all 7 pillars.
    #    Typical time: 2-5s (cache warm), 60-90s (cold Bloomberg fetch).
    ex.load()

    # 2. Composite snapshot — today's full ranked table.
    #    Identical to the output of run_screener.py.
    ex.snapshot()

    # 3. Drill into one pillar — ranked by that pillar, sub-signals appended.
    ex.snapshot("carry")
    ex.snapshot("trend", bloc="G10")
    ex.snapshot("fundamental", asof="2026-06-30")

    # 4. Sub-signals — the normalised inputs to combine(), in sigma units.
    ex.components("trend")                  # dict of full-history DataFrames
    ex.components("carry", asof="2026-08-14")  # dict of cross-section Series

    # 5. Raw market data — unscored, unscaled.
    ex.raw("carry")                         # annualised carry %, all pairs
    ex.raw("carry", "BRL")                  # BRL carry time series
    ex.raw("spot", "EUR")                   # EURUSD spot
    ex.raw("atm_vol", "JPY")               # USDJPY 3m ATM implied vol
    ex.raw("regime", "vix")                # VIX index
    ex.raw("cds5y", "ZAR")                 # South Africa 5y CDS

    # 6. Score time series — pillar or composite score history.
    ex.history("carry", "BRL")             # BRL carry score over time
    ex.history("composite")                # all composite scores, full history
    ex.history("trend").tail(60)           # trend panel, last 60 days

    # 7. Cross (RV) scores — base minus term, today.
    ex.crosses()
    ex.crosses(asof="2026-06-30")

    # 8. Export to Excel — one sheet per pillar + composite + carry diagnostic.
    ex.to_excel()
    ex.to_excel("drill_20260814.xlsx", asof="2026-08-14")

===============================================================================
COHERENCE GUARANTEES
===============================================================================

* MarketData is loaded once and shared across all calls in a session.
  Reloading mid-session (ex.load(reload=True)) resets everything cleanly.

* Pillar scores are computed via the exact same functions (pillars.py) and
  the exact same normalisation pipeline (normalize.py) as run_screener.py.
  There is no separate "explore" code path.

* Sub-components (ex.components()) are the normalised inputs captured
  immediately before combine() is called inside each pillar function.
  They are in sigma units and directly comparable across signals. They are
  what actually went into the blended pillar score — not a reconstruction.

* Snapshots use score._asof_row(), which forward-fills across market
  holidays. An EM currency whose market was shut on a given date is never
  silently dropped from the cross-section.

* The carry sanity check (config.CARRY_SANITY_MAX / MIN) runs at load time.
  If a forward convention is wrong, a warning is printed before any scores
  are shown. Do not ignore it — a divisor error in FWD_CONVENTIONS is a
  ranking error, not a data-quality nuisance.

===============================================================================
PILLAR SUB-COMPONENTS REFERENCE
===============================================================================

  trend           m3, m6, m12, dist_200d, macd_hist
  value           steer, beer
  carry           carry_level, carry_slope
  risk_adj_carry  carry_iv, carry_downside, carry_cds
  vol             iv_change, rv_change, term_slope
  positioning     cot_level, cot_change, skew, skew_change
  fundamental     rate_diff_1m, rate_diff_3m, cesi_diff,
                  cds_change, tot_mom, ca_gdp

===============================================================================
RAW MARKET DATA ATTRIBUTES
===============================================================================

  Price:       spot, fwd_points
  Vol surface: atm_vol, rr25, bf25, atm_1m
  Rates/macro: carry (ann. %, fwd-implied), swap2y, cpi, cds5y,
               reer, tot, cot, cesi, ca_gdp
  Derived:     rv_3m, rv_1m, ret (daily log returns), risk_beta
  Regime:      regime  — use ccy= to select column (vix, dxy, oil, ...)
  USD legs:    usd_swap2y, usd_cpi, usd_cesi  (Series, ccy= ignored)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import bbg
import expression as _expr
import pillars as _pillars_mod
import score as _score
from config import UNIVERSE
from pillars import BLOCS, PILLAR_FUNCS, MarketData

log = logging.getLogger(__name__)

# Convenience alias: "EM" expands to three blocs
_BLOC_ALIASES: dict[str, list[str]] = {
    "EM": ["LATAM", "CEEMEA", "APAC"],
}

_RAW_ATTRS = frozenset({
    "spot", "fwd_points", "carry", "atm_vol", "rr25", "bf25", "atm_1m",
    "swap2y", "cpi", "cds5y", "reer", "tot", "cot", "cesi", "ca_gdp",
    "rv_3m", "rv_1m", "risk_beta", "ret", "regime",
    "usd_swap2y", "usd_cpi", "usd_cesi",
})


# ===========================================================================
# Session
# ===========================================================================

class Session:
    """
    Loaded market data + computed pillars, all in memory.

    Construct via explore.load(), not directly.

    Attributes
    ----------
    md : MarketData
        The full market data container (spot, vol, rates, etc.)
    pillars : dict[str, pd.DataFrame]
        Normalised pillar scores. Keys: "trend", "value", "carry",
        "risk_adj_carry", "vol", "positioning", "fundamental".
        Each DataFrame: index=dates, columns=currency codes, values=sigma.
    total : pd.DataFrame
        Composite score (equal-weight average across resolved pillars).
    conv : dict[str, pd.DataFrame]
        Conviction metrics: "agreement", "dispersion", "n_pillars".
    """

    def __init__(
        self,
        md: MarketData,
        pillars: dict[str, pd.DataFrame],
        total: pd.DataFrame,
        conv: dict[str, pd.DataFrame],
        components: dict[str, dict[str, pd.DataFrame]],
    ):
        self.md = md
        self.pillars = pillars
        self.total = total
        self.conv = conv
        self._components = components

    @property
    def available_dates(self) -> pd.DatetimeIndex:
        """All dates for which a composite score exists."""
        return self.total.dropna(how="all").index

    @property
    def latest(self) -> pd.Timestamp:
        """Latest date with a valid composite score."""
        return self.available_dates[-1]

    def __repr__(self) -> str:
        n = len(self.pillars)
        nc = len(self.total.columns)
        failed = len(bbg.FAILED)
        d0, d1 = self.available_dates[0].date(), self.latest.date()
        s = f"<Session pillars={n}/7  currencies={nc}  dates={d0}->{d1}"
        if failed:
            s += f"  FAILED={failed}"
        return s + ">"


# ===========================================================================
# Module-level session (lazy)
# ===========================================================================

_session: Optional[Session] = None


def load(
    start: str = "2012-01-01",
    end: Optional[str] = None,
    reload: bool = False,
) -> Session:
    """
    Load market data from the parquet cache and compute all seven pillars.

    Must be called before any other function. Subsequent calls within the
    same Python session are instant (returns the cached Session) unless
    reload=True is passed.

    Parameters
    ----------
    start : str
        History start in ISO format. Default "2012-01-01".
        The normalisation window (TS_Z_WINDOW = 1260 days) needs ~5y of
        history to produce stable z-scores. A shorter start date produces
        scores but they will have fewer observations in the rolling window.
    end : str or None
        History end in ISO format. None = today.
    reload : bool
        Force a full reload. Use this if you change config.py or want
        a different time window.

    Returns
    -------
    Session

    Examples
    --------
    >>> ex.load()
    <Session pillars=7/7  currencies=33  dates=2012-01-03->2026-08-14>

    >>> ex.load(start="2015-01-01", reload=True)   # rebuild from 2015
    >>> ex.load(end="2025-12-31", reload=True)      # fixed end date
    """
    global _session
    if _session is not None and not reload:
        return _session

    log.info("Loading MarketData (start=%s end=%s)", start, end or "today")
    md = MarketData(start=start, end=end)

    # Carry sanity — a wrong divisor in FWD_CONVENTIONS is a ranking error.
    # Warn loudly. The full run (run_screener.py) raises; here we warn so
    # that exploration is still possible while the convention is being fixed.
    bad = md.sanity_check_carry(raise_on_fail=False)
    if not bad.empty:
        log.warning(
            "CARRY SANITY FAILED for: %s\n"
            "  Check divisors in config.FWD_CONVENTIONS. "
            "  Scores for these currencies are unreliable until fixed.",
            ", ".join(bad.index.tolist()),
        )

    # Build all pillars, capturing sub-components in the same pass.
    pillar_scores: dict[str, pd.DataFrame] = {}
    pillar_components: dict[str, dict[str, pd.DataFrame]] = {}

    for name, fn in PILLAR_FUNCS.items():
        try:
            scored, comps = _run_with_components(fn, md)
            if scored is None or scored.empty:
                log.warning("pillar '%s' produced no data", name)
                continue
            pillar_scores[name] = scored
            pillar_components[name] = comps
            log.info("pillar %-16s %d dates × %d ccy", name, *scored.shape)
        except Exception as e:  # noqa: BLE001
            log.error("pillar '%s' failed: %s", name, e)

    if not pillar_scores:
        raise RuntimeError(
            "No pillars resolved. Run `python bbg.py --validate` to "
            "identify failing tickers."
        )

    total = _score.composite(pillar_scores)
    conv = _score.conviction(pillar_scores, total)
    _session = Session(md, pillar_scores, total, conv, pillar_components)

    failed = len(bbg.FAILED)
    if failed:
        log.warning(
            "%d tickers unresolved — pillars renormalised over survivors. "
            "Run `python bbg.py --validate` for details.",
            failed,
        )

    print(_session)
    return _session


# ===========================================================================
# Internal helpers
# ===========================================================================

def _run_with_components(
    fn,
    md: MarketData,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Run a pillar function and capture the normalised sub-signals passed to
    combine() before blending.

    Temporarily replaces normalize.combine with a capturing wrapper. The
    pillar functions themselves are not modified in any way.
    """
    # pillars.py does `from normalize import combine`, so the binding lives in
    # the pillars module namespace. Patching normalize.combine has no effect
    # on already-imported references; we must patch pillars.combine directly.
    captured: dict[str, pd.DataFrame] = {}
    original = _pillars_mod.combine

    def _capturing(components, weights=None):
        for k, v in components.items():
            if v is not None and not v.empty:
                captured[k] = v.copy()
        return original(components, weights)

    _pillars_mod.combine = _capturing
    try:
        result = fn(md)
    finally:
        _pillars_mod.combine = original  # always restore, even on exception

    return result, captured


def _require_session() -> Session:
    if _session is None:
        raise RuntimeError(
            "No session loaded. Call ex.load() first.\n"
            "  import explore as ex\n"
            "  ex.load()"
        )
    return _session


def _parse_asof(asof, s: Session) -> pd.Timestamp:
    """Resolve asof to the nearest available scored date at or before it."""
    if asof is None:
        return s.latest
    ts = pd.Timestamp(asof)
    avail = s.available_dates
    if ts < avail[0]:
        raise ValueError(
            f"asof={ts.date()} is before the session start ({avail[0].date()}). "
            f"Reload with ex.load(start='earlier-date', reload=True)."
        )
    valid = avail[avail <= ts]
    if valid.empty:
        raise ValueError(f"No scored data on or before {ts.date()}.")
    resolved = valid[-1]
    if resolved != ts:
        log.debug("asof %s -> resolved to %s (nearest scored date)", ts.date(), resolved.date())
    return resolved


def _filter_bloc(df: pd.DataFrame, bloc: Optional[str]) -> pd.DataFrame:
    if bloc is None:
        return df
    blocs_expanded = _BLOC_ALIASES.get(bloc.upper(), [bloc])
    keep = [c for c in df.index if BLOCS.get(c) in blocs_expanded]
    if not keep:
        raise ValueError(
            f"No currencies found for bloc='{bloc}'. "
            f"Valid blocs: G10, LATAM, CEEMEA, APAC, EM."
        )
    return df.loc[keep]


# ===========================================================================
# Public API
# ===========================================================================

def snapshot(
    pillar: Optional[str] = None,
    asof: Optional[str] = None,
    bloc: Optional[str] = None,
) -> pd.DataFrame:
    """
    Cross-sectional ranked table at a point in time.

    Parameters
    ----------
    pillar : str or None
        None  -> composite view. Identical to run_screener.py output:
                columns include score, all pillar scores, agreement,
                n_pillars, carry_pct, iv_3m, rv_3m, rr25, structure.
        str   -> drill into one pillar. Table is sorted by that pillar's
                score and sub-component columns are appended so you can
                see exactly what drove each currency's rank.
                Valid values: "trend", "value", "carry", "risk_adj_carry",
                              "vol", "positioning", "fundamental".
    asof : str or None
        Snapshot date in ISO format (e.g. "2026-06-30"). None = latest.
        If the date falls on a non-trading day, the nearest prior
        trading date is used automatically.
    bloc : str or None
        Filter to a bloc: "G10", "LATAM", "CEEMEA", "APAC",
        or "EM" (= LATAM + CEEMEA + APAC combined). None = all.

    Returns
    -------
    pd.DataFrame
        Rows = currencies sorted by score (best long at top).
        Columns:
          composite view:  score, bloc, trend, value, carry, ...,
                           agreement, n_pillars, carry_pct, iv_3m, structure
          pillar view:     score, composite, bloc, <sub-components>, rank

    Examples
    --------
    >>> ex.snapshot()
                  score  bloc  trend  value  carry  ...  structure
    ccy
    BRL            1.42  LATAM   0.83   0.31   1.78  ...  vanilla long
    EUR            0.97  G10     0.54   1.23   0.31  ...  call spread
    ...

    >>> ex.snapshot("carry")
                  score  composite  bloc  carry_level  carry_slope  rank
    ccy
    BRL            1.78       1.42  LATAM         1.92         1.21   1.0
    TRY            1.65       0.84  CEEMEA        1.71         1.34   2.0
    ...

    >>> ex.snapshot("trend", bloc="G10", asof="2026-01-31")
    >>> ex.snapshot(bloc="EM")           # composite, EM only
    """
    s = _require_session()

    if pillar is not None and pillar not in PILLAR_FUNCS:
        raise ValueError(
            f"Unknown pillar '{pillar}'. "
            f"Valid: {list(PILLAR_FUNCS)}"
        )

    ts = _parse_asof(asof, s)

    if pillar is None:
        # Composite — runs the same expression layer as run_screener.py
        ivrv = _expr.ivrv_z(s.md.atm_vol, s.md.rv_3m)
        skw = _expr.skew_z(s.md.rr25, s.md.pairs)
        snap = _score.snapshot(s.md, s.pillars, s.total, s.conv, asof=ts)
        snap = _expr.build(snap, ivrv, skw)
        return _filter_bloc(snap, bloc)

    # Pillar drill-down
    panel = s.pillars[pillar]
    score_row = _score._asof_row(panel, panel.columns, ts).rename("score")

    df = score_row.to_frame()
    df["composite"] = _score._asof_row(s.total, s.total.columns, ts)
    df["bloc"] = [BLOCS.get(c) for c in df.index]

    for comp_name, comp_panel in s._components.get(pillar, {}).items():
        df[comp_name] = _score._asof_row(comp_panel, df.index, ts)

    df["rank"] = df["score"].rank(ascending=False).astype(int)
    df = df.sort_values("score", ascending=False)
    df.index.name = "ccy"
    return _filter_bloc(df, bloc)


def components(
    pillar: str,
    asof: Optional[str] = None,
) -> dict[str, pd.DataFrame] | dict[str, pd.Series]:
    """
    The normalised sub-signals that make up a pillar, before blending.

    Values are in sigma units (same scale as the pillar score itself), so
    sub-signals are directly comparable to each other and across pillars.
    A value of +1.5 means 1.5 standard deviations above the blended TS/XS
    norm for that signal; −1.0 is 1 sigma below.

    Parameters
    ----------
    pillar : str
        Pillar name. One of:
          "trend"         -> m3, m6, m12, dist_200d, macd_hist
          "value"         -> steer, beer
          "carry"         -> carry_level, carry_slope
          "risk_adj_carry"-> carry_iv, carry_downside, carry_cds
          "vol"           -> iv_change, rv_change, term_slope
          "positioning"   -> cot_level, cot_change, skew, skew_change
          "fundamental"   -> rate_diff_1m, rate_diff_3m, cesi_diff,
                            cds_change, tot_mom, ca_gdp
    asof : str or None
        None  -> returns dict[str, pd.DataFrame] — full history for each
                sub-signal. Use this for time-series analysis.
        date  -> returns dict[str, pd.Series]   — one value per currency
                at that date. Use this alongside snapshot() for diagnosis.

    Returns
    -------
    dict[str, pd.DataFrame]   when asof=None
    dict[str, pd.Series]      when asof is a date string

    Examples
    --------
    >>> c = ex.components("trend")
    >>> list(c.keys())
    ['m3', 'm6', 'm12', 'dist_200d', 'macd_hist']

    >>> c["m3"].tail()          # recent 3-month momentum scores, all ccys
    >>> c["m3"]["EUR"].plot()   # EUR 3m momentum z-score over time

    >>> c = ex.components("carry", asof="2026-08-14")
    >>> pd.DataFrame(c).sort_values("carry_level", ascending=False)
    #  -> which currencies have the best carry level vs slope today?

    >>> # Diagnose why BRL scores well on risk_adj_carry:
    >>> c = ex.components("risk_adj_carry", asof="2026-08-14")
    >>> pd.DataFrame(c).loc["BRL"]
    carry_iv        1.82    ← carry relative to implied vol: very high
    carry_downside  1.45    ← carry relative to downside vol: high
    carry_cds       0.93    ← carry relative to CDS: moderate
    """
    s = _require_session()

    if pillar not in PILLAR_FUNCS:
        raise ValueError(
            f"Unknown pillar '{pillar}'. Valid: {list(PILLAR_FUNCS)}"
        )

    comps = s._components.get(pillar, {})
    if not comps:
        raise RuntimeError(
            f"No components captured for '{pillar}'. "
            "Try ex.load(reload=True)."
        )

    if asof is None:
        return {k: v.copy() for k, v in comps.items()}

    ts = _parse_asof(asof, s)
    return {
        k: _score._asof_row(v, v.columns, ts).rename(k)
        for k, v in comps.items()
    }


def raw(
    attr: str,
    ccy: Optional[str] = None,
) -> pd.DataFrame | pd.Series:
    """
    Underlying market data — unscored, unscaled, in native units.

    Parameters
    ----------
    attr : str
        Which data series to return. Full list:

        Price:       spot          — spot rate (market convention)
                     fwd_points   — 3m fwd points in spot units
        Vol surface: atm_vol      — 3m ATM implied vol (%)
                     rr25         — 25d risk reversal (vol pts)
                     bf25         — 25d butterfly (vol pts)
                     atm_1m       — 1m ATM implied vol (%)
        Rates/macro: carry        — annualised fwd-implied carry (%),
                                    positive = attractive long non-USD.
                     swap2y       — 2y IRS rate (%)
                     cpi          — headline CPI y/y (%)
                     cds5y        — 5y USD sovereign CDS (bps)
                     reer         — real effective exchange rate index
                     tot          — terms-of-trade proxy index
                     cot          — CFTC net non-commercial position
                     cesi         — economic surprise index
                     ca_gdp       — current account % GDP
        Derived:     rv_3m        — 3m realised vol, close-to-close (%)
                     rv_1m        — 1m realised vol (%)
                     ret          — daily log returns, long non-USD
                     risk_beta    — rolling beta to credit spreads (-1,1)
        Regime:      regime       — panel of vix,move,dxy,oil,copper,
                                    gold,bcom,hy_oas. Pass ccy= to select
                                    one column (e.g. ccy="vix").
        USD legs:    usd_swap2y   — US 2y IRS (Series, ccy= ignored)
                     usd_cpi      — US CPI y/y (Series, ccy= ignored)
                     usd_cesi     — US CESI (Series, ccy= ignored)

    ccy : str or None
        Currency code to extract one Series. None = full panel.
        For `regime`, ccy selects the variable (e.g. "vix", "dxy", "oil").
        For USD legs (usd_*), ccy has no effect.

    Returns
    -------
    pd.DataFrame (ccy=None) or pd.Series (ccy specified)

    Examples
    --------
    >>> ex.raw("carry")                    # carry %, all 33 pairs
    >>> ex.raw("carry", "BRL")             # BRL carry time series
    >>> ex.raw("atm_vol", "JPY")           # USDJPY 3m ATM vol
    >>> ex.raw("cds5y", "ZAR")             # ZAR sovereign CDS in bps
    >>> ex.raw("regime")                   # all regime variables
    >>> ex.raw("regime", "vix")            # VIX index
    >>> ex.raw("spot").tail()              # latest spot levels
    >>> ex.raw("rv_3m").corr()             # realised vol correlation matrix
    """
    s = _require_session()

    if attr not in _RAW_ATTRS:
        raise ValueError(
            f"Unknown attribute '{attr}'.\n"
            f"Available: {sorted(_RAW_ATTRS)}"
        )

    data = getattr(s.md, attr)

    if ccy is None:
        return data

    if isinstance(data, pd.DataFrame):
        if ccy not in data.columns:
            raise ValueError(
                f"'{ccy}' not found in '{attr}'.\n"
                f"Available currencies: {sorted(data.columns.tolist())}"
            )
        return data[ccy].rename(f"{attr}[{ccy}]")

    # pd.Series (USD legs) — ccy is not applicable
    return data


def history(
    pillar: str,
    ccy: Optional[str] = None,
) -> pd.DataFrame | pd.Series:
    """
    Full time series of a normalised pillar or composite score.

    Parameters
    ----------
    pillar : str
        "composite" or any pillar name: "trend", "value", "carry",
        "risk_adj_carry", "vol", "positioning", "fundamental".
    ccy : str or None
        Currency code. None = full panel (all currencies).

    Returns
    -------
    pd.DataFrame (ccy=None) or pd.Series (ccy specified)
    Values are in sigma units.

    Examples
    --------
    >>> ex.history("carry", "BRL")           # BRL carry score since 2012
    >>> ex.history("composite").tail(252)    # last year, all currencies
    >>> ex.history("trend")["EUR"].plot()    # EUR trend score chart

    >>> # Which currencies have had the most stable carry score?
    >>> ex.history("carry").std().sort_values()
    """
    s = _require_session()

    valid = ["composite"] + list(PILLAR_FUNCS)
    if pillar not in valid:
        raise ValueError(
            f"Unknown pillar '{pillar}'. Valid: {valid}"
        )

    panel = s.total if pillar == "composite" else s.pillars[pillar]

    if ccy is None:
        return panel

    if ccy not in panel.columns:
        raise ValueError(
            f"'{ccy}' not in pillar '{pillar}'.\n"
            f"Available: {sorted(panel.columns.tolist())}"
        )
    return panel[ccy].rename(f"{pillar}[{ccy}]")


def crosses(asof: Optional[str] = None) -> pd.DataFrame:
    """
    Cross (RV) scores: score(base) − score(term).

    Derived from the composite scores, so they are internally consistent
    with the outright rankings — an RV long EURUSD can never contradict
    a composite short EUR and long USD.

    Parameters
    ----------
    asof : str or None
        Snapshot date. None = latest.

    Returns
    -------
    pd.DataFrame
        Index = cross labels (e.g. "EURGBP"), column "score",
        sorted by absolute conviction (most actionable first).

    Examples
    --------
    >>> ex.crosses()
    >>> ex.crosses(asof="2026-06-30")
    """
    s = _require_session()
    ts = _parse_asof(asof, s)

    total_ffilled = s.total.ffill()
    xscores = _score.cross_scores(total_ffilled)

    row = _score._asof_row(xscores, xscores.columns, ts)
    df = row.rename("score").to_frame()
    df["abs"] = df["score"].abs()
    return df.sort_values("abs", ascending=False).drop(columns="abs")


def to_excel(
    path: str = "explore.xlsx",
    asof: Optional[str] = None,
) -> Path:
    """
    Export the full drill-down to an Excel workbook.

    Sheets
    ------
    composite       Full ranked snapshot (= run_screener.py output).
    carry_diag      Forward convention / carry plausibility check.
                    Read this first if any carry number looks wrong.
    crosses         RV scores sorted by absolute conviction.
    <pillar>        One sheet per pillar: ranked by pillar score with
                    all sub-component columns appended.
    score_history   Composite score time series (full history).

    Parameters
    ----------
    path : str
        Output file path. Defaults to "explore.xlsx" in the current
        working directory.
    asof : str or None
        Snapshot date for all ranked sheets. None = latest.

    Returns
    -------
    pathlib.Path
        Resolved path to the written file.

    Examples
    --------
    >>> ex.to_excel()
    >>> ex.to_excel("drill_20260630.xlsx", asof="2026-06-30")
    """
    s = _require_session()
    out = Path(path).resolve()

    with pd.ExcelWriter(out, engine="openpyxl") as writer:

        snapshot(asof=asof).round(3).to_excel(writer, sheet_name="composite")

        if not s.md.fwd_diagnostic.empty:
            s.md.fwd_diagnostic.round(4).to_excel(writer, sheet_name="carry_diag")

        crosses(asof=asof).round(3).to_excel(writer, sheet_name="crosses")

        for name in PILLAR_FUNCS:
            try:
                df = snapshot(pillar=name, asof=asof)
                df.round(3).to_excel(writer, sheet_name=name[:31])
            except Exception as e:  # noqa: BLE001
                log.warning("Could not write sheet '%s': %s", name, e)

        s.total.round(4).to_excel(writer, sheet_name="score_history")

    print(f"Saved -> {out}")
    return out


def to_excel_raw(path: str = "raw_data.xlsx") -> Path:
    """
    Export all raw Bloomberg market data to an Excel workbook.

    One sheet per data panel, in native units (%, bps, index levels).
    No normalisation, no scoring — straight from the parquet cache.

    Sheets
    ------
    spot            Spot rates (market convention)
    fwd_points      3m forward points in spot units
    carry           Annualised forward-implied carry (%)
    atm_vol         3m ATM implied vol (%)
    rr25            25d risk reversal (vol pts)
    swap2y          2y IRS/OIS rate (%)
    reer            Real effective exchange rate index
    cds5y           5y USD sovereign CDS (bps)
    cot_net         CFTC net non-commercial position (long - short)
    cot_oi          CFTC open interest (long + short)
    cesi            Economic surprise index
    ca_gdp          Current account % GDP
    tot             Terms of trade proxy
    rv_3m           3m realised vol (%)
    rv_1m           1m realised vol (%)
    regime          Regime variables (VIX, MOVE, DXY, oil, etc.)

    Parameters
    ----------
    path : str
        Output file path. Defaults to "raw_data.xlsx".

    Returns
    -------
    pathlib.Path
        Resolved path to the written file.

    Examples
    --------
    >>> ex.to_excel_raw()
    >>> ex.to_excel_raw("raw_20260817.xlsx")
    """
    s = _require_session()
    out = Path(path).resolve()

    sheets = [
        ("spot",       s.md.spot),
        ("fwd_points", s.md.fwd_points),
        ("carry",      s.md.carry),
        ("atm_vol",    s.md.atm_vol),
        ("rr25",       s.md.rr25),
        ("swap2y",     s.md.swap2y),
        ("reer",       s.md.reer),
        ("cds5y",      s.md.cds5y),
        ("cot_net",    s.md.cot),
        ("cot_oi",     s.md.cot_oi),
        ("cesi",       s.md.cesi),
        ("ca_gdp",     s.md.ca_gdp),
        ("tot",        s.md.tot),
        ("rv_3m",      s.md.rv_3m),
        ("rv_1m",      s.md.rv_1m),
        ("regime",     s.md.regime),
    ]

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for sheet_name, df in sheets:
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name)

    print(f"Saved -> {out}")
    return out
