"""
Pillar construction.

Every pillar returns a panel (index = dates, columns = currency codes) already
oriented so that POSITIVE = attractive to be LONG the non-USD currency vs USD,
and already normalised to sigma units by normalize.normalise_panel.

Orientation multipliers (defined once, in config.Pair.sign):
  price-based signals   ->  p.sign   (rising USDBRL = weaker BRL)
  rate/carry-based      -> -p.sign   (USDBRL fwd premium = BRL earns carry)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import bbg
from config import (
    BEER_WINDOW, CARRY_TENOR_DAYS, FAIR_VALUE_WINDOW, REGIME_TICKERS,
    STEER_RISK_PROXY, UNIVERSE, USD_CESI, USD_CPI, USD_SWAP2Y, VOL_TENOR,
)
from normalize import combine, normalise_panel

log = logging.getLogger(__name__)

ANN = 252
BLOCS = pd.Series({p.ccy: p.bloc for p in UNIVERSE})


# ==========================================================================
# Market data container
# ==========================================================================

class MarketData:
    """Loads and caches everything the pillars need. One instance per run."""

    def __init__(self, start: str = "2010-01-01", end: str | None = None):
        self.start, self.end = start, end
        self.pairs = {p.ccy: p for p in UNIVERSE}
        log.info("Loading market data for %d currencies", len(self.pairs))

        self.spot = self._panel("ticker")
        self.fwd_points = self._fwd_points()
        self.atm_vol = self._vol_panel("atm")
        self.rr25 = self._vol_panel("rr25")
        self.bf25 = self._vol_panel("bf25")
        self.atm_1m = self._vol_panel("atm", tenor="1M")
        self.swap2y = self._panel("swap2y")
        self.cpi = self._panel("cpi_yoy", self.FFILL_LIMIT_SLOW)
        self.cds5y = self._panel("cds5y")
        self.reer = self._panel("reer", self.FFILL_LIMIT_SLOW)
        self.tot = self._panel("tot")
        _cot_long = self._panel("cot_long", self.FFILL_LIMIT_SLOW)
        _cot_short = self._panel("cot_short", self.FFILL_LIMIT_SLOW)
        common = _cot_long.columns.intersection(_cot_short.columns)
        if len(common):
            self.cot = _cot_long[common] - _cot_short[common]
            self.cot_oi = _cot_long[common] + _cot_short[common]
        else:
            self.cot = pd.DataFrame()
            self.cot_oi = pd.DataFrame()
        self.cesi = self._panel("cesi", 10)
        self.ca_gdp = self._panel("ca_gdp", self.FFILL_LIMIT_SLOW)

        self.regime = bbg.frame(REGIME_TICKERS.values(), start=start, end=end)
        self.regime.columns = [
            k for k, v in REGIME_TICKERS.items() if v in self.regime.columns
        ]

        self.usd_swap2y = bbg.history(USD_SWAP2Y, start=start, end=end)
        self.usd_cpi = bbg.history(USD_CPI, start=start, end=end)
        self.usd_cesi = bbg.history(USD_CESI, start=start, end=end)

        # Derived
        self.ret = self._returns()
        self.rv_3m = self.ret.rolling(63, min_periods=40).std() * np.sqrt(ANN) * 100
        self.rv_1m = self.ret.rolling(21, min_periods=14).std() * np.sqrt(ANN) * 100
        self.carry = self._carry()
        self.risk_beta = self._risk_beta()

    # -- loaders ----------------------------------------------------------
    # Panels are built on the UNION of all currencies' trading days, so every
    # currency carries NaN on days its own market was shut. Left untreated
    # those NaNs destroy every rolling window that touches them, and they hit
    # hardest exactly where calendars diverge most — the EM names. Daily
    # panels are therefore forward-filled to a common calendar, with a limit
    # so a genuinely dead series still shows up as dead rather than being
    # propagated forward indefinitely.

    FFILL_LIMIT_DAILY = 5        # a local holiday run; beyond this it is stale
    FFILL_LIMIT_SLOW = 90        # CPI, current account, REER: monthly/quarterly

    def _panel(self, attr: str, ffill_limit: int | None = None) -> pd.DataFrame:
        cols = {}
        for ccy, p in self.pairs.items():
            tk = getattr(p, attr)
            if not tk:
                continue
            if isinstance(tk, tuple):
                s = bbg.stitch(tk[0], tk[1], start=self.start, end=self.end)
            else:
                s = bbg.history(tk, start=self.start, end=self.end)
            if not s.empty:
                cols[ccy] = s
        if not cols:
            return pd.DataFrame()
        df = pd.DataFrame(cols).sort_index()
        lim = self.FFILL_LIMIT_DAILY if ffill_limit is None else ffill_limit
        return df.ffill(limit=lim)

    def _vol_panel(self, kind: str, tenor: str = VOL_TENOR) -> pd.DataFrame:
        cols = {}
        for ccy, p in self.pairs.items():
            tk = bbg.vol_tickers(p.vol_root, tenor)[kind]
            s = bbg.history(tk, start=self.start, end=self.end)
            if not s.empty:
                cols[ccy] = s
        if not cols:
            return pd.DataFrame()
        return pd.DataFrame(cols).sort_index().ffill(limit=self.FFILL_LIMIT_DAILY)

    def _fwd_points(self) -> pd.DataFrame:
        """3m forward points in spot units, from the declared conventions.

            points mode   : points = raw / divisor,  fwd = spot + points
            outright mode : points = raw - spot,     fwd = raw

        Conventions come from config.FWD_CONVENTIONS and are applied literally.
        Nothing is inferred. The implied carry is recomputed here and checked
        against a plausibility band, because a divisor that is wrong by a
        factor of ten produces a number that ranks rather than an error that
        surfaces.
        """
        from config import FWD_CONVENTIONS

        cols, diag = {}, []
        for ccy, p in self.pairs.items():
            if ccy not in FWD_CONVENTIONS or ccy not in self.spot.columns:
                continue
            tk, div, mode = FWD_CONVENTIONS[ccy]
            raw = bbg.history(tk, start=self.start, end=self.end)
            if raw.empty:
                log.warning("fwd %s: %s returned nothing", ccy, tk)
                continue

            spot = self.spot[ccy].reindex(raw.index).ffill()
            if mode == "outright":
                # Ticker quotes the outright. Back out points so that every
                # currency in the universe is handled by the same arithmetic:
                # FWD = SPOT + POINTS, without exception.
                pts = raw - spot
            else:
                pts = raw / div
            fwd = spot + pts

            carry = ((fwd / spot - 1.0) * (365.0 / CARRY_TENOR_DAYS)
                     * 100.0 * (-p.sign))
            c_last = float(carry.iloc[-1]) if len(carry.dropna()) else np.nan

            diag.append({
                "ccy": ccy, "ticker": tk, "mode": mode, "divisor": div,
                "spot": round(float(spot.iloc[-1]), 4),
                "raw_pts": round(float(raw.iloc[-1]), 4),
                "outright": round(float(fwd.iloc[-1]), 4),
                "carry_pct": round(c_last, 2),
            })
            cols[ccy] = pts

        self.fwd_diagnostic = pd.DataFrame(diag).set_index("ccy") if diag else pd.DataFrame()
        return pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()

    def sanity_check_carry(self, raise_on_fail: bool = True) -> pd.DataFrame:
        """Reject implausible carry before it reaches the ranks.

        A forward-convention error is not a data-quality nuisance; it is a
        ranking error. A TRY divisor off by ten puts TRY top of the universe on
        carry and carry-adjusted risk simultaneously and drags the whole EM
        cross-section with it, because the z-scores are computed within bloc.
        Better to stop the run.
        """
        from config import CARRY_SANITY_MAX, CARRY_SANITY_MIN

        if self.fwd_diagnostic.empty:
            return pd.DataFrame()
        d = self.fwd_diagnostic
        # Only a grossly oversized carry stops the run. A near-zero carry is
        # suspicious but can be legitimate, so it warns via check_data.py
        # rather than blocking here.
        bad = d[(d["carry_pct"].abs() > CARRY_SANITY_MAX) | d["carry_pct"].isna()]
        if not bad.empty and raise_on_fail:
            msg = ("Implausible carry — check the divisor in "
                   "config.FWD_CONVENTIONS:\n" + bad.to_string())
            raise ValueError(msg)
        return bad

    # -- derived ----------------------------------------------------------

    def _returns(self) -> pd.DataFrame:
        """Daily log return of being LONG the non-USD currency."""
        out = {}
        for ccy in self.spot.columns:
            r = np.log(self.spot[ccy]).diff() * self.pairs[ccy].sign
            out[ccy] = r
        return pd.DataFrame(out)

    def _carry(self) -> pd.DataFrame:
        """Annualised forward-implied carry of being long the non-USD ccy, %.

        Uses forward points rather than policy or swap differentials so that
        NDF premium, funding squeezes and CIP basis are captured. In EM this
        is frequently the difference between a printable trade and a paper one.
        """
        out = {}
        for ccy in self.spot.columns:
            if ccy not in self.fwd_points.columns:
                continue
            p = self.pairs[ccy]
            s = self.spot[ccy]
            f = s + self.fwd_points[ccy]
            prem = (f / s) - 1.0
            out[ccy] = prem * (365.0 / CARRY_TENOR_DAYS) * 100.0 * (-p.sign)
        return pd.DataFrame(out)

    def _risk_beta(self, window: int = 252) -> pd.DataFrame:
        """Rolling beta of each currency's return to a risk-off factor.

        Positive beta = the currency rallies when credit widens (JPY, CHF).
        Negative beta = high-beta risk currency (ZAR, BRL, AUD). Used to sign
        the vol pillar without hardcoding a haven list, so the classification
        updates when a currency changes character.
        """
        if "hy_oas" not in self.regime.columns or self.ret.empty:
            return pd.DataFrame()
        factor = self.regime["hy_oas"].reindex(self.ret.index).ffill().diff()
        out = {}
        for ccy in self.ret.columns:
            y = self.ret[ccy]
            cov = y.rolling(window, min_periods=120).cov(factor)
            var = factor.rolling(window, min_periods=120).var()
            out[ccy] = cov / var.replace(0, np.nan)
        b = pd.DataFrame(out)
        return np.tanh(b / b.abs().stack().std())      # squash to (-1, 1)


# ==========================================================================
# Helpers
# ==========================================================================

def _rate_diff(md: MarketData) -> pd.DataFrame:
    """2y swap differential vs USD, in bp, with a forward-implied fallback.

    The *SW2 swap roots are LIBOR-referenced and have been progressively
    retired; several resolve but return stale series, which is worse than
    failing because it passes validation silently. So: use the swap
    differential where the series is genuinely live, and fall back to the
    forward-implied differential everywhere else.

    Covered interest parity means the 3m forward premium IS a rate
    differential — a shorter one, and the tradeable one, including any CIP
    basis. It is a defensible substitute rather than a patch: for a 1-3m
    signal horizon the front-end differential is arguably the better input,
    and it removes the model's dependence on a ticker map that keeps rotting.
    """
    fwd_implied = md.carry * 100.0 if not md.carry.empty else pd.DataFrame()

    if md.swap2y.empty or md.usd_swap2y.empty:
        return fwd_implied

    usd = md.usd_swap2y.reindex(md.swap2y.index).ffill()
    # A USD leg that is stale poisons every currency, so check it explicitly.
    usd_live = usd.notna().sum() >= 0.6 * len(usd)
    if not usd_live:
        log.warning("USD 2y swap series is stale — using forward-implied "
                    "differentials for the whole universe")
        return fwd_implied

    diff = md.swap2y.sub(usd, axis=0) * 100

    # Per-currency staleness check; substitute the forward-implied leg where
    # the swap series has stopped updating.
    if not fwd_implied.empty:
        recent = diff.tail(252)
        for ccy in diff.columns:
            if recent[ccy].notna().sum() < 0.5 * len(recent) and ccy in fwd_implied.columns:
                log.info("swap2y stale for %s — using forward-implied", ccy)
                diff[ccy] = fwd_implied[ccy]
        for ccy in fwd_implied.columns:
            if ccy not in diff.columns:
                diff[ccy] = fwd_implied[ccy]
    return diff


def _real_rate_diff(md: MarketData) -> pd.DataFrame:
    """2y swap less headline CPI y/y, differential vs USD, in bp."""
    rd = _rate_diff(md)
    if rd.empty or md.cpi.empty or md.usd_cpi.empty:
        return rd
    usd_cpi = md.usd_cpi.reindex(rd.index).ffill()
    infl_diff = md.cpi.reindex(rd.index).ffill().sub(usd_cpi, axis=0) * 100
    return rd.sub(infl_diff.reindex(columns=rd.columns), fill_value=0.0)


def _rolling_ols_residual(
    y: pd.Series,
    X: pd.DataFrame,
    window: int,
    step: int = 5,
) -> pd.Series:
    """Residual of y on X from a strictly backward-looking rolling OLS.

    Refit every `step` business days and hold the coefficients between refits.
    This is 5x cheaper than a daily refit and materially reduces the parameter
    churn that makes naive rolling fair-value models look better in sample than
    they trade out of sample.
    """
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(df) < window + 10:
        return pd.Series(np.nan, index=y.index)

    resid = pd.Series(np.nan, index=df.index)
    cols = list(X.columns)
    beta = None
    for i in range(window, len(df)):
        if (i - window) % step == 0 or beta is None:
            w = df.iloc[i - window:i]
            A = np.column_stack([np.ones(len(w)), w[cols].values])
            try:
                beta, *_ = np.linalg.lstsq(A, w["y"].values, rcond=None)
            except np.linalg.LinAlgError:
                continue
        row = df.iloc[i]
        fitted = beta[0] + float(np.dot(beta[1:], row[cols].values))
        resid.iloc[i] = row["y"] - fitted
    return resid.reindex(y.index)


def _mom(df: pd.DataFrame, days: int) -> pd.DataFrame:
    return np.log(df).diff(days)


# ==========================================================================
# Pillar 1 — Trend
# ==========================================================================

def trend(md: MarketData) -> pd.DataFrame:
    """Vol-normalised price momentum. Direction only; no regime gate."""
    sgn = pd.Series({c: md.pairs[c].sign for c in md.spot.columns})
    rv = md.rv_3m.replace(0, np.nan) / 100.0

    m3 = (_mom(md.spot, 63).mul(sgn, axis=1)).div(rv)
    m6 = (_mom(md.spot, 126).mul(sgn, axis=1)).div(rv)
    m12 = (_mom(md.spot, 252).mul(sgn, axis=1)).div(rv)

    ma200 = md.spot.rolling(200, min_periods=120).mean()
    dist = ((md.spot / ma200 - 1.0).mul(sgn, axis=1)).div(rv)

    ema12 = md.spot.ewm(span=12).mean()
    ema26 = md.spot.ewm(span=26).mean()
    macd = ema12 - ema26
    hist = ((macd - macd.ewm(span=9).mean()) / md.spot).mul(sgn, axis=1).div(rv)

    return combine({
        "m3": normalise_panel(m3, BLOCS),
        "m6": normalise_panel(m6, BLOCS),
        "m12": normalise_panel(m12, BLOCS),
        "dist_200d": normalise_panel(dist, BLOCS),
        "macd_hist": normalise_panel(hist, BLOCS),
    }, weights={"m3": 1.5, "m6": 1.5, "m12": 1.0, "dist_200d": 1.0, "macd_hist": 0.5})


# ==========================================================================
# Pillar 2 — Value (self-estimated, deliberately minimal)
# ==========================================================================

def value(md: MarketData) -> pd.DataFrame:
    """BEER-lite and STEER-lite misvaluation residuals.

    STEER-lite : log spot ~ 2y nominal rate diff + commodity ToT + credit risk,
                 2y rolling window. Short-horizon financial fair value.
    BEER-lite  : log REER ~ 2y real rate diff + ToT + current account %GDP,
                 5y rolling window. Medium-horizon fundamental fair value.

    Sign: a positive residual means the currency trades ABOVE fair value, so the
    value score is the negative of the residual z. Three regressors each, on
    purpose — every extra regressor buys in-sample fit and sells out-of-sample
    signal, and this pillar is the easiest place in the framework to overfit.
    """
    rd = _rate_diff(md)
    rrd = _real_rate_diff(md)
    risk = md.regime.get(STEER_RISK_PROXY)

    steer, beer = {}, {}
    for ccy in md.spot.columns:
        p = md.pairs[ccy]

        # log price oriented as "value of the non-USD currency"
        lspot = np.log(md.spot[ccy]) * p.sign

        # ---- STEER-lite
        X = {}
        if ccy in rd.columns:
            X["rate_diff"] = rd[ccy]
        if ccy in md.tot.columns:
            X["tot"] = np.log(md.tot[ccy])
        if risk is not None:
            X["risk"] = risk.reindex(lspot.index).ffill()
        if len(X) >= 2:
            r = _rolling_ols_residual(lspot, pd.DataFrame(X), FAIR_VALUE_WINDOW)
            steer[ccy] = r

        # ---- BEER-lite
        base = np.log(md.reer[ccy]) if ccy in md.reer.columns else lspot
        Xb = {}
        if ccy in rrd.columns:
            Xb["real_rate_diff"] = rrd[ccy]
        if ccy in md.tot.columns:
            Xb["tot"] = np.log(md.tot[ccy])
        if ccy in md.ca_gdp.columns:
            Xb["ca_gdp"] = md.ca_gdp[ccy].reindex(base.index).ffill()
        if len(Xb) >= 2:
            r = _rolling_ols_residual(base, pd.DataFrame(Xb), BEER_WINDOW)
            beer[ccy] = r

    out = {}
    if steer:
        out["steer"] = normalise_panel(pd.DataFrame(steer), BLOCS, higher_is_better=False)
    if beer:
        out["beer"] = normalise_panel(pd.DataFrame(beer), BLOCS, higher_is_better=False)
    return combine(out, weights={"steer": 1.0, "beer": 1.0})


# ==========================================================================
# Pillar 3 — Carry
# ==========================================================================

def carry(md: MarketData) -> pd.DataFrame:
    """Forward-implied carry plus roll-down along the forward curve."""
    lvl = md.carry
    # Roll-down: 3m carry less 1m-equivalent, i.e. slope of the points curve.
    slope = lvl - lvl.rolling(21, min_periods=14).mean()
    return combine({
        "carry_level": normalise_panel(lvl, BLOCS),
        "carry_slope": normalise_panel(slope, BLOCS),
    }, weights={"carry_level": 2.0, "carry_slope": 1.0})


# ==========================================================================
# Pillar 4 — Carry adjusted for risk
# ==========================================================================

def risk_adj_carry(md: MarketData) -> pd.DataFrame:
    """Is the carry paying you for the risk you are taking?

    carry / implied vol   : the option market's price of the risk
    carry / downside RV   : realised left-tail, which is what actually hurts
    carry / 5y CDS        : is the currency compensating for sovereign credit
                            (EM only; G10 renormalises over the other two)

    This pillar is what separates a genuine carry opportunity from levered beta.
    """
    comps = {}

    if not md.atm_vol.empty:
        comps["carry_iv"] = normalise_panel(
            md.carry / md.atm_vol.reindex_like(md.carry).replace(0, np.nan), BLOCS)

    downside = md.ret.where(md.ret < 0).rolling(63, min_periods=20).std() * np.sqrt(ANN) * 100
    comps["carry_downside"] = normalise_panel(
        md.carry / downside.reindex_like(md.carry).replace(0, np.nan), BLOCS)

    if not md.cds5y.empty:
        cds = md.cds5y.reindex_like(md.carry).replace(0, np.nan)
        comps["carry_cds"] = normalise_panel(md.carry / (cds / 100.0), BLOCS)

    return combine(comps, weights={"carry_iv": 1.0, "carry_downside": 1.0, "carry_cds": 1.0})


# ==========================================================================
# Pillar 5 — Volatility regime
# ==========================================================================

def vol_regime(md: MarketData) -> pd.DataFrame:
    """Directional content of the vol surface.

    Rising and inverting vol is bearish for high-beta currencies and bullish for
    funders, so each component is signed by the currency's own rolling risk beta
    rather than by a hardcoded haven list. Rich/cheap vol (IV/RV) and skew are
    NOT scored here — they are expression inputs and live in expression.py.
    """
    if md.atm_vol.empty:
        return pd.DataFrame()
    beta = md.risk_beta
    if beta.empty:
        beta = pd.DataFrame(-1.0, index=md.atm_vol.index, columns=md.atm_vol.columns)
    beta = beta.reindex_like(md.atm_vol).ffill()

    d_iv = md.atm_vol.diff(21)
    d_rv = md.rv_1m.reindex_like(md.atm_vol).diff(21)
    term = (md.atm_vol - md.atm_1m.reindex_like(md.atm_vol))   # 3M less 1M

    return combine({
        "iv_change": normalise_panel(-d_iv * beta, BLOCS),
        "rv_change": normalise_panel(-d_rv * beta, BLOCS),
        "term_slope": normalise_panel(term * beta.abs(), BLOCS),
    }, weights={"iv_change": 1.0, "rv_change": 1.0, "term_slope": 0.5})


# ==========================================================================
# Pillar 6 — Positioning
# ==========================================================================

def positioning(md: MarketData) -> pd.DataFrame:
    """Contrarian at extremes: crowded longs score negative.

    CoT covers G10 plus MXN, BRL and ZAR. Everything else runs on 25d risk
    reversal skew as the sentiment proxy, which combine() handles by
    renormalising over whatever resolved.
    """
    comps = {}

    if not md.cot.empty:
        # Net = L - S: direction. Level and 3m change both matter — a crowded
        # position that is still building is less dangerous than one that peaked.
        comps["cot_level"] = normalise_panel(md.cot, BLOCS, higher_is_better=False)
        comps["cot_change"] = normalise_panel(md.cot.diff(63), BLOCS, higher_is_better=False)
        # Skew = Net / OI: removes noise from absolute OI changes; ranges -1 to +1.
        if not md.cot_oi.empty:
            skew_ratio = md.cot.div(md.cot_oi.replace(0, np.nan))
            comps["cot_skew"] = normalise_panel(skew_ratio, BLOCS, higher_is_better=False)

    if not md.rr25.empty:
        sgn = pd.Series({c: -md.pairs[c].sign for c in md.rr25.columns})
        skew = md.rr25.mul(sgn, axis=1)      # + = market pays up for ccy calls
        comps["skew"] = normalise_panel(skew, BLOCS, higher_is_better=False)
        comps["skew_change"] = normalise_panel(skew.diff(21), BLOCS, higher_is_better=False)

    return combine(comps, weights={
        "cot_level": 1.0, "cot_change": 0.5, "cot_skew": 1.0,
        "skew": 1.0, "skew_change": 0.5})


# ==========================================================================
# Pillar 7 — Fundamental momentum
# ==========================================================================

def fundamental(md: MarketData) -> pd.DataFrame:
    """Rate of change of the fundamentals, not their level.

    Levels are already priced through carry and value; the tradeable edge at a
    1-3 month horizon sits in the second derivative.
    """
    comps = {}

    rd = _rate_diff(md)
    if not rd.empty:
        comps["rate_diff_1m"] = normalise_panel(rd.diff(21), BLOCS)
        comps["rate_diff_3m"] = normalise_panel(rd.diff(63), BLOCS)

    if not md.cesi.empty and not md.usd_cesi.empty:
        usd = md.usd_cesi.reindex(md.cesi.index).ffill()
        comps["cesi_diff"] = normalise_panel(md.cesi.sub(usd, axis=0), BLOCS)

    if not md.cds5y.empty:
        comps["cds_change"] = normalise_panel(md.cds5y.diff(63), BLOCS, higher_is_better=False)

    if not md.tot.empty:
        comps["tot_mom"] = normalise_panel(_mom(md.tot, 63), BLOCS)

    if not md.ca_gdp.empty:
        comps["ca_gdp"] = normalise_panel(md.ca_gdp, BLOCS)

    return combine(comps, weights={
        "rate_diff_1m": 1.0, "rate_diff_3m": 1.0, "cesi_diff": 1.0,
        "cds_change": 1.0, "tot_mom": 0.75, "ca_gdp": 0.5})


PILLAR_FUNCS = {
    "trend": trend,
    "value": value,
    "carry": carry,
    "risk_adj_carry": risk_adj_carry,
    "vol": vol_regime,
    "positioning": positioning,
    "fundamental": fundamental,
}
