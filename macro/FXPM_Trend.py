"""
FXPM_Trend.py — G10 + EM + precious metals trend monitor.

FX, Precious Metals: one z/u/phase framework across all underlyings.

Design principle: crosses are NEVER scored directly. Every score is built from
USD-leg z-scores and then differenced, so EURJPY is always exactly consistent
with EURUSD and USDJPY. Scoring EURJPY off its own price series would let the
three disagree.

Pipeline
    1. fetch USD legs only (19 tickers cover 20 currencies)
    2. restate all as USDXXX  -> one sign convention
    3. per-horizon z on each USD leg
    4. derive per-horizon cross z:  z(BASEQUOTE) = z(USDQUOTE) - z(USDBASE)
    5. renormalise the derived z by its own 252d std   <- see note below
    6. response function -> u; state engine -> phase/coherence

Note on step 5: differencing two unit-variance legs gives var = 2(1-rho).
Since both legs share the USD factor, rho > 0 and the naive /2 rescaling used
earlier is wrong and pair-dependent. Renormalising against the derived series'
own history is self-calibrating and keeps every pair on one scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fx_momentum import HORIZONS, _response, fetch_spot

# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
G10_PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD',
    'AUDUSD', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY',
    'AUDJPY', 'CADJPY', 'NZDCHF', 'EURCHF', 'AUDNZD',
    'EURNOK', 'USDNOK', 'EURSEK', 'USDSEK',
]

EM_PAIRS = [
    'USDBRL', 'USDMXN', 'USDCLP', 'USDCOP',
    'EURHUF', 'USDPLN', 'EURCZK',
    'USDZAR', 'USDKRW', 'USDIDR', 'USDINR',
]

# Precious metals. Quoted XAUUSD / XAGUSD, i.e. metal is the BASE, so a
# positive score means long metal / short USD - same convention as EURUSD.
# Treated as currencies throughout: the USD leg is stored as USDXAU = 1/XAUUSD
# so the whole USD-leg machinery applies unchanged.
PM_PAIRS = ['XAUUSD', 'XAGUSD']

ALL_PAIRS = G10_PAIRS + EM_PAIRS + PM_PAIRS

# Metals also allow crosses if you want them later, e.g. 'XAUEUR', 'XAUJPY'.
# They derive from the USD legs with no extra data - just add to ALL_PAIRS.

# Bloomberg quotes these as XXXUSD; everything else is USDXXX.
USD_BASE_EXCEPTIONS = {'EUR', 'GBP', 'AUD', 'NZD', 'XAU', 'XAG'}

# Metals price off Comdty-style roots but XAUUSD/XAGUSD trade as Curncy on BBG.
METALS = {'XAU', 'XAG'}

# Non-deliverable / restricted onshore. Flagged in the monitor, not excluded.
NDF_OR_RESTRICTED = {'CLP', 'COP', 'KRW', 'IDR', 'BRL', 'INR'}

# Vol tenor per horizon leg, matched to the crossover's centre of mass
# (l-s)/ln2 days: (8,24)->23d, (16,48)->46d, (32,96)->92d
IV_TENOR = {(8, 24): 'V1M', (16, 48): 'V2M', (32, 96): 'V3M'}
IV_DAYS = {(8, 24): 21, (16, 48): 42, (32, 96): 63}

Z_PEAK = np.sqrt(2.0)
Z_ENTER_STRETCHED, Z_EXIT_STRETCHED = 1.55, 1.27


# Metals carry structurally higher vol than G10 FX (silver especially) and
# their vol regimes are longer-lived. The z-scoring normalises level, but the
# 252d signal window can be short relative to metal cycles - override here if
# a sweep shows it matters.
SIG_WIN_OVERRIDE: dict[str, int] = {}   # e.g. {'XAGUSD': 504}


def currencies(pairs=ALL_PAIRS) -> list[str]:
    ccy = {'USD'}
    for p in pairs:
        ccy.add(p[:3]); ccy.add(p[3:])
    return sorted(ccy)


def usd_legs(pairs=ALL_PAIRS) -> list[str]:
    """Bloomberg tickers for the USD legs needed to span the universe."""
    out = []
    for c in currencies(pairs):
        if c == 'USD':
            continue
        out.append(f'{c}USD' if c in USD_BASE_EXCEPTIONS else f'USD{c}')
    return out


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_spot(pairs=ALL_PAIRS, start='2015-01-01', end=None,
              max_ffill: int = 3) -> pd.DataFrame:
    """
    Fetch USD legs and restate all as USDXXX.
    EM holiday calendars differ, so align on the union index and forward-fill
    a bounded number of days. Unbounded ffill would manufacture false calm.
    """
    end = end or pd.Timestamp.today()
    px = fetch_spot(usd_legs(pairs), start, end)
    out = {}
    for c in px.columns:
        out[c] = px[c] if c.startswith('USD') else 1.0 / px[c]
        if not c.startswith('USD'):
            out.pop(c); out['USD' + c[:3]] = 1.0 / px[c]
    df = pd.DataFrame(out).sort_index()
    return df.ffill(limit=max_ffill)


def load_ivol(pairs=ALL_PAIRS, start='2015-01-01', end=None,
              tenor='V1M') -> pd.DataFrame:
    """ATM implied vol for the USD legs, e.g. EURUSDV1M Curncy."""
    end = end or pd.Timestamp.today()
    tks = [f'{leg}{tenor}' for leg in usd_legs(pairs)]
    iv = fetch_spot(tks, start, end)
    iv.columns = [c.replace(tenor, '') for c in iv.columns]
    return iv.rename(columns={c: c if c.startswith('USD') else 'USD' + c[:3]
                              for c in iv.columns})


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def leg_z(px: pd.DataFrame,
          horizons=HORIZONS,
          vol_win: int = 63,
          sig_win: int = 252,
          ivol: dict[tuple, pd.DataFrame] | None = None
          ) -> dict[tuple, pd.DataFrame]:
    """Per-horizon z for each USD leg. Returns {horizon: DataFrame}."""
    lp = np.log(px.astype(float))
    out = {}
    for h in horizons:
        s, l = h
        x = lp.ewm(halflife=s, min_periods=s).mean() - \
            lp.ewm(halflife=l, min_periods=l).mean()
        if ivol is not None and h in ivol:
            scale = ivol[h].reindex_like(lp) / 100 * np.sqrt(IV_DAYS[h] / 252)
        else:
            scale = lp.diff().rolling(vol_win).std() * np.sqrt(vol_win)
        y = x / scale
        out[h] = y / y.rolling(sig_win).std()
    return out


def pair_z(zlegs: dict[tuple, pd.DataFrame], pair: str,
           sig_win: int = 252) -> dict[tuple, pd.Series]:
    """
    Derive per-horizon z for any pair from the USD legs.
        z(BASEQUOTE) = z(USDQUOTE) - z(USDBASE),  z(USDUSD) := 0
    then renormalise each horizon against its own trailing std.
    """
    base, quote = pair[:3], pair[3:]
    out = {}
    for h, Z in zlegs.items():
        zb = 0.0 if base == 'USD' else Z[f'USD{base}']
        zq = 0.0 if quote == 'USD' else Z[f'USD{quote}']
        raw = (zq - zb) if not isinstance(zq - zb, float) else None
        raw = pd.Series(zq, index=Z.index) - pd.Series(zb, index=Z.index) \
            if raw is None else raw
        out[h] = raw / raw.rolling(sig_win).std()
    return out


def score_universe(px: pd.DataFrame, pairs=ALL_PAIRS, **kw):
    """Composite z, composite u, and per-horizon sign coherence for every pair."""
    zlegs = leg_z(px, **kw)
    z, u, coh = {}, {}, {}
    for p in pairs:
        ph = pair_z(zlegs, p)
        stack = pd.concat(ph.values(), axis=1)
        z[p] = stack.mean(axis=1)
        # Response applied to the COMPOSITE, not averaged across horizons.
        # Averaging u_k lets sign(u) disagree with sign(z) when one leg is
        # stretched and another is fresh - which breaks the direction label.
        u[p] = _response(z[p])
        coh[p] = np.sign(stack).eq(np.sign(z[p]), axis=0).sum(axis=1)
    return pd.DataFrame(z), pd.DataFrame(u), pd.DataFrame(coh)


# --------------------------------------------------------------------------- #
# State engine
# --------------------------------------------------------------------------- #
def monitor(px: pd.DataFrame, pairs=ALL_PAIRS, lookback: int = 5,
            deadband: float = 0.04, flat_below: float = 0.10,
            target_vol: float = 0.10, **kw) -> pd.DataFrame:
    """One row per pair: direction, phase, coherence, strength, weight."""
    z, u, coh = score_universe(px, pairs, **kw)
    az = z.abs()

    reg = pd.DataFrame(np.nan, index=az.index, columns=az.columns)
    reg = reg.mask(az >= Z_ENTER_STRETCHED, 1.0).mask(az <= Z_EXIT_STRETCHED, 0.0)
    reg = reg.ffill().fillna((az > Z_PEAK).astype(float))

    dz = az - az.shift(lookback)
    trav = pd.DataFrame(0, index=az.index, columns=az.columns)
    trav = trav.mask(dz > deadband, 1).mask(dz < -deadband, -1)

    PHASE = {(0, 1): 'BUILDING', (0, 0): 'RUNNING', (0, -1): 'DECAYING',
             (1, 1): 'EXTENDED', (1, 0): 'TOPPING', (1, -1): 'NORMALISING'}

    # pair price implied from USD legs, for the vol denominator
    ann = {}
    for p in pairs:
        b, q = p[:3], p[3:]
        lb = 0.0 if b == 'USD' else np.log(px[f'USD{b}'])
        lq = 0.0 if q == 'USD' else np.log(px[f'USD{q}'])
        ann[p] = (lq - lb).diff().rolling(63).std() * np.sqrt(252)
    ann = pd.DataFrame(ann)

    r, t = reg.iloc[-1], trav.iloc[-1]
    zl, ul, cl, vl = z.iloc[-1], u.iloc[-1], coh.iloc[-1], ann.iloc[-1]

    # days in the CURRENT phase (regime x travel), not just regime
    ph_id = reg * 10 + trav
    days = (ph_id != ph_id.shift()).iloc[::-1].cumsum().iloc[::-1]
    days = (days == days.iloc[-1]).sum()

    out = pd.DataFrame({
        'dir': np.where(ul >= flat_below, 'LONG',
                        np.where(ul <= -flat_below, 'SHORT', '-')),
        'phase': [PHASE.get((int(r[p]), int(t[p])), '') if pd.notna(r[p]) else ''
                  for p in z.columns],
        'days': [int(days[p]) for p in z.columns],
        'coh': [f'{int(cl[p])}/3' for p in z.columns],
        'z': zl.round(2),
        'pct': (z.rank(pct=True).iloc[-1] * 100).round(0),
        'dz5': (az - az.shift(5)).iloc[-1].round(2),
        'u': ul.round(2),
        'wt': (ul * target_vol / vl).clip(-2, 2).round(2),
        'ndf': ['*' if (p[:3] in NDF_OR_RESTRICTED or p[3:] in NDF_OR_RESTRICTED)
                else '' for p in z.columns],
    }, index=z.columns)
    out.loc[out['dir'] == '-', 'phase'] = 'NEUTRAL'
    out['block'] = ['G10' if p in G10_PAIRS
                    else ('PM' if p in PM_PAIRS else 'EM') for p in out.index]
    return out.sort_values(['block', 'phase', 'z'],
                           ascending=[True, True, False])


def gold_silver_ratio(px: pd.DataFrame) -> pd.Series:
    """XAU/XAG. Derived from USD legs, so consistent with the pair scores."""
    return px['USDXAG'] / px['USDXAU']


if __name__ == '__main__':
    px = load_spot()
    mon = monitor(px)
    for blk in ('G10', 'EM', 'PM'):
        sub = mon[mon.block == blk]
        if len(sub):
            print(f'\n=== {blk} ===')
            print(sub.drop(columns='block').to_string())
