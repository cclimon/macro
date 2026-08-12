"""
fx_momentum.py — simple time-series momentum signal for FX.

Multi-horizon, vol-normalised trend (Baz et al. 2015 style) plus a plain
breakout fallback. Designed to slot into the trend pillar of the screener.

Sign convention: score is on the quoted pair XXXYYY.
    score > 0  ->  long XXX / short YYY
Cross scores are derived from outright USD scores for internal consistency.

Data: Bloomberg blpapi (PX_LAST, daily, active days only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (short halflife, long halflife) in business days
HORIZONS = [(8, 24), (16, 48), (32, 96)]

G10 = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "NZDUSD",
       "USDCAD", "USDCHF", "USDNOK", "USDSEK"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def fetch_spot(tickers, start, end, field="PX_LAST", yellow_key="Curncy"):
    """Daily spot history from Bloomberg. Swap for your existing bbg loader."""
    import blpapi

    opts = blpapi.SessionOptions()
    opts.setServerHost("localhost")
    opts.setServerPort(8194)
    session = blpapi.Session(opts)
    if not session.start() or not session.openService("//blp/refdata"):
        raise ConnectionError("blpapi session/service failed")

    svc = session.getService("//blp/refdata")
    req = svc.createRequest("HistoricalDataRequest")
    for t in tickers:
        req.append("securities", f"{t} {yellow_key}")
    req.append("fields", field)
    req.set("startDate", pd.Timestamp(start).strftime("%Y%m%d"))
    req.set("endDate", pd.Timestamp(end).strftime("%Y%m%d"))
    req.set("periodicitySelection", "DAILY")
    req.set("nonTradingDayFillOption", "ACTIVE_DAYS_ONLY")
    session.sendRequest(req)

    out: dict[str, dict] = {}
    try:
        while True:
            ev = session.nextEvent(500)
            for msg in ev:
                if not msg.hasElement("securityData"):
                    continue
                sd = msg.getElement("securityData")
                name = sd.getElementAsString("security").rsplit(" ", 1)[0]
                fd = sd.getElement("fieldData")
                out[name] = {
                    fd.getValue(i).getElementAsDatetime("date"):
                        fd.getValue(i).getElementAsFloat(field)
                    for i in range(fd.numValues())
                    if fd.getValue(i).hasElement(field)
                }
            if ev.eventType() == blpapi.Event.RESPONSE:
                break
    finally:
        session.stop()

    df = pd.DataFrame(out).sort_index()
    df.index = pd.to_datetime(df.index)
    return df[[t for t in tickers if t in df.columns]]


# --------------------------------------------------------------------------- #
# Signal
# --------------------------------------------------------------------------- #
def _response(z: np.ndarray | pd.DataFrame):
    """Damp extreme trend readings — peaks near |z|=1.4, decays beyond."""
    return z * np.exp(-(z ** 2) / 4.0) / 0.89


def momentum_score(px: pd.DataFrame,
                   horizons=HORIZONS,
                   price_vol_win: int = 63,
                   sig_vol_win: int = 252,
                   damp: bool = True) -> pd.DataFrame:
    """
    Continuous trend score in roughly [-1, 1], one column per pair.

    For each horizon:
        x = EWMA_short(log P) - EWMA_long(log P)      raw trend
        y = x / rolling_std(log P, 63d)               scale by price vol
        z = y / rolling_std(y, 252d)                  make comparable across pairs
    Then average across horizons.
    """
    lp = np.log(px.astype(float))
    parts = []
    for s, l in horizons:
        x = lp.ewm(halflife=s, min_periods=s).mean() - \
            lp.ewm(halflife=l, min_periods=l).mean()
        y = x / lp.rolling(price_vol_win).std()
        z = y / y.rolling(sig_vol_win).std()
        parts.append(_response(z) if damp else z.clip(-2, 2) / 2)
    return sum(parts) / len(parts)


def breakout_score(px: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    """Simplest version: 3M log return, vol-normalised. Use as a sanity check."""
    lp = np.log(px.astype(float))
    r = lp - lp.shift(lookback)
    vol = lp.diff().rolling(lookback).std() * np.sqrt(lookback)
    return (r / vol).clip(-2, 2) / 2


# --------------------------------------------------------------------------- #
# USD-leg mechanics
# --------------------------------------------------------------------------- #
def to_usd_base(px: pd.DataFrame) -> pd.DataFrame:
    """Restate every pair as USDXXX so all scores share one sign convention."""
    out = {}
    for c in px.columns:
        if c.startswith("USD"):
            out[c] = px[c]
        else:  # XXXUSD -> USDXXX
            out["USD" + c[:3]] = 1.0 / px[c]
    return pd.DataFrame(out)


def cross_score(score_usd: pd.DataFrame, base: str, quote: str) -> pd.Series:
    """
    Score for BASEQUOTE built from USD-leg scores (score_usd on USDXXX).
    score(USDXXX) > 0 means USD strong vs XXX, so:
        score(BASEQUOTE) = score(USDQUOTE) - score(USDBASE), rescaled.
    """
    s = score_usd[f"USD{quote}"] - score_usd[f"USD{base}"]
    return (s / 2.0).rename(f"{base}{quote}")


def positions(score: pd.DataFrame,
              px: pd.DataFrame,
              target_vol: float = 0.10,
              vol_win: int = 63,
              cap: float = 2.0) -> pd.DataFrame:
    """Vol-target the score into notional weights. Lagged one day, no look-ahead."""
    ann_vol = np.log(px.astype(float)).diff().rolling(vol_win).std() * np.sqrt(252)
    return (score.shift(1) * target_vol / ann_vol).clip(-cap, cap)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    px = fetch_spot(G10, "2015-01-01", pd.Timestamp.today())
    usd = to_usd_base(px)
    score = momentum_score(usd)
    w = positions(score, usd)

    print("Latest USD-leg trend scores (positive = USD strong):")
    print(score.iloc[-1].sort_values(ascending=False).round(2))
    print("\nEURJPY:", round(cross_score(score, "EUR", "JPY").iloc[-1], 2))
