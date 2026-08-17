"""
Bloomberg data layer.

Thin blpapi wrapper (historical + reference) with a parquet disk cache and a
ticker validation routine.

Design notes
------------
* Every request goes through `history()`, which caches to parquet keyed on
  (ticker, field, start). On a cache hit the last stored date is compared to
  the requested end date; only the missing tail is fetched from Bloomberg and
  appended. A cold run fetches ~500 series; subsequent runs only fetch the
  delta since the last cache write (typically 1-2 data points per series).
* `history()` NEVER raises on a bad ticker. It returns an empty Series and
  records the failure in `FAILED`. Pillars drop empty inputs and renormalise
  over survivors, so one dead ticker degrades one input rather than the run.
* Run `python bbg.py --validate` to get the failure list before trusting output.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

TIMEOUT_MS = 10_000      # per nextEvent poll
MAX_TIMEOUTS = 3         # give up on a ticker after 3 empty polls (~30s)

FAILED: dict[str, str] = {}      # ticker -> reason
_SESSION = None


# --------------------------------------------------------------------------
# Session management
# --------------------------------------------------------------------------

def _session():
    """Lazily open a blpapi session. Raises only if Bloomberg is genuinely
    unreachable — callers that can tolerate missing data should use history()."""
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    try:
        import blpapi
    except ImportError as e:
        raise ImportError(
            "blpapi not installed. In the FX_BBG conda env:\n"
            "  pip install --index-url=https://blpapi.bloomberg.com/repository/"
            "releases/python/simple/ blpapi"
        ) from e

    opts = blpapi.SessionOptions()
    opts.setServerHost("localhost")
    opts.setServerPort(8194)
    s = blpapi.Session(opts)
    if not s.start():
        raise ConnectionError("Could not start blpapi session — is the Terminal running?")
    if not s.openService("//blp/refdata"):
        raise ConnectionError("Could not open //blp/refdata")
    _SESSION = s
    return s


def _cache_path(ticker: str, field: str, start: str) -> Path:
    safe = re.sub(r"[^\w\-+.]", "_", f"{ticker}__{field}__{start}")
    return CACHE_DIR / f"{safe}.parquet"


# --------------------------------------------------------------------------
# Historical data
# --------------------------------------------------------------------------

_BBGID = re.compile(r"^BBG[A-Z0-9]{9}$")


def normalise_security(ticker: str) -> str:
    """Accept a bare BBGID and convert it to the form blpapi expects.

    CDS long names encode doc clause and restructuring convention and get
    re-versioned over time, so BBGIDs are the stable identifier for that asset
    class. Both forms are accepted; bare BBGIDs get the /bbgid/ prefix.
    """
    t = (ticker or "").strip()
    if _BBGID.match(t):
        return f"/bbgid/{t}"
    return t


def history(
    ticker: str,
    field: str = "PX_LAST",
    start: str = "2010-01-01",
    end: Optional[str] = None,
    use_cache: bool = True,
) -> pd.Series:
    """Daily history for one ticker. Returns an empty Series on failure."""
    if not ticker or ticker.startswith(("VERIFY_", "PROBE_")):
        FAILED[ticker or "<none>"] = "placeholder ticker, not configured"
        return pd.Series(dtype=float, name=ticker)

    end = end or dt.date.today().isoformat()
    end_date = dt.date.fromisoformat(end)
    cache = _cache_path(ticker, field, start)

    # --- cache delta logic -------------------------------------------
    cached: pd.Series | None = None
    fetch_start = start
    if use_cache and cache.exists():
        cached = pd.read_parquet(cache).iloc[:, 0].rename(ticker)
        cached_last = cached.index[-1].date()
        if cached_last >= end_date:
            return cached.loc[:end].rename(ticker)
        # Only fetch the missing tail from BBG
        fetch_start = (cached_last + dt.timedelta(days=1)).isoformat()

    try:
        import blpapi
        s = _session()
        svc = s.getService("//blp/refdata")
        req = svc.createRequest("HistoricalDataRequest")
        req.getElement("securities").appendValue(normalise_security(ticker))
        req.getElement("fields").appendValue(field)
        req.set("startDate", fetch_start.replace("-", ""))
        req.set("endDate", end.replace("-", ""))
        req.set("periodicitySelection", "DAILY")
        req.set("nonTradingDayFillOption", "ACTIVE_DAYS_ONLY")
        s.sendRequest(req)

        dates, vals = [], []
        timeouts = 0
        while True:
            ev = s.nextEvent(TIMEOUT_MS)
            if ev.eventType() == blpapi.Event.TIMEOUT:
                # nextEvent RETURNS a timeout event, it does not raise. Without
                # this branch the loop spins forever on an unanswered request.
                timeouts += 1
                if timeouts >= MAX_TIMEOUTS:
                    FAILED[ticker] = f"timeout after {MAX_TIMEOUTS * TIMEOUT_MS / 1000:.0f}s"
                    return cached.loc[:end].rename(ticker) if cached is not None else pd.Series(dtype=float, name=ticker)
                continue
            for msg in ev:
                sd = msg.getElement("securityData") if msg.hasElement("securityData") else None
                if sd is None:
                    continue
                if sd.hasElement("securityError"):
                    FAILED[ticker] = "security error (bad ticker or no entitlement)"
                    return cached.loc[:end].rename(ticker) if cached is not None else pd.Series(dtype=float, name=ticker)
                fd = sd.getElement("fieldData")
                for i in range(fd.numValues()):
                    row = fd.getValue(i)
                    if row.hasElement(field):
                        dates.append(pd.Timestamp(row.getElementAsDatetime("date")))
                        vals.append(row.getElementAsFloat(field))
            if ev.eventType() == blpapi.Event.RESPONSE:
                break

        new = pd.Series(vals, index=pd.DatetimeIndex(dates), name=ticker).sort_index()

        if cached is not None:
            if not new.empty:
                out = pd.concat([cached, new]).sort_index()
                out = out[~out.index.duplicated(keep="last")]
                if use_cache:
                    out.to_frame().to_parquet(cache)
            else:
                # No new trading days (weekend/holiday gap) — cache unchanged
                out = cached
        else:
            out = new
            if out.empty:
                FAILED[ticker] = "empty response"
            elif use_cache:
                out.to_frame().to_parquet(cache)

        return out.loc[:end].rename(ticker) if not out.empty else out

    except Exception as e:                                    # noqa: BLE001
        FAILED[ticker] = f"{type(e).__name__}: {e}"
        log.warning("history failed for %s: %s", ticker, e)
        return cached.loc[:end].rename(ticker) if cached is not None else pd.Series(dtype=float, name=ticker)


def frame(
    tickers: Iterable[str],
    field: str = "PX_LAST",
    start: str = "2010-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Aligned DataFrame of several tickers. Failed tickers are omitted."""
    cols = {}
    for t in tickers:
        s = history(t, field, start, end)
        if not s.empty:
            cols[t] = s
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()


def stitch(
    primary: str,
    fallback: str,
    field: str = "PX_LAST",
    start: str = "2010-01-01",
    end: Optional[str] = None,
    use_cache: bool = True,
) -> pd.Series:
    """Merge two series: primary where available, fallback for earlier dates.

    Used when a series was relaunched under a new ticker (e.g. AUD monthly CPI
    replacing the quarterly one). Primary takes precedence in any overlap period
    so the more current source always wins.

    The stitched result is cached under a combined key and updated via the same
    delta-fetch logic as history() — on subsequent calls only the primary tail
    is fetched from Bloomberg.
    """
    end = end or dt.date.today().isoformat()
    end_date = dt.date.fromisoformat(end)
    cache = _cache_path(f"{primary}+{fallback}", field, start)

    cached: pd.Series | None = None
    if use_cache and cache.exists():
        cached = pd.read_parquet(cache).iloc[:, 0].rename(primary)
        if cached.index[-1].date() >= end_date:
            return cached.loc[:end].rename(primary)
        # Only the primary series can have new data — fallback is historical
        delta_start = (cached.index[-1].date() + dt.timedelta(days=1)).isoformat()
        delta = history(primary, field, delta_start, end, use_cache=False)
        if not delta.empty:
            out = pd.concat([cached, delta]).sort_index()
            out = out[~out.index.duplicated(keep="last")]
            if use_cache:
                out.to_frame().to_parquet(cache)
            return out.loc[:end].rename(primary)
        return cached.loc[:end].rename(primary)

    # Cold fetch — get both series and merge; primary wins in overlap
    s_primary = history(primary, field, start, end, use_cache=False)
    s_fallback = history(fallback, field, start, end, use_cache=False)
    if s_primary.empty and s_fallback.empty:
        FAILED[primary] = "both primary and fallback returned no data"
        return pd.Series(dtype=float, name=primary)

    out = pd.concat([s_fallback, s_primary]).sort_index()
    out = out[~out.index.duplicated(keep="last")]   # last = primary wins
    if use_cache:
        out.to_frame().to_parquet(cache)
    return out.loc[:end].rename(primary)


def reference(tickers: Iterable[str], field: str = "PX_LAST") -> pd.Series:
    """Snapshot reference data (BDP equivalent)."""
    tickers = list(tickers)
    tickers = [t for t in tickers if t and not t.startswith(("VERIFY_", "PROBE_"))]
    if not tickers:
        return pd.Series(dtype=float)
    try:
        import blpapi
        s = _session()
        svc = s.getService("//blp/refdata")
        req = svc.createRequest("ReferenceDataRequest")
        for t in tickers:
            req.getElement("securities").appendValue(normalise_security(t))
        req.getElement("fields").appendValue(field)
        s.sendRequest(req)

        out = {}
        timeouts = 0
        while True:
            ev = s.nextEvent(TIMEOUT_MS)
            if ev.eventType() == blpapi.Event.TIMEOUT:
                timeouts += 1
                if timeouts >= MAX_TIMEOUTS:
                    log.warning("reference request timed out")
                    return pd.Series(out)
                continue
            for msg in ev:
                if not msg.hasElement("securityData"):
                    continue
                arr = msg.getElement("securityData")
                for i in range(arr.numValues()):
                    sd = arr.getValue(i)
                    tk = sd.getElementAsString("security")
                    if sd.hasElement("securityError"):
                        FAILED[tk] = "security error"
                        continue
                    fd = sd.getElement("fieldData")
                    if fd.hasElement(field):
                        out[tk] = fd.getElementAsFloat(field)
            if ev.eventType() == blpapi.Event.RESPONSE:
                break
        return pd.Series(out)
    except Exception as e:                                    # noqa: BLE001
        log.warning("reference failed: %s", e)
        return pd.Series(dtype=float)


# --------------------------------------------------------------------------
# Derived FX tickers
# --------------------------------------------------------------------------

def vol_tickers(vol_root: str, tenor: str = "3M") -> dict[str, str]:
    """ATM vol, 25d risk reversal, 25d butterfly for a pair root."""
    return {
        "atm": f"{vol_root}V{tenor} BGN Curncy",
        "rr25": f"{vol_root}25R{tenor} BGN Curncy",
        "bf25": f"{vol_root}25B{tenor} BGN Curncy",
    }


def fwd_points_ticker(root: str, tenor: str = "3M", ccy: str | None = None) -> str:
    """3m forward points / NDF ticker.

    Confirmed tickers live in config.FWD_TICKERS keyed by currency; everything
    else falls back to the generic BGN outright form.
    """
    if ccy:
        from config import FWD_CONVENTIONS
        if ccy in FWD_CONVENTIONS:
            return FWD_CONVENTIONS[ccy][0]
    return f"{root}{tenor} BGN Curncy"


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def preflight() -> bool:
    """Fail fast and loudly if Bloomberg is unreachable."""
    print("Preflight: opening blpapi session...", flush=True)
    try:
        _session()
    except Exception as e:                                    # noqa: BLE001
        print(f"  SESSION FAILED: {e}", flush=True)
        return False
    print("  session OK. Testing EURUSD Curncy...", flush=True)
    s = history("EURUSD Curncy", start="2024-01-01", use_cache=False)
    if s.empty:
        print(f"  TEST TICKER FAILED: {FAILED.get('EURUSD Curncy')}", flush=True)
        return False
    print(f"  OK, {len(s)} obs, last {s.index[-1].date()} = {s.iloc[-1]:.4f}\n", flush=True)
    return True


def validate_universe(start: str = "2023-01-01") -> pd.DataFrame:
    """Ping every ticker referenced by the config. Returns a status table."""
    from config import UNIVERSE, REGIME_TICKERS, USD_SWAP2Y, USD_CPI, USD_CESI, VOL_TENOR

    rows = []
    n = [0]

    # Expected observation frequency per role, so a monthly CPI series with 43
    # points is not flagged alongside a daily swap series with 130. Staleness
    # is judged on the LAST DATE, which is what actually matters.
    STALE_DAYS = {
        "cpi_yoy": 120, "ca_gdp": 200, "cot_long": 30, "cot_short": 30, "reer": 120, "tot": 15,
    }
    DEFAULT_STALE_DAYS = 15

    def check(ticker, owner, role):
        if not ticker:
            return
        # Tuple = stitched pair (primary, fallback) — validate primary only
        if isinstance(ticker, tuple):
            check(ticker[0], owner, role)
            return
        if ticker.startswith(("VERIFY_", "PROBE_")):
            rows.append({"ticker": ticker, "owner": owner, "role": role,
                         "status": "PLACEHOLDER", "obs": 0, "last_date": None,
                         "days_old": None})
            return
        n[0] += 1
        s = history(ticker, start=start)
        if s.empty:
            status, last, age = "FAIL", None, None
        else:
            last = s.index[-1].date()
            age = (dt.date.today() - last).days
            limit = STALE_DAYS.get(role, DEFAULT_STALE_DAYS)
            status = "STALE" if age > limit else "OK"
        print(f"  [{n[0]:>3}] {status:<5} {ticker:<28} {owner:<6} {role}"
              + (f"  last={last} ({age}d)" if last else ""), flush=True)
        rows.append({"ticker": ticker, "owner": owner, "role": role,
                     "status": status, "obs": len(s), "last_date": last,
                     "days_old": age})

    for p in UNIVERSE:
        check(p.ticker, p.ccy, "spot")
        check(fwd_points_ticker(p.name, "3M", p.ccy), p.ccy, "fwd_points")
        for role, tk in vol_tickers(p.vol_root, VOL_TENOR).items():
            check(tk, p.ccy, f"vol_{role}")
        for role in ("swap2y", "cpi_yoy", "cds5y", "reer", "tot", "cot_long", "cot_short", "cesi", "ca_gdp"):
            check(getattr(p, role), p.ccy, role)

    for role, tk in REGIME_TICKERS.items():
        check(tk, "REGIME", role)
    check(USD_SWAP2Y, "USD", "swap2y")
    check(USD_CPI, "USD", "cpi_yoy")
    check(USD_CESI, "USD", "cesi")

    df = pd.DataFrame(rows)
    return df.sort_values(["status", "owner", "role"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--test", action="store_true", help="connection check only")
    ap.add_argument("--out", default="ticker_validation.csv")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if a.test:
        raise SystemExit(0 if preflight() else 1)

    if a.validate:
        if not preflight():
            raise SystemExit("Aborting — fix the Bloomberg connection first.")
        tbl = validate_universe()
        tbl.to_csv(a.out, index=False)
        print(tbl["status"].value_counts().to_string())
        print(f"\nFull report -> {a.out}")
        bad = tbl[tbl.status != "OK"]
        if not bad.empty:
            print("\nNeeds attention:")
            print(bad.to_string(index=False))
