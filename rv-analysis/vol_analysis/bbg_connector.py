"""
bbg_connector.py — blpapi session management and BDH (historical data) wrapper.

All Bloomberg calls in this project go through `bdh()`.
"""

from __future__ import annotations

import datetime
from typing import Union

import blpapi
import pandas as pd


# ── Session singleton ─────────────────────────────────────────────────────────

_session: blpapi.Session | None = None
_ref_data_service: blpapi.Service | None = None


def _get_session() -> tuple[blpapi.Session, blpapi.Service]:
    """Open and cache a Bloomberg session + reference-data service."""
    global _session, _ref_data_service

    if _session is not None:
        return _session, _ref_data_service

    options = blpapi.SessionOptions()
    options.setServerHost("localhost")
    options.setServerPort(8194)

    session = blpapi.Session(options)
    if not session.start():
        raise ConnectionError("Failed to start Bloomberg session. Is the Bloomberg terminal running?")
    if not session.openService("//blp/refdata"):
        raise ConnectionError("Failed to open //blp/refdata service.")

    _session = session
    _ref_data_service = session.getService("//blp/refdata")
    return _session, _ref_data_service


def stop_session() -> None:
    """Gracefully stop the Bloomberg session (call at end of script)."""
    global _session, _ref_data_service
    if _session is not None:
        _session.stop()
        _session = None
        _ref_data_service = None


# ── BDH wrapper ───────────────────────────────────────────────────────────────

def bdh(
    tickers: Union[str, list[str]],
    fields:  Union[str, list[str]],
    start_date: Union[str, datetime.date],
    end_date:   Union[str, datetime.date],
    periodicity: str = "DAILY",
) -> pd.DataFrame:
    """Fetch historical Bloomberg data (BDH equivalent).

    Parameters
    ----------
    tickers:     One or more Bloomberg tickers.
    fields:      One or more Bloomberg field names.
    start_date:  Start date as "YYYYMMDD" string or datetime.date.
    end_date:    End date as "YYYYMMDD" string or datetime.date.
    periodicity: Bloomberg periodicitySelection override (default "DAILY").

    Returns
    -------
    pd.DataFrame with a DatetimeIndex and MultiIndex columns (ticker, field)
    when multiple tickers are requested, or (field,) columns for a single ticker.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(fields, str):
        fields = [fields]

    start_str = _to_bbg_date(start_date)
    end_str   = _to_bbg_date(end_date)

    session, svc = _get_session()

    request = svc.createRequest("HistoricalDataRequest")
    for t in tickers:
        request.append("securities", t)
    for f in fields:
        request.append("fields", f)
    request.set("periodicitySelection", periodicity)
    request.set("startDate", start_str)
    request.set("endDate", end_str)

    session.sendRequest(request)

    raw: dict[str, dict[str, list]] = {t: {"date": [], **{f: [] for f in fields}} for t in tickers}

    while True:
        event = session.nextEvent(500)
        for msg in event:
            if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                sec_data  = msg.getElement("securityData")
                ticker_in = sec_data.getElementAsString("security")
                field_data = sec_data.getElement("fieldData")

                for i in range(field_data.numValues()):
                    point = field_data.getValueAsElement(i)
                    date_val = point.getElementAsDatetime("date")
                    raw[ticker_in]["date"].append(
                        datetime.date(date_val.year, date_val.month, date_val.day)
                    )
                    for f in fields:
                        try:
                            raw[ticker_in][f].append(point.getElementAsFloat(f))
                        except Exception:
                            raw[ticker_in][f].append(float("nan"))

        if event.eventType() == blpapi.Event.RESPONSE:
            break

    # Build per-ticker DataFrames then concatenate
    dfs = []
    for t in tickers:
        df = pd.DataFrame(raw[t]).set_index("date")
        df.index = pd.to_datetime(df.index)
        df.columns = pd.MultiIndex.from_tuples([(t, f) for f in fields], names=["ticker", "field"])
        dfs.append(df)

    result = pd.concat(dfs, axis=1)

    # Flatten to simple column index when only one ticker is requested
    if len(tickers) == 1:
        result.columns = result.columns.get_level_values("field")

    return result


# ── IntradayBar wrapper ───────────────────────────────────────────────────────

def intraday_bars(
    ticker: str,
    start_dt: datetime.datetime,
    end_dt: datetime.datetime,
    interval: int = 30,
    event_type: str = "BID",
) -> pd.DataFrame:
    """Fetch intraday OHLC bars from Bloomberg via IntradayBarRequest.

    Parameters
    ----------
    ticker:     Bloomberg ticker (e.g. 'EURUSD Curncy').
    start_dt:   Start datetime (UTC).
    end_dt:     End datetime (UTC).
    interval:   Bar size in minutes (default 30).
    event_type: Bloomberg event type — 'BID' for FX, 'TRADE' for equities.

    Returns
    -------
    pd.DataFrame with DatetimeIndex and columns [open, high, low, close].
    """
    session, svc = _get_session()

    request = svc.createRequest("IntradayBarRequest")
    request.set("security",      ticker)
    request.set("eventType",     event_type)
    request.set("interval",      interval)
    request.set("startDateTime", start_dt)
    request.set("endDateTime",   end_dt)

    session.sendRequest(request)

    rows = []
    while True:
        event = session.nextEvent(500)
        for msg in event:
            if msg.messageType() == blpapi.Name("IntradayBarResponse"):
                bar_data = msg.getElement("barData").getElement("barTickData")
                for i in range(bar_data.numValues()):
                    bar = bar_data.getValueAsElement(i)
                    t   = bar.getElementAsDatetime("time")
                    rows.append({
                        "time":  datetime.datetime(t.year, t.month, t.day, t.hour, t.minute),
                        "open":  bar.getElementAsFloat("open"),
                        "high":  bar.getElementAsFloat("high"),
                        "low":   bar.getElementAsFloat("low"),
                        "close": bar.getElementAsFloat("close"),
                    })
        if event.eventType() == blpapi.Event.RESPONSE:
            break

    if not rows:
        raise ValueError(f"No intraday bars returned for {ticker}.")

    df = pd.DataFrame(rows).set_index("time")
    df.index = pd.to_datetime(df.index)
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_bbg_date(d: Union[str, datetime.date]) -> str:
    """Convert a date value to Bloomberg's "YYYYMMDD" string format."""
    if isinstance(d, str):
        # Accept ISO format yyyy-mm-dd or already correct yyyymmdd
        d = d.replace("-", "")
        if len(d) != 8:
            raise ValueError(f"Unrecognised date string: {d!r}")
        return d
    return d.strftime("%Y%m%d")
