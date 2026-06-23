# data/bloomberg.py
# Bloomberg data acquisition layer using blpapi

import blpapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

# ── Bloomberg session ──────────────────────────────────────────────────────────

class BloombergSession:
    """Manages a single blpapi session. Use as context manager or call open/close."""

    def __init__(self, host: str = "localhost", port: int = 8194):
        self.host = host
        self.port = port
        self._session: Optional[blpapi.Session] = None
        self._ref_service = None
        self._hist_service = None

    def open(self):
        opts = blpapi.SessionOptions()
        opts.setServerHost(self.host)
        opts.setServerPort(self.port)
        self._session = blpapi.Session(opts)
        if not self._session.start():
            raise ConnectionError("Bloomberg session failed to start. Check terminal connection.")
        if not self._session.openService("//blp/refdata"):
            raise ConnectionError("Could not open Bloomberg //blp/refdata service.")
        self._ref_service = self._session.getService("//blp/refdata")
        logger.info("Bloomberg session open.")
        return self

    def close(self):
        if self._session:
            self._session.stop()
            logger.info("Bloomberg session closed.")

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()

    # ── BDP — single field, latest value ──────────────────────────────────────

    def bdp(
        self,
        tickers: Union[str, List[str]],
        fields: Union[str, List[str]],
    ) -> pd.DataFrame:
        """Bloomberg BDP (reference data) — returns latest point per ticker/field."""
        if isinstance(tickers, str):
            tickers = [tickers]
        if isinstance(fields, str):
            fields = [fields]

        request = self._ref_service.createRequest("ReferenceDataRequest")
        for t in tickers:
            request.getElement("securities").appendValue(t)
        for f in fields:
            request.getElement("fields").appendValue(f)

        self._session.sendRequest(request)
        results = {t: {} for t in tickers}

        while True:
            event = self._session.nextEvent(500)
            for msg in event:
                if msg.messageType() == blpapi.Name("ReferenceDataResponse"):
                    sec_data = msg.getElement("securityData")
                    for i in range(sec_data.numValues()):
                        sd = sec_data.getValueAsElement(i)
                        ticker = sd.getElementAsString("security")
                        fd = sd.getElement("fieldData")
                        for f in fields:
                            try:
                                results[ticker][f] = fd.getElementAsFloat(f)
                            except Exception:
                                results[ticker][f] = np.nan
            if event.eventType() == blpapi.Event.RESPONSE:
                break

        return pd.DataFrame(results).T

    # ── BDH — historical time series ──────────────────────────────────────────

    def bdh(
        self,
        tickers: Union[str, List[str]],
        fields: Union[str, List[str]],
        start_date: str,          # "YYYYMMDD"
        end_date: str = "",       # defaults to today
        periodicity: str = "DAILY",
    ) -> pd.DataFrame:
        """Bloomberg BDH (historical data). Returns MultiIndex (ticker, field) or
        simple DataFrame when single ticker + field."""
        if isinstance(tickers, str):
            tickers = [tickers]
        if isinstance(fields, str):
            fields = [fields]
        if not end_date:
            end_date = datetime.today().strftime("%Y%m%d")

        request = self._ref_service.createRequest("HistoricalDataRequest")
        for t in tickers:
            request.getElement("securities").appendValue(t)
        for f in fields:
            request.getElement("fields").appendValue(f)
        request.set("startDate", start_date)
        request.set("endDate", end_date)
        request.set("periodicitySelection", periodicity)

        self._session.sendRequest(request)
        frames = {}

        while True:
            event = self._session.nextEvent(500)
            for msg in event:
                if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                    sec_data = msg.getElement("securityData")
                    ticker = sec_data.getElementAsString("security")
                    fd_array = sec_data.getElement("fieldData")
                    rows = []
                    for j in range(fd_array.numValues()):
                        row = fd_array.getValueAsElement(j)
                        d = {"date": row.getElementAsDatetime("date")}
                        for f in fields:
                            try:
                                d[f] = row.getElementAsFloat(f)
                            except Exception:
                                d[f] = np.nan
                        rows.append(d)
                    if rows:
                        df = pd.DataFrame(rows).set_index("date")
                        df.index = pd.to_datetime(df.index)
                        frames[ticker] = df
            if event.eventType() == blpapi.Event.RESPONSE:
                break

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1)
        combined.columns.names = ["ticker", "field"]
        return combined

# ── Convenience helpers ────────────────────────────────────────────────────────

def fetch_spot_history(
    bbg: BloombergSession,
    tickers: List[str],
    days: int = 400,
) -> pd.DataFrame:
    """Return daily PX_LAST for all spot tickers. Columns = ticker."""
    start = (datetime.today() - timedelta(days=int(days * 1.5))).strftime("%Y%m%d")
    raw = bbg.bdh(tickers, "PX_LAST", start)
    if raw.empty:
        return pd.DataFrame()
    df = raw.xs("PX_LAST", level="field", axis=1)
    df.columns = [c.replace(" Curncy", "").replace(" Index", "") for c in df.columns]
    return df.sort_index().dropna(how="all")


def fetch_latest_rates(
    bbg: BloombergSession,
    rate_tickers: Dict[str, str],
) -> pd.Series:
    """Return latest PX_LAST for a dict of {currency: bloomberg_ticker}."""
    inv = {v: k for k, v in rate_tickers.items()}
    df = bbg.bdp(list(rate_tickers.values()), "PX_LAST")
    if df.empty:
        return pd.Series(dtype=float)
    s = df["PX_LAST"].rename(index=inv)
    return s.astype(float)


def fetch_macro_latest(
    bbg: BloombergSession,
    ticker_dict: Dict[str, str],
    field: str = "PX_LAST",
) -> pd.Series:
    """Generic latest-value fetch for macro tickers {ccy: bbg_ticker}."""
    valid = {k: v for k, v in ticker_dict.items() if v is not None}
    inv = {v: k for k, v in valid.items()}
    df = bbg.bdp(list(valid.values()), field)
    if df.empty:
        return pd.Series(dtype=float)
    return df[field].rename(index=inv).astype(float)


def fetch_macro_history(
    bbg: BloombergSession,
    ticker_dict: Dict[str, str],
    days: int = 400,
    periodicity: str = "MONTHLY",
) -> pd.DataFrame:
    """Return monthly history for macro tickers. Columns = currency code."""
    valid = {k: v for k, v in ticker_dict.items() if v is not None}
    inv = {v: k for k, v in valid.items()}
    start = (datetime.today() - timedelta(days=int(days * 1.5))).strftime("%Y%m%d")
    raw = bbg.bdh(list(valid.values()), "PX_LAST", start, periodicity=periodicity)
    if raw.empty:
        return pd.DataFrame()
    df = raw.xs("PX_LAST", level="field", axis=1)
    df.columns = [inv.get(c, c) for c in df.columns]
    return df.sort_index()
