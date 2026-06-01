"""
test_intraday.py — quick trial: fetch today's 30-min EURUSD bars via IntradayBarRequest.

Run from the RVanalysis directory:
    python test_intraday.py
"""

import datetime
import blpapi
import pandas as pd


def fetch_intraday_bars(
    ticker: str,
    interval_minutes: int = 30,
    start_dt: datetime.datetime | None = None,
    end_dt:   datetime.datetime | None = None,
) -> pd.DataFrame:
    """Fetch intraday OHLCV bars from Bloomberg via IntradayBarRequest."""
    now = datetime.datetime.utcnow()
    if end_dt is None:
        end_dt = now
    if start_dt is None:
        # Default: start of today UTC
        start_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)

    options = blpapi.SessionOptions()
    options.setServerHost("localhost")
    options.setServerPort(8194)

    session = blpapi.Session(options)
    if not session.start():
        raise ConnectionError("Bloomberg session failed to start — is the terminal running?")
    if not session.openService("//blp/refdata"):
        raise ConnectionError("Failed to open //blp/refdata service.")

    svc     = session.getService("//blp/refdata")
    request = svc.createRequest("IntradayBarRequest")

    request.set("security",       ticker)
    request.set("eventType",      "BID")          # BID mid for FX; switch to TRADE for equities
    request.set("interval",       interval_minutes)
    request.set("startDateTime",  start_dt)
    request.set("endDateTime",    end_dt)

    session.sendRequest(request)

    rows = []
    while True:
        event = session.nextEvent(500)
        for msg in event:
            if msg.messageType() == blpapi.Name("IntradayBarResponse"):
                bar_data = msg.getElement("barData").getElement("barTickData")
                for i in range(bar_data.numValues()):
                    bar  = bar_data.getValueAsElement(i)
                    t    = bar.getElementAsDatetime("time")
                    rows.append({
                        "time":   datetime.datetime(t.year, t.month, t.day, t.hour, t.minute),
                        "open":   bar.getElementAsFloat("open"),
                        "high":   bar.getElementAsFloat("high"),
                        "low":    bar.getElementAsFloat("low"),
                        "close":  bar.getElementAsFloat("close"),
                        "volume": bar.getElementAsInteger("volume"),
                    })
        if event.eventType() == blpapi.Event.RESPONSE:
            break

    session.stop()

    if not rows:
        print("WARNING: no bars returned — check ticker, hours, or BBG permissions.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("time")
    df.index = pd.to_datetime(df.index)
    return df


if __name__ == "__main__":
    TICKER = "EURUSD Curncy"
    print(f"Fetching 30-min intraday bars for {TICKER} — today UTC ...")

    df = fetch_intraday_bars(TICKER, interval_minutes=30)

    if df.empty:
        print("No data returned.")
    else:
        print(f"\nShape: {df.shape}   ({len(df)} bars)")
        print(f"Range: {df.index[0]}  ->  {df.index[-1]}\n")
        print(df.to_string())
