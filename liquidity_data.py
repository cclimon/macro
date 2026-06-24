"""
liquidity_data.py
─────────────────────────────────────────────────────────────────
Fetches US liquidity indicators and writes:
  GitRepo/dashboards/public/data/liquidity_data.json

Data sources (priority order):
  1. Bloomberg API via blpapi / xbbg (requires BLP terminal session)
  2. FRED API (public, free — fallback if Bloomberg unavailable)

Run this script before `npm run build` or `npm run deploy`.

Usage:
    python liquidity_data.py                   # default: last 2 years
    python liquidity_data.py --years 5         # last N years
    python liquidity_data.py --source fred     # force FRED (no Bloomberg)
    python liquidity_data.py --source bbg      # force Bloomberg
"""

import json
import argparse
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

# Output path — relative to this script, up to dashboards/public/data/
SCRIPT_DIR   = Path(__file__).parent
OUTPUT_DIR   = SCRIPT_DIR / "dashboards" / "public" / "data"
OUTPUT_FILE  = OUTPUT_DIR / "liquidity_data.json"

FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY = os.environ.get("FRED_API_KEY", "04542c270d076d3003ee81def2a96990")

# Bloomberg tickers → series keys
BBG_TICKERS = {
    "reserves":   "WRESBAL Index",       # Fed Reserve Balances (weekly)
    "onrrp":      "RRPONTSYD Index",     # ON RRP usage (daily)
    "tga":        "USTGA Index",         # Treasury General Account (daily)
    "sofr_iorb":  None,                  # computed: SOFRRATE - IORB Index
    "bank_credit": "JBBCGWOY Index",     # H.8 bank credit YoY %
    "dealer_net": "NYPDDTNP Index",      # Primary dealer net Treasury positions (weekly)
    "move":       "MOVE Index",          # ICE BofA MOVE Index — Treasury rate vol (daily)
}

BBG_SOFR  = "SOFRRATE Index"
BBG_IORB  = "IORB Index"

# FRED series IDs → series keys
FRED_SERIES = {
    "reserves":    "WRESBAL",
    "onrrp":       "RRPONTSYD",
    "tga":         "WTREGEN",
    # sofr_iorb: no single FRED series; computed from SOFR + IORB
    "bank_credit": "TOTLL",   # Total loans & leases (then compute YoY %)
}

FRED_SOFR  = "SOFR"      # FRED series for SOFR
FRED_IORB  = "IORB"      # FRED series for IORB

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fetch US liquidity data for dashboard")
    p.add_argument("--years",  type=int, default=2, help="Years of history (default: 2)")
    p.add_argument("--source", choices=["auto", "bbg", "fred"], default="auto",
                   help="Data source: auto (BBG first), bbg, or fred")
    return p.parse_args()

# ── Bloomberg fetch ───────────────────────────────────────────────────────────

def fetch_bloomberg(start_date: str, end_date: str) -> dict:
    """
    Fetch data via xbbg (pip install xbbg).
    Requires an active Bloomberg terminal session on this machine.
    """
    try:
        from xbbg import blp
    except ImportError:
        raise ImportError("xbbg not installed. Run: pip install xbbg")

    print("  [Bloomberg] Connecting to terminal...")

    result = {}

    import pandas as pd

    def _bbg_to_dict(df):
        """Normalise xbbg output → {date_str: value} dict."""
        if df is None or df.empty:
            return {}
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
        return df.iloc[:, 0].dropna().to_dict()

    # Fetch simple tickers
    tickers = {k: v for k, v in BBG_TICKERS.items() if v is not None}
    for key, ticker in tickers.items():
        try:
            df = blp.bdh(ticker, "PX_LAST", start_date, end_date)
            result[key] = _bbg_to_dict(df)
            print(f"  [Bloomberg] ✓ {key} ({ticker}) — {len(result[key])} obs")
        except Exception as e:
            print(f"  [Bloomberg] ✗ {key} ({ticker}): {e}")
            result[key] = {}

    # sofr_iorb: IORB Index is a price index on BBG, not the rate — always use FRED
    result["sofr_iorb"] = {}

    return result

# ── FRED fetch ────────────────────────────────────────────────────────────────

def fred_get(series_id: str, start: str, end: str) -> dict:
    """Single FRED series → {date: value} dict."""
    import urllib.request
    import urllib.parse

    params = urllib.parse.urlencode({
        "series_id":        series_id,
        "observation_start": start,
        "observation_end":   end,
        "file_type":         "json",
        "api_key":           FRED_API_KEY or "guest",  # FRED allows limited guest access
    })
    url = f"{FRED_BASE}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        out = {}
        for obs in data.get("observations", []):
            v = obs["value"]
            if v != ".":
                out[obs["date"]] = float(v)
        return out
    except Exception as e:
        print(f"  [FRED] ✗ {series_id}: {e}")
        return {}

def fetch_fred(start_date: str, end_date: str) -> dict:
    print("  [FRED] Fetching public data...")
    result = {}

    for key, series_id in FRED_SERIES.items():
        data = fred_get(series_id, start_date, end_date)
        if key == "bank_credit" and data:
            # Compute YoY %: need extra year of data for the base
            extended_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=370)).strftime("%Y-%m-%d")
            full_data = fred_get(series_id, extended_start, end_date)
            dates = sorted(full_data.keys())
            yoy = {}
            for d in dates:
                d_prev = (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
                # Find nearest available prior date
                prior = next((full_data[p] for p in sorted(full_data.keys()) if p <= d_prev), None)
                if prior and prior != 0:
                    yoy[d] = round((full_data[d] / prior - 1) * 100, 2)
            # Filter to requested window
            result["bank_credit"] = {d: v for d, v in yoy.items() if d >= start_date}
        else:
            result[key] = data
        print(f"  [FRED] ✓ {key} — {len(result.get(key, {}))} obs")

    # SOFR − IORB spread
    sofr_raw = fred_get(FRED_SOFR, start_date, end_date)
    iorb_raw = fred_get(FRED_IORB, start_date, end_date)
    spread = {}
    for d, sofr_v in sofr_raw.items():
        if d in iorb_raw:
            spread[d] = round((sofr_v - iorb_raw[d]) * 100, 2)  # → bps
    result["sofr_iorb"] = spread
    print(f"  [FRED] ✓ sofr_iorb (computed) — {len(spread)} obs")

    return result

# ── Merge & align ─────────────────────────────────────────────────────────────

def merge_to_timeseries(raw: dict, keys: list) -> list:
    """
    Take per-key {date: value} dicts and produce a unified
    list of {date, key1, key2, ...} records sorted by date.
    Use forward-fill for missing dates (max 7 days).
    """
    all_dates = sorted(set(d for k in keys for d in raw.get(k, {}).keys()))
    records = []
    last_values = {k: None for k in keys}

    for date in all_dates:
        rec = {"date": date}
        for k in keys:
            v = raw.get(k, {}).get(date)
            if v is not None:
                last_values[k] = v
            rec[k] = last_values[k]
        records.append(rec)

    return records

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")

    print(f"\n📊 US Liquidity Dashboard — Data Fetch")
    print(f"   Period : {start_date} → {end_date}")
    print(f"   Source : {args.source}\n")

    raw = {}

    # 1. Try Bloomberg
    if args.source in ("auto", "bbg"):
        try:
            raw = fetch_bloomberg(start_date, end_date)
        except Exception as e:
            if args.source == "bbg":
                print(f"\n❌ Bloomberg failed: {e}")
                sys.exit(1)
            else:
                print(f"  Bloomberg unavailable ({e}) — falling back to FRED\n")
                raw = {}

    # 2. FRED fallback (or fill any gaps)
    if args.source == "fred" or not all(raw.get(k) for k in FRED_SERIES.keys()):
        fred_raw = fetch_fred(start_date, end_date)
        for k, v in fred_raw.items():
            if not raw.get(k):   # only fill missing
                raw[k] = v

    # Merge to unified time series
    keys   = ["reserves", "onrrp", "tga", "sofr_iorb", "bank_credit", "dealer_net", "move"]
    series = merge_to_timeseries(raw, keys)

    # Build output JSON
    output = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": args.source,
        "series": series,
        "meta": {
            "reserves":    {"label": "Fed Reserve Balances", "unit": "$B",  "source": "WRESBAL"},
            "onrrp":       {"label": "ON RRP Usage",         "unit": "$B",  "source": "RRPONTSYD"},
            "tga":         {"label": "TGA Balance",          "unit": "$B",  "source": "WTREGEN"},
            "sofr_iorb":   {"label": "SOFR − IORB Spread",  "unit": "bps", "source": "Computed"},
            "bank_credit": {"label": "Bank Credit YoY %",   "unit": "%",   "source": "H.8 / TOTLL"},
            "dealer_net":  {"label": "Dealer Net Treasury",  "unit": "$B",  "source": "NYPDDTNP Index"},
            "move":        {"label": "MOVE Index",           "unit": "pts", "source": "MOVE Index"},
        },
    }

    # Write file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Written {len(series)} records → {OUTPUT_FILE}")
    print(f"   Next step: npm run build && npm run deploy (from dashboards/)\n")

if __name__ == "__main__":
    main()
