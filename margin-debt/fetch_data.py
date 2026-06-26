"""
fetch_data.py
Downloads FINRA margin statistics, merges FRED GDP + market cap, saves margin_data.csv.
Run: python fetch_data.py
"""

import os
import sys
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import BytesIO
from pathlib import Path
from fredapi import Fred
from dotenv import load_dotenv

FINRA_PAGE = "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics"
OUT_FILE   = Path(__file__).parent / "margin_data.csv"
ENV_FILE   = Path(__file__).parent.parent / ".env"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


# ── FINRA ─────────────────────────────────────────────────────────────────────

def get_excel_url() -> str:
    r = requests.get(FINRA_PAGE, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".xlsx" in href or ".xls" in href:
            return href if href.startswith("http") else "https://www.finra.org" + href
    raise ValueError("No Excel link found on FINRA margin statistics page")


def fetch_margin_excel() -> BytesIO:
    url = get_excel_url()
    print(f"  Downloading: {url}")
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return BytesIO(r.content)


def parse_margin_excel(buf: BytesIO) -> pd.DataFrame:
    xl = pd.ExcelFile(buf)
    print(f"  Sheets found: {xl.sheet_names}")

    # Pick 'Customer Margin Balances' or first sheet containing 'margin'/'customer'
    sheet = next(
        (s for s in xl.sheet_names
         if any(kw in s.lower() for kw in ("customer", "margin", "balance"))),
        xl.sheet_names[0],
    )
    print(f"  Using sheet: '{sheet}'")

    raw = pd.read_excel(buf, sheet_name=sheet, header=None)

    # Find the header row — look for 'debit' anywhere
    header_row = next(
        (i for i, row in raw.iterrows()
         if any("debit" in str(v).lower() for v in row.values)),
        0,
    )
    print(f"  Header detected at row {header_row}")

    df = pd.read_excel(buf, sheet_name=sheet, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    # Map columns by keyword
    rename = {}
    for col in df.columns:
        low = col.lower()
        if "debit" in low:
            rename[col] = "debit"
        elif "cash" in low:
            rename[col] = "free_credit_cash"
        elif "securities" in low or "margin" in low:
            rename[col] = "free_credit_margin"
        elif any(k in low for k in ("month", "date", "year", "period")) or col == df.columns[0]:
            if "date" not in rename.values():
                rename[col] = "date"

    df = df.rename(columns=rename)

    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df.set_index("date").sort_index()

    keep = [c for c in ("debit", "free_credit_cash", "free_credit_margin") if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce").dropna(how="all")

    if "free_credit_cash" in df.columns and "free_credit_margin" in df.columns:
        df["total_free_credit"] = df["free_credit_cash"] + df["free_credit_margin"]
    else:
        raise ValueError("Could not find both free-credit columns in Excel")

    # Normalise index to month-start
    df.index = df.index.to_period("M").to_timestamp()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# ── FRED ──────────────────────────────────────────────────────────────────────

def fetch_fred(api_key: str) -> tuple[pd.Series, pd.Series]:
    fred = Fred(api_key=api_key)

    print("  Fetching GDP (nominal, SAAR, billions)…")
    gdp = fred.get_series("GDP")           # billions, quarterly SAAR
    gdp.index = pd.to_datetime(gdp.index).to_period("M").to_timestamp()

    # Total equity + investment fund shares, all sectors, millions (FRED Z.1 accounts, quarterly)
    # Note: WILL5000INDFC (Wilshire Full Cap) was discontinued by FRED June 2024.
    # BOGZ1LM893064105Q is the best available proxy — includes mutual fund double-counting so
    # the absolute ratio is lower than Wilshire-based estimates, but the trend is identical.
    print("  Fetching BOGZ1LM893064105Q (total equity, all sectors, millions)…")
    mktcap = fred.get_series("BOGZ1LM893064105Q")   # millions, quarterly
    mktcap.index = pd.to_datetime(mktcap.index).to_period("M").to_timestamp()

    return gdp, mktcap


def quarterly_to_monthly(series: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill a quarterly series onto a monthly DatetimeIndex."""
    combined = series.reindex(idx.union(series.index)).sort_index().ffill()
    return combined.reindex(idx)


# ── Main ──────────────────────────────────────────────────────────────────────

def build():
    load_dotenv(ENV_FILE)
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        sys.exit("Error: FRED_API_KEY not found in .env")

    print("\n[1/3] Fetching FINRA margin data…")
    buf = fetch_margin_excel()
    margin = parse_margin_excel(buf)
    print(f"  Parsed {len(margin)} monthly rows  ({margin.index[0]:%b %Y} – {margin.index[-1]:%b %Y})")

    print("\n[2/3] Fetching FRED series…")
    gdp, mktcap = fetch_fred(api_key)

    print("\n[3/3] Merging & computing ratios…")
    idx = margin.index
    gdp_m    = quarterly_to_monthly(gdp,    idx)   # billions
    mktcap_m = quarterly_to_monthly(mktcap, idx)   # millions

    df = margin.copy()
    df["gdp_b"]       = gdp_m
    df["mktcap_m"]    = mktcap_m   # BOGZ1LM893064105Q in millions

    # Debit (millions) / GDP (billions → millions via *1000) * 100
    df["debt_gdp_pct"]    = df["debit"] / (df["gdp_b"] * 1_000) * 100
    # Debit (millions) / MktCap (millions) * 100
    df["debt_mktcap_pct"] = df["debit"] / df["mktcap_m"] * 100
    # Net leverage ratio
    df["net_leverage"]    = df["debit"] / df["total_free_credit"]

    df.to_csv(OUT_FILE)
    print(f"\n  Saved: {OUT_FILE}  ({len(df)} rows)")

    latest = df.dropna(subset=["debt_gdp_pct"]).iloc[-1]
    print(f"\n  Latest snapshot ({df.index[-1]:%b %Y}):")
    print(f"    Debit balance   : ${latest['debit']:>12,.0f}M")
    print(f"    Debt / GDP      : {latest['debt_gdp_pct']:.2f}%")
    print(f"    Debt / Mkt Cap  : {latest['debt_mktcap_pct']:.2f}%")
    print(f"    Net leverage    : {latest['net_leverage']:.2f}×")


if __name__ == "__main__":
    build()
