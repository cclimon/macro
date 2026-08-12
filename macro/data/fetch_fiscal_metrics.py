"""data/fetch_fiscal_metrics.py -- sovereign fiscal metrics puller.

Pulls the 5 sovereign-fiscal metrics for the fiscal universe (derived from the FX
vol universe) and writes an atomic JSON snapshot for the dashboards to consume.

METRICS (per country):
  1. Fiscal balance % GDP           (headline, general government)
  2. Primary balance % GDP          (ex-interest)
  3. Gross debt % GDP               (general government)
  4. Interest servicing cost        (% GDP  and  % of revenue)
  5. r - g differential             (effective rate = interest/debt; g = nominal GDP growth)

SOURCES (metric-by-metric best; per-field source + as-of are tagged in the output):
  - Bloomberg Economics  EHBB/ECBB  -> fresh headline balance actual + consensus forecast
  - IMF WEO / Fiscal Monitor VIA Bloomberg (INLB/GGL%/IGS%/GGR%/GGX%/IGNP by ISO3)
        -> general-government backbone for all metrics; interest is DERIVED
           (interest%GDP = primary - overall ; %rev = interest / revenue)
  - Eurostat / World Bank            -> gap-fill for cells Bloomberg leaves empty
        (the IMF REST API itself is WAF-blocked from this environment; the IMF data
         is obtained through Bloomberg's IMF mirror instead.)

Run:  conda run -n FX_BBG --no-capture-output python data/fetch_fiscal_metrics.py
      (from the macro/ project root; requires a live Bloomberg Terminal)
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys

import numpy as np
import pandas as pd

# make "config"/"data" importable when run from the macro/ root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.fiscal_universe import UNIVERSE, imf_ticker_bundle  # noqa: E402
from config.fiscal_universe import (  # noqa: E402
    bbg_balance_actual, bbg_balance_forecast,
)
from data.bloomberg import BloombergSession  # noqa: E402

CURRENT_YEAR = _dt.datetime.today().year
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fiscal_metrics_latest.json")

SRC_BBG_ECON = "bloomberg_economics"
SRC_IMF_BBG = "imf_weo_bloomberg"
SRC_DERIVED = "derived"
SRC_EUROSTAT = "eurostat"
SRC_WORLDBANK = "worldbank"


# ── helpers ───────────────────────────────────────────────────────────────────
def _series_by_year(combined: pd.DataFrame, ticker: str) -> dict:
    """Extract {year:int -> value:float} for one ticker from a bdh MultiIndex frame."""
    if combined is None or combined.empty or ticker not in combined.columns.get_level_values(0):
        return {}
    s = combined[(ticker, "PX_LAST")].dropna()
    out = {}
    for idx, val in s.items():
        yr = pd.Timestamp(idx).year
        out[yr] = float(val)
    return out


def _latest_actual(sd: dict):
    """(year, value) for the most recent year <= CURRENT_YEAR, else (None, None)."""
    yrs = [y for y in sd if y <= CURRENT_YEAR]
    if not yrs:
        return (None, None)
    y = max(yrs)
    return (y, sd[y])


def _val_at(sd: dict, year: int):
    return sd.get(year)


def _yoy(sd: dict, year):
    """Nominal YoY growth % at `year` (needs year and year-1)."""
    if year is None or year not in sd or (year - 1) not in sd or sd[year - 1] in (0, None):
        return None
    return (sd[year] / sd[year - 1] - 1.0) * 100.0


def _round(x, n=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


# ── Bloomberg pull ─────────────────────────────────────────────────────────────
def pull_bloomberg(bbg: BloombergSession) -> dict:
    """Return {iso2: partial_record}. Populates every metric it can from Bloomberg."""
    countries = UNIVERSE

    # 1) IMF WEO family -- one big BDH (yearly) for every ticker across all countries
    imf_tickers, ticker_owner = [], {}   # ticker -> (iso2, concept)
    for c in countries:
        for concept, tk in imf_ticker_bundle(c["iso3"]).items():
            imf_tickers.append(tk)
            ticker_owner[tk] = (c["iso2"], concept)
    start = f"{CURRENT_YEAR - 12}0101"
    print(f"[bbg] BDH {len(imf_tickers)} IMF-WEO tickers (yearly, from {start}) ...")
    imf_raw = bbg.bdh(imf_tickers, "PX_LAST", start, periodicity="YEARLY")

    # per-country IMF series
    imf = {c["iso2"]: {} for c in countries}
    for tk, (iso2, concept) in ticker_owner.items():
        imf[iso2][concept] = _series_by_year(imf_raw, tk)

    # 2) Bloomberg Economics headline balance: EHBB actual + ECBB consensus vintages
    yy0, yy1 = CURRENT_YEAR % 100, (CURRENT_YEAR + 1) % 100
    ehbb_tickers, ecbb_tickers = {}, {}
    for c in countries:
        code = c["bbg_econ_code"]
        if not code:
            continue
        ehbb_tickers[c["iso2"]] = bbg_balance_actual(code)
        ecbb_tickers[(c["iso2"], CURRENT_YEAR)] = bbg_balance_forecast(code, yy0)
        ecbb_tickers[(c["iso2"], CURRENT_YEAR + 1)] = bbg_balance_forecast(code, yy1)

    print(f"[bbg] BDP {len(ehbb_tickers)} EHBB (actual) + {len(ecbb_tickers)} ECBB (consensus) ...")
    ehbb_df = bbg.bdp(list(ehbb_tickers.values()), ["PX_LAST", "ECO_RELEASE_DT", "LATEST_ANNOUNCEMENT_PERIOD"])
    ecbb_df = bbg.bdp(list(ecbb_tickers.values()), ["PX_LAST"])

    def _bdp_get(df, ticker, field):
        try:
            v = df.loc[ticker, field]
            return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v
        except Exception:
            return None

    # 3) assemble per-country records
    records = {}
    for c in countries:
        iso2 = c["iso2"]
        ser = imf[iso2]
        rec = {
            "country": c["name"], "iso2": iso2, "iso3": c["iso3"], "group": c["group"],
            "sources": {}, "asof": {}, "tickers": {},
        }

        # -- overall / primary / debt / revenue / expenditure (IMF WEO) --
        ay_ov, v_overall = _latest_actual(ser.get("overall_balance", {}))
        # primary balance: IMF coding varies (GGL% for most, GGLP for e.g. Mexico)
        primary_series = ser.get("primary_balance") or {}
        primary_alt = "GGL%"
        if not primary_series:
            primary_series = ser.get("primary_balance_alt") or {}
            primary_alt = "GGLP"
        ay_pr, v_primary = _latest_actual(primary_series)
        ay_de, v_debt = _latest_actual(ser.get("gross_debt", {}))
        ay_re, v_rev = _latest_actual(ser.get("revenue", {}))
        gdp_series = ser.get("nominal_gdp", {})
        ay_gd, _ = _latest_actual(gdp_series)

        # --- METRIC 1: fiscal balance % GDP (prefer Bloomberg Economics EHBB) ---
        bal = _bdp_get(ehbb_df, ehbb_tickers.get(iso2), "PX_LAST") if iso2 in ehbb_tickers else None
        if bal is not None:
            rec["fiscal_balance_pct_gdp"] = _round(bal)
            rec["sources"]["fiscal_balance_pct_gdp"] = SRC_BBG_ECON
            asof = (_bdp_get(ehbb_df, ehbb_tickers[iso2], "ECO_RELEASE_DT")
                    or _bdp_get(ehbb_df, ehbb_tickers[iso2], "LATEST_ANNOUNCEMENT_PERIOD"))
            rec["asof"]["fiscal_balance_pct_gdp"] = str(asof) if asof else None
            rec["tickers"]["fiscal_balance_pct_gdp"] = ehbb_tickers[iso2]
            # consensus forecast
            f0 = _bdp_get(ecbb_df, ecbb_tickers.get((iso2, CURRENT_YEAR)), "PX_LAST")
            f1 = _bdp_get(ecbb_df, ecbb_tickers.get((iso2, CURRENT_YEAR + 1)), "PX_LAST")
            rec["fiscal_balance_forecast"] = {str(CURRENT_YEAR): _round(f0), str(CURRENT_YEAR + 1): _round(f1)}
        elif v_overall is not None:
            rec["fiscal_balance_pct_gdp"] = _round(v_overall)
            rec["sources"]["fiscal_balance_pct_gdp"] = SRC_IMF_BBG
            rec["asof"]["fiscal_balance_pct_gdp"] = f"{ay_ov} (WEO)"
            rec["tickers"]["fiscal_balance_pct_gdp"] = imf_ticker_bundle(c["iso3"])["overall_balance"]
        else:
            rec["fiscal_balance_pct_gdp"] = None
            rec["sources"]["fiscal_balance_pct_gdp"] = None

        # --- METRIC 2: primary balance % GDP (IMF WEO) ---
        if v_primary is not None:
            rec["primary_balance_pct_gdp"] = _round(v_primary)
            rec["sources"]["primary_balance_pct_gdp"] = SRC_IMF_BBG
            rec["asof"]["primary_balance_pct_gdp"] = f"{ay_pr} (WEO)"
            rec["tickers"]["primary_balance_pct_gdp"] = f"{primary_alt}{c['iso3']} Index"
        else:
            rec["primary_balance_pct_gdp"] = None
            rec["sources"]["primary_balance_pct_gdp"] = None

        # --- METRIC 3: gross debt % GDP (IMF WEO, general government) ---
        if v_debt is not None:
            rec["debt_pct_gdp"] = _round(v_debt)
            rec["sources"]["debt_pct_gdp"] = SRC_IMF_BBG
            rec["asof"]["debt_pct_gdp"] = f"{ay_de} (WEO)"
            rec["tickers"]["debt_pct_gdp"] = imf_ticker_bundle(c["iso3"])["gross_debt"]
        else:
            rec["debt_pct_gdp"] = None
            rec["sources"]["debt_pct_gdp"] = None

        # --- METRIC 4: interest servicing cost (DERIVED: primary - overall) ---
        interest_gdp = None
        if v_primary is not None and v_overall is not None:
            interest_gdp = v_primary - v_overall   # primary = overall + interest
        rec["interest_pct_gdp"] = _round(interest_gdp)
        rec["interest_pct_revenue"] = _round(
            (interest_gdp / v_rev * 100.0) if (interest_gdp is not None and v_rev) else None)
        rec["sources"]["interest_pct_gdp"] = SRC_DERIVED if interest_gdp is not None else None
        rec["sources"]["interest_pct_revenue"] = SRC_DERIVED if rec["interest_pct_revenue"] is not None else None
        rec["asof"]["interest_pct_gdp"] = f"{ay_pr} (WEO, primary-overall)" if interest_gdp is not None else None
        rec["asof"]["interest_pct_revenue"] = f"{ay_re} (WEO)" if rec["interest_pct_revenue"] is not None else None
        rec["revenue_pct_gdp"] = _round(v_rev)

        # --- METRIC 5: r - g differential ---
        g_actual = _yoy(gdp_series, ay_gd)
        g_fcst = _yoy(gdp_series, CURRENT_YEAR + 1)
        r_eff = (interest_gdp / v_debt * 100.0) if (interest_gdp is not None and v_debt) else None
        r_minus_g = (r_eff - g_actual) if (r_eff is not None and g_actual is not None) else None
        rec["nominal_gdp_growth"] = _round(g_actual)
        rec["nominal_gdp_growth_forecast"] = _round(g_fcst)
        rec["r_effective"] = _round(r_eff)
        rec["r_minus_g"] = _round(r_minus_g)
        rec["sources"]["r_minus_g"] = SRC_DERIVED if r_minus_g is not None else None
        rec["asof"]["r_minus_g"] = f"{ay_gd} (WEO)" if r_minus_g is not None else None

        records[iso2] = rec
    return records


# ── coverage report ────────────────────────────────────────────────────────────
METRIC_FIELDS = [
    "fiscal_balance_pct_gdp", "primary_balance_pct_gdp", "debt_pct_gdp",
    "interest_pct_gdp", "interest_pct_revenue", "r_minus_g",
]


def print_coverage(records: dict):
    print("\n" + "=" * 108)
    print("COVERAGE REPORT  (value | source)   -- blank = EMPTY (routes to Eurostat/World Bank gap-fill)")
    print("=" * 108)
    hdr = f'{"country":<16}' + "".join(f"{m[:15]:<17}" for m in METRIC_FIELDS)
    print(hdr); print("-" * len(hdr))
    src_short = {SRC_BBG_ECON: "bbgE", SRC_IMF_BBG: "imf", SRC_DERIVED: "der",
                 SRC_EUROSTAT: "eur", SRC_WORLDBANK: "wb", None: "-"}
    n_ok = {m: 0 for m in METRIC_FIELDS}
    for iso2, rec in records.items():
        cells = []
        for m in METRIC_FIELDS:
            v = rec.get(m)
            s = src_short.get(rec["sources"].get(m))
            if v is not None:
                n_ok[m] += 1
                cells.append(f"{v:>8} [{s}]".ljust(17))
            else:
                cells.append("".ljust(17))
        print(f'{rec["iso2"]+" "+rec["country"][:12]:<16}' + "".join(cells))
    print("-" * len(hdr))
    tot = len(records)
    summary_cells = "".join(f"{n_ok[m]}/{tot}".ljust(17) for m in METRIC_FIELDS)
    print(f'{"resolved / " + str(tot):<16}' + summary_cells)
    print("\nsource key: bbgE=Bloomberg Economics, imf=IMF WEO via Bloomberg, der=derived,")
    print("            eur=Eurostat, wb=World Bank")


# ── atomic write (macrofx pattern) ──────────────────────────────────────────────
def atomic_write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    print("=" * 108)
    print(f"SOVEREIGN FISCAL METRICS PULLER  --  {len(UNIVERSE)} countries  --  {_dt.datetime.now().isoformat(timespec='seconds')}")
    print("=" * 108)

    with BloombergSession() as bbg:
        records = pull_bloomberg(bbg)

    # TODO(next): eurostat/world-bank gap-fill for empty cells (both endpoints reachable).

    print_coverage(records)

    payload = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "reference_year": CURRENT_YEAR,
        "metrics": [
            "fiscal_balance_pct_gdp", "primary_balance_pct_gdp", "debt_pct_gdp",
            "interest_pct_gdp", "interest_pct_revenue", "r_minus_g",
        ],
        "source_legend": {
            SRC_BBG_ECON: "Bloomberg Economics (EHBB/ECBB) - headline balance actual + consensus",
            SRC_IMF_BBG: "IMF WEO / Fiscal Monitor via Bloomberg (general government, % GDP)",
            SRC_DERIVED: "Derived (interest = primary - overall; r-g = interest/debt - nominal growth)",
            SRC_EUROSTAT: "Eurostat (Maastricht basis, EU members)",
            SRC_WORLDBANK: "World Bank WDI",
        },
        "countries": list(records.values()),
    }
    atomic_write_json(OUT_PATH, payload)
    print(f"\n[written] {OUT_PATH}")


if __name__ == "__main__":
    main()
