# config/fiscal_universe.py
# Sovereign-fiscal dashboard universe.
#
# The country list is DERIVED from the FX vol dashboard universe (config/vol_universe.py)
# rather than redefined: we take the distinct currencies traded there and map each to its
# sovereign issuer.  The euro is a currency bloc -> expanded to the major EMU sovereigns
# (DE/FR/IT/ES) because a sovereign-fiscal view without France/Italy would be misleading.
#
# Bloomberg field conventions confirmed live via data/discover_fiscal_fields.py and
# data/discover_imf_family.py (see data/discover_fiscal_report.txt).  Two families:
#   - Bloomberg Economics headline balance:  EHBB<CC> Index (actual), ECBB<CC> <YY> Index (consensus)
#   - IMF WEO / Fiscal Monitor (general govt, % GDP, by ISO3), pulled THROUGH Bloomberg:
#         INLB<ISO3>  overall balance %GDP      GGR%<ISO3>  revenue %GDP
#         GGL%<ISO3>  primary balance %GDP      GGX%<ISO3>  expenditure %GDP
#         IGS%<ISO3>  gross debt %GDP           IGNP<ISO3>  nominal GDP (nat ccy, current prices)

from config.vol_universe import UNIVERSE as _FX_UNIVERSE


# ── currency -> sovereign(s) ─────────────────────────────────────────────────
# one entry per distinct FX currency; EUR expands to the EMU majors.
_CCY_TO_COUNTRIES = {
    "USD": ["US"],
    "EUR": ["DE", "FR", "IT", "ES"],   # euro bloc -> EMU majors
    "GBP": ["GB"],
    "JPY": ["JP"],
    "CHF": ["CH"],
    "CAD": ["CA"],
    "AUD": ["AU"],
    "NZD": ["NZ"],
    "NOK": ["NO"],
    "SEK": ["SE"],
    "MXN": ["MX"],
    "ZAR": ["ZA"],
    "BRL": ["BR"],
    "TRY": ["TR"],
    "PLN": ["PL"],
    "HUF": ["HU"],
    "CNH": ["CN"], "CNY": ["CN"],
    "KRW": ["KR"],
    "INR": ["IN"],
}

# ── per-country reference data ───────────────────────────────────────────────
# iso2 -> (name, iso3, group, bbg_econ_code, is_euro_area)
#   bbg_econ_code = country code used by the Bloomberg Economics EHBB/ECBB family
#     (mostly the iso2, but Japan is 'JPY' -- confirmed live; None -> resolve at runtime).
_COUNTRY_META = {
    "US": ("United States", "USA", "G10", "US",  False),
    "DE": ("Germany",       "DEU", "G10", "DE",  True),
    "FR": ("France",        "FRA", "G10", "FR",  True),
    "IT": ("Italy",         "ITA", "G10", "IT",  True),
    "ES": ("Spain",         "ESP", "G10", "SP",  True),   # BBG uses SP for Spain in some econ codes
    "GB": ("United Kingdom","GBR", "G10", "UK",  False),  # BBG econ uses UK
    "JP": ("Japan",         "JPN", "G10", "JPY", False),  # confirmed: EHBBJPY
    "CH": ("Switzerland",   "CHE", "G10", "SZ",  False),
    "CA": ("Canada",        "CAN", "G10", "CA",  False),
    "AU": ("Australia",     "AUS", "G10", "AU",  False),
    "NZ": ("New Zealand",   "NZL", "G10", "NZ",  False),
    "NO": ("Norway",        "NOR", "G10", "NO",  False),
    "SE": ("Sweden",        "SWE", "G10", "SW",  False),
    "MX": ("Mexico",        "MEX", "EM",  "MX",  False),
    "ZA": ("South Africa",  "ZAF", "EM",  "ZA",  False),
    "BR": ("Brazil",        "BRA", "EM",  "BR",  False),
    "TR": ("Turkey",        "TUR", "EM",  "TU",  False),
    "PL": ("Poland",        "POL", "EM",  "PO",  False),
    "HU": ("Hungary",       "HUN", "EM",  "HU",  False),
    "CN": ("China",         "CHN", "EM",  "CH",  False),   # NB BBG econ code for China is CH
    "KR": ("South Korea",   "KOR", "EM",  "SK",  False),
    "IN": ("India",         "IND", "EM",  "IN",  False),
}


def _distinct_currencies():
    """Distinct currencies appearing in the FX vol universe pairs, order-preserved."""
    seen, out = set(), []
    for pair in _FX_UNIVERSE:
        for ccy in (pair[:3], pair[3:6]):
            if ccy and ccy not in seen:
                seen.add(ccy); out.append(ccy)
    return out


def build_universe():
    """Return ordered list of country dicts derived from the FX universe."""
    seen, out = set(), []
    for ccy in _distinct_currencies():
        for iso2 in _CCY_TO_COUNTRIES.get(ccy, []):
            if iso2 in seen or iso2 not in _COUNTRY_META:
                continue
            seen.add(iso2)
            name, iso3, group, econ, euro = _COUNTRY_META[iso2]
            out.append({
                "iso2": iso2, "iso3": iso3, "name": name, "group": group,
                "bbg_econ_code": econ, "is_euro_area": euro, "from_ccy": ccy,
            })
    return out


UNIVERSE = build_universe()
ISO3 = {c["iso2"]: c["iso3"] for c in UNIVERSE}


# ── Bloomberg ticker builders ────────────────────────────────────────────────
# IMF WEO / Fiscal Monitor family (general government, % GDP unless noted).
def imf_overall_balance(iso3):   return f"INLB{iso3} Index"   # net lending/borrowing %GDP
def imf_primary_balance(iso3):   return f"GGL%{iso3} Index"   # primary net lending/borrowing %GDP
def imf_primary_balance_alt(iso3): return f"GGLP{iso3} Index"  # alt IMF coding (e.g. Mexico) %GDP
def imf_gross_debt(iso3):        return f"IGS%{iso3} Index"   # gross debt %GDP
def imf_revenue(iso3):           return f"GGR%{iso3} Index"   # revenue %GDP
def imf_expenditure(iso3):       return f"GGX%{iso3} Index"   # expenditure %GDP
def imf_nominal_gdp(iso3):       return f"IGNP{iso3} Index"   # GDP current prices, national ccy


def imf_ticker_bundle(iso3):
    """All IMF-basis tickers for one country (concept -> ticker)."""
    return {
        "overall_balance": imf_overall_balance(iso3),
        "primary_balance": imf_primary_balance(iso3),
        "primary_balance_alt": imf_primary_balance_alt(iso3),
        "gross_debt":      imf_gross_debt(iso3),
        "revenue":         imf_revenue(iso3),
        "expenditure":     imf_expenditure(iso3),
        "nominal_gdp":     imf_nominal_gdp(iso3),
    }


# Bloomberg Economics headline budget balance (% GDP).
def bbg_balance_actual(econ_code):        return f"EHBB{econ_code} Index"          # latest actual (quarterly)
def bbg_balance_actual_yearly(econ_code): return f"EHBB{econ_code}Y Index"         # latest actual (yearly)
def bbg_balance_forecast(econ_code, yy):  return f"ECBB{econ_code} {yy:02d} Index"  # consensus fcst, year vintage


if __name__ == "__main__":
    print(f"Fiscal universe: {len(UNIVERSE)} countries "
          f"(derived from {len(_distinct_currencies())} FX currencies)\n")
    for c in UNIVERSE:
        print(f"  {c['iso2']} {c['iso3']} {c['name']:<16} {c['group']:<5} "
              f"econ={c['bbg_econ_code']:<4} euro={c['is_euro_area']}  <- {c['from_ccy']}")
