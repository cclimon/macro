"""
FX Screener — configuration.

Universe, quote conventions, Bloomberg ticker map, pillar weights.

QUOTE CONVENTION
----------------
Every signal in this library is expressed as "attractiveness of being LONG the
non-USD currency vs USD". Bloomberg tickers follow market convention, so for
USD-base pairs (USDBRL, USDJPY) a rising ticker means the currency is WEAKER.
The `sign` property handles that inversion once, centrally. Never invert
anywhere else.

TICKER VERIFICATION
-------------------
Tickers marked VERIFY below are best-guess conventions and depend on your
entitlements. Run `python -m fx_screener.bbg --validate` before trusting output:
it pings every ticker and reports which fail to resolve. Anything unresolved is
dropped from its pillar and the pillar is renormalised over what survives, so a
bad ticker degrades one input rather than silently NaN-ing a whole currency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------------
# Horizon / engine settings
# --------------------------------------------------------------------------

HORIZON = "core"            # 1-3 month signal horizon
CARRY_TENOR_DAYS = 91       # 3M forward used for carry
VOL_TENOR = "3M"            # ATM vol / RR / BF tenor
TS_Z_WINDOW = 756           # 3y rolling window for time-series z
TS_Z_MIN_OBS = 126          # minimum obs before a TS z is emitted (~6m)
XS_TS_BLEND = 0.50          # 0 = pure time-series z, 1 = pure cross-sectional z
WINSOR_SIGMA = 3.0
FAIR_VALUE_WINDOW = 504     # 2y rolling window for STEER-lite regression
BEER_WINDOW = 756           # 3y rolling window for BEER-lite regression

# Equal weight across the seven pillars. Deliberately naive for v1 —
# do not tune these until the IC study in backtest.py has run out of sample.
PILLAR_WEIGHTS: dict[str, float] = {
    "trend": 1 / 7,
    "value": 1 / 7,
    "carry": 1 / 7,
    "risk_adj_carry": 1 / 7,
    "vol": 1 / 7,
    "positioning": 1 / 7,
    "fundamental": 1 / 7,
}

# Cross-sectional z is computed within these peer groups, not across the whole
# universe. A 6% BRL carry and a 2% NOK carry are not comparable observations.
BLOCS = ["G10", "LATAM", "CEEMEA", "APAC"]



# --------------------------------------------------------------------------
# CDS
# --------------------------------------------------------------------------
# 5y USD senior sovereign CDS. Roots confirmed on the terminal. Most follow the
# {COUNTRY}1U5 convention; Czech and Taiwan are contract-level identifiers that
# do not, which is why every root is listed explicitly rather than derived.
#
# CDS_SUFFIX is the source/yellow-key tail. It is NOT yet confirmed — run
# `python probe_tickers.py --role cds_suffix` to settle it against the live
# terminal before trusting the carry-adjusted-risk pillar.
CDS_SUFFIX = "CBIL Curncy"


def CDS(root: str) -> str:
    return f"{root} {CDS_SUFFIX}"


# --------------------------------------------------------------------------
# Pair definition
# --------------------------------------------------------------------------

@dataclass
class Pair:
    ccy: str                     # the non-USD currency
    bloc: str
    ticker: str                  # BBG spot ticker, market convention
    usd_base: bool               # True if quoted USDXXX
    deliverable: bool = True     # False => NDF
    pip_factor: float = 10000.0  # forward points divisor
    vol_root: Optional[str] = None      # root for V/25R/25B tickers
    swap2y: Optional[str] = None        # 2y IRS/OIS ticker
    cpi_yoy: Optional[str] = None       # headline CPI y/y, for real rates
    cds5y: Optional[str] = None         # 5y USD sovereign CDS
    reer: Optional[str] = None          # broad real effective exchange rate
    tot: Optional[str] = None           # terms-of-trade proxy index
    cot_long: Optional[str] = None       # CFTC non-commercial gross longs
    cot_short: Optional[str] = None      # CFTC non-commercial gross shorts
    cesi: Optional[str] = None          # economic surprise index
    ca_gdp: Optional[str] = None        # current account % GDP

    @property
    def sign(self) -> int:
        """+1 if a rising ticker means a STRONGER non-USD currency."""
        return -1 if self.usd_base else 1

    @property
    def name(self) -> str:
        return self.ticker.split()[0]


def _p(ccy, bloc, ticker, usd_base, **kw) -> Pair:
    root = ticker.split()[0]
    kw.setdefault("vol_root", root)
    return Pair(ccy=ccy, bloc=bloc, ticker=ticker, usd_base=usd_base, **kw)


# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------
# USD leg tickers used by several pillars
USD_SWAP2Y = "USOSFR2 BGN Curncy"
USD_CPI = "CPI YOY Index"
USD_CESI = "CESIUSD Index"

G10 = [
    _p("EUR", "G10", "EURUSD Curncy", False, swap2y="EUSA2 Curncy",
       cpi_yoy="ECCPEMUY Index", cesi="CESIEUR Index", reer="CTTWBREU Index",
       cot_long="IMMBENCL Index", cot_short="IMMBENCS Index", ca_gdp="EHCAEU Index"),
    _p("JPY", "G10", "USDJPY Curncy", True, pip_factor=100.0,
       swap2y="JYSO2 BGN Curncy", cpi_yoy="JNCPIYOY Index", cesi="CESIJPY Index",
       reer="CTTWBRJP Index", cot_long="IMM5JNCL Index", cot_short="IMM5JNCS Index",
       ca_gdp="EHCAJP Index"),
    _p("GBP", "G10", "GBPUSD Curncy", False, swap2y="BPSWS2 Curncy",
       cpi_yoy="UKRPCJYR Index", cesi="CESIGBP Index", reer="CTTWBRGB Index",
       cot_long="IMM5PNCL Index", cot_short="IMM5PNCS Index", ca_gdp="EHCAGB Index"),
    _p("CHF", "G10", "USDCHF Curncy", True, swap2y="SFSNT2 BGN Curncy",
       cpi_yoy="SZCPIYOY Index", cesi="CESICHF Index", reer="CTTWBRCH Index",
       cot_long="IMM4SNCL Index", cot_short="IMM4SNCS Index", ca_gdp="EHCACH Index"),
    _p("CAD", "G10", "USDCAD Curncy", True, swap2y="CDSO2 BGN Curncy",
       cpi_yoy="CACPIYOY Index", cesi="CESICAD Index", reer="CTTWBRCA Index",
       cot_long="IMM3CNCL Index", cot_short="IMM3CNCS Index", tot="CL1 Comdty", ca_gdp="EHCACA Index"),
    _p("AUD", "G10", "AUDUSD Curncy", False, swap2y="ADSW2 Curncy",
       cpi_yoy=("ACPMYOY Index", "AUCPIYOY Index"), cesi="CESIAUD Index", reer="CTTWBRAU Index",
       cot_long="IMM6ANCL Index", cot_short="IMM6ANCS Index", tot="BCOM Index", ca_gdp="EHCAAU Index"),
    _p("NZD", "G10", "NZDUSD Curncy", False, swap2y="NDSW2 Curncy",
       cpi_yoy="NZCPIYOY Index", cesi="CESINZD Index", reer="CTTWBRNZ Index",
       cot_long="IMM6ZNCL Index", cot_short="IMM6ZNCS Index", ca_gdp="EHCANZ Index"),
    _p("NOK", "G10", "USDNOK Curncy", True, swap2y="NKSW2 Curncy",
       cpi_yoy="NOCPIYOY Index", cesi="CESISEK Index", reer="CTTWBRNO Index",
       tot="CO1 Comdty", ca_gdp="EHCANO Index"),
    _p("SEK", "G10", "USDSEK Curncy", True, swap2y="SKSW2 Curncy",
       cpi_yoy="SWCPYOY Index", cesi="CESISEK Index", reer="CTTWBRSE Index",
       ca_gdp="EHCASE Index"),
]

LATAM = [
    _p("BRL", "LATAM", "USDBRL Curncy", True, deliverable=False,
       swap2y="BCSFPPDV BLC Curncy", cpi_yoy="BZPIIPCY Index",
       cds5y=CDS("CBRZ1U5"), reer="CTTWBRBR Index", tot="BCOM Index",
       cot_long="IMM1VNCL Index", cot_short="IMM1VNCS Index", cesi="CESIEM Index", ca_gdp="EHCABR Index"),
    _p("MXN", "LATAM", "USDMXN Curncy", True, swap2y="MPSWF2B BGN Curncy",
       cpi_yoy="MXCPYOY Index", cds5y=CDS("CMEX1U5"),
       reer="CTTWBRMX Index", tot="CL1 Comdty", cot_long="IMM6MNCL Index", cot_short="IMM6MNCS Index",
       cesi="CESIEM Index", ca_gdp="EHCAMX Index"),
    _p("CLP", "LATAM", "USDCLP Curncy", True, deliverable=False,
       pip_factor=1.0, swap2y="CHSWP2 Curncy", cpi_yoy="CNPINSYO Index",
       cds5y=CDS("CCHIL1U5"), reer="CTTWBRCL Index", tot="LMCADS03 Comdty",
       cesi="CESIEM Index", ca_gdp="EHCACL Index"),
    _p("COP", "LATAM", "USDCOP Curncy", True, deliverable=False,
       pip_factor=1.0, swap2y="CLSWIB2 Curncy", cpi_yoy="COCPIYOY Index",
       cds5y=CDS("CCOL1U5"), reer="CTTWBRCO Index", tot="CO1 Comdty",
       cesi="CESIEM Index", ca_gdp="EHCACO Index"),
    _p("PEN", "LATAM", "USDPEN Curncy", True, deliverable=False,
       swap2y="PENSSS2 BGN Curncy", cpi_yoy="PRCPYOY Index",
       cds5y=CDS("CPERU1U5"), reer="CTTWBRPE Index", tot="LMCADS03 Comdty",
       cesi="CESIEM Index", ca_gdp="EHCAPE Index"),
]

CEEMEA = [
    _p("PLN", "CEEMEA", "USDPLN Curncy", True, swap2y="PZSW2 Curncy",
       cpi_yoy="POCPIYOY Index", cds5y="POLAND CDS USD SR 5Y D14 Corp",
       reer="CTTWBRPL Index", cesi="CESIEM Index", ca_gdp="EHCAPL Index"),
    _p("HUF", "CEEMEA", "USDHUF Curncy", True, pip_factor=100.0,
       swap2y="HFSW2 Curncy", cpi_yoy="HUCPIYY Index",
       cds5y="HUNGARY CDS USD SR 5Y D14 Corp", reer="CTTWBRHU Index", cesi="CESIEM Index",
       ca_gdp="EHCAHU Index"),
    _p("CZK", "CEEMEA", "USDCZK Curncy", True, swap2y="CKSW2 Curncy",
       cpi_yoy="CZCPYOY Index", cds5y="CZECH REP CDS USD SR 5Y D14 Corp",
       reer="CTTWBRCZ Index", cesi="CESIEM Index", ca_gdp="EHCACZ Index"),
    _p("ZAR", "CEEMEA", "USDZAR Curncy", True, swap2y="SASW2 Curncy",
       cpi_yoy="SACPIYOY Index", cds5y=CDS("CSOAF1U5"),
       reer="CTTWBRZA Index", tot="XAU Curncy", cot_long="CFF4ANCL Index", cot_short="CFF4ANCS Index",
       cesi="CESIEM Index", ca_gdp="EHCAZA Index"),
    _p("TRY", "CEEMEA", "USDTRY Curncy", True, swap2y="TYSO2 BGN Curncy",
       cpi_yoy="TUCPIY Index", cds5y=CDS("CTURK1U5"),
       reer="CTTWBRTR Index", cesi="CESIEM Index", ca_gdp="EHCATR Index"),
    _p("ILS", "CEEMEA", "USDILS Curncy", True, swap2y="ILSW2 Curncy",
       cpi_yoy="ISCPIY Index", cds5y=CDS("CISR1U5"),
       reer="CTTWBRIL Index", cesi="CESIEM Index", ca_gdp="EHCAIL Index"),
]

APAC = [
    _p("KRW", "APAC", "USDKRW Curncy", True, deliverable=False, pip_factor=1.0,
       swap2y="KWSWNI2 Curncy", cpi_yoy="KOCPIYOY Index",
       cds5y=CDS("CKREA1U5"), reer="CTTWBRKR Index", cesi="CESIEM Index",
       ca_gdp="EHCAKR Index"),
    _p("CNH", "APAC", "USDCNH Curncy", True, swap2y="CCSWO2 Curncy",
       cpi_yoy="CNCPIYOY Index", cds5y=CDS("CCHIN1U5"),
       reer="CTTWBRCN Index", cesi="CESICNY Index", ca_gdp="EHCACN Index"),
    _p("TWD", "APAC", "USDTWD Curncy", True, deliverable=False,
       swap2y="TDSWO2 BGN Curncy", cpi_yoy="TWCPIYOY Index",  # no liquid USD sovereign CDS
       reer="CTTWBRTW Index", cesi="CESIEM Index", ca_gdp="EHCATW Index"),
    _p("INR", "APAC", "USDINR Curncy", True, deliverable=False,
       swap2y="IRSWNI2 Curncy", cpi_yoy="INFUTOTY Index",
       cds5y="INDIA CDS USD SR 5Y D14 Corp", reer="CTTWBRIN Index", cesi="CESIEM Index",
       ca_gdp="EHCAIN Index"),
    _p("IDR", "APAC", "USDIDR Curncy", True, deliverable=False, pip_factor=1.0,
       swap2y="IHSWOO2 BGN Curncy", cpi_yoy="IDCPIY Index",
       cds5y=CDS("CINO1U5"), reer="CTTWBRID Index", cesi="CESIEM Index",
       ca_gdp="EHCAID Index"),
    _p("THB", "APAC", "USDTHB Curncy", True, swap2y="TBSWI2 BGN Curncy",
       cpi_yoy="THCPIYOY Index", cds5y=CDS("CTHAI1U5"),
       reer="CTTWBRTH Index", cesi="CESIEM Index", ca_gdp="EHCATH Index"),
]

UNIVERSE: list[Pair] = G10 + LATAM + CEEMEA + APAC
BY_CCY: dict[str, Pair] = {p.ccy: p for p in UNIVERSE}


# --------------------------------------------------------------------------
# Crosses — scored as the difference of two USD-leg scores
# --------------------------------------------------------------------------
# A cross score is score(base) - score(term). This keeps the engine to one
# scoring pass and guarantees internal consistency between outrights and RV.

CROSSES: list[tuple[str, str]] = [
    ("EUR", "GBP"), ("EUR", "CHF"), ("EUR", "SEK"), ("EUR", "NOK"),
    ("EUR", "JPY"), ("GBP", "JPY"), ("AUD", "JPY"), ("AUD", "NZD"),
    ("CAD", "JPY"), ("NOK", "SEK"), ("CHF", "JPY"), ("EUR", "AUD"),
    # EM relative value, within bloc only
    ("BRL", "MXN"), ("BRL", "CLP"), ("MXN", "CLP"), ("CLP", "PEN"),
    ("COP", "MXN"), ("PLN", "HUF"), ("PLN", "CZK"), ("HUF", "CZK"),
    ("ZAR", "TRY"), ("KRW", "TWD"), ("INR", "IDR"), ("THB", "KRW"),
]


# --------------------------------------------------------------------------
# Market regime inputs (reported alongside the ranks, not used as a weight)
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Forward conventions
# --------------------------------------------------------------------------
# Explicit per-currency forward handling. FWD = SPOT + POINTS/DIVISOR, always.
# No auto-detection — the convention is declared here and applied literally.
#
#   mode "points"   : ticker quotes swap points; divide by `divisor`
#   mode "outright" : ticker quotes the NDF outright; used as-is
#
# The divisor is the single most dangerous number in this file. Getting it
# wrong by a factor of ten does not raise — it produces carry that is merely
# wrong, and a wrong carry RANKS. sanity_check_carry() in pillars.py therefore
# recomputes implied annualised carry on every run and refuses to proceed if
# any currency falls outside a plausible band.

FWD_CONVENTIONS: dict[str, tuple[str, float, str]] = {
    # ccy:  (ticker,               divisor,  mode)
    "EUR":  ("EUR3M Curncy",        10000.0, "points"),
    "JPY":  ("JPY3M Curncy",          100.0, "points"),
    "GBP":  ("GBP3M Curncy",        10000.0, "points"),
    "CHF":  ("CHF3M Curncy",        10000.0, "points"),
    "CAD":  ("CAD3M Curncy",        10000.0, "points"),
    "AUD":  ("AUD3M Curncy",        10000.0, "points"),
    "NZD":  ("NZD3M Curncy",        10000.0, "points"),
    "NOK":  ("NOK3M Curncy",        10000.0, "points"),
    "SEK":  ("SEK3M Curncy",        10000.0, "points"),

    "BRL":  ("BCN3M Curncy",        10000.0, "points"),
    "MXN":  ("MXN3M Curncy",        10000.0, "points"),
    "CLP":  ("CHN3M Curncy",            1.0, "points"),   # confirmed: divisor 1
    "COP":  ("COP3M Curncy",            1.0, "points"),
    "PEN":  ("PSN+3M BGN Curncy",       1.0, "outright"),

    "PLN":  ("PLN3M Curncy",        10000.0, "points"),
    "HUF":  ("HUF3M Curncy",          100.0, "points"),
    "CZK":  ("CZK3M Curncy",         1000.0, "points"),   # confirmed: divisor 1000
    # ZAR: supplied as 1,000, which implies +29% annualised carry. Corrected to
    # 10,000 -> +2.9%, consistent with the ZAR-USD policy differential.
    "ZAR":  ("ZAR3M Curncy",        10000.0, "points"),
    # TRY: supplied as 1,000, which implies +305% annualised carry. Corrected
    # to 10,000 -> +30.5%, consistent with the TRY-USD differential.
    "TRY":  ("TRY3M Curncy",        10000.0, "points"),
    "ILS":  ("ILS3M Curncy",        10000.0, "points"),

    "KRW":  ("KRW3M Curncy",            1.0, "points"),
    "CNH":  ("CNH3M Curncy",        10000.0, "points"),
    "TWD":  ("NTN3M Curncy",            1.0, "points"),
    # THB: supplied as 1,000, which implies -0.17% annualised — too small
    # against a roughly -2.5% differential. Corrected to 100 -> about -1.7%.
    "THB":  ("THB3M Curncy",          100.0, "points"),
    "INR":  ("IRN+3M BGN Curncy",       1.0, "outright"),
    "IDR":  ("IHN+3M BGN Curncy",       1.0, "outright"),
}

# Plausible band for |annualised 3m carry|, %. Anything outside this is a
# convention error, not a market observation.
CARRY_SANITY_MAX = 60.0
CARRY_SANITY_MIN = 0.01

REGIME_TICKERS = {
    "vix": "VIX Index",
    "move": "MOVE Index",
    "hy_oas": "LF98OAS Index",
    "dxy": "DXY Curncy",
    "oil": "CO1 Comdty",
    "copper": "LMCADS03 Comdty",
    "gold": "XAU Curncy",
    "bcom": "BCOM Index",
}

# Risk proxies used in the STEER-lite fair value regression
STEER_RISK_PROXY = "hy_oas"
