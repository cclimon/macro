# config/vol_universe.py
# FX Vol Dashboard — universe, tenors and Bloomberg ticker mapping
# Convention: implied ATM vol tickers are "<PAIR>V<TENOR> BGN Curncy"
#             spot OHLC from "<PAIR> Curncy" (PX_OPEN/HIGH/LOW/LAST)

# ── Universe ─────────────────────────────────────────────────────────────────
G10_MAJORS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "USDCAD", "AUDUSD", "NZDUSD",
]

CROSSES = [
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURNZD",
    "GBPJPY", "GBPAUD", "GBPCAD",
    "AUDJPY", "CADJPY", "AUDNZD",
    "NOKSEK",
]

EM_PAIRS = [
    "USDMXN", "USDZAR", "USDBRL", "USDTRY",
    "USDPLN", "USDHUF", "USDCNH", "USDKRW", "USDINR",
]

UNIVERSE = G10_MAJORS + CROSSES + EM_PAIRS

# Group labels used by the dashboard for row separators
PAIR_GROUP = (
    {p: "G10 Majors" for p in G10_MAJORS}
    | {p: "Crosses" for p in CROSSES}
    | {p: "EM" for p in EM_PAIRS}
)

# ── Tenors ───────────────────────────────────────────────────────────────────
# tenor label -> (bbg vol suffix, realized window in business days)
TENORS = {
    "1W":  ("V1W",  5),
    "2W":  ("V2W",  10),
    "1M":  ("V1M",  21),
    "3M":  ("V3M",  63),
    "6M":  ("V6M",  126),
    "1Y":  ("V1Y",  252),
}

ANNUALIZATION = 252  # trading days, matches BBG FX vol quoting


def implied_ticker(pair: str, tenor: str, source: str = "BGN") -> str:
    """Bloomberg ATM implied vol ticker, e.g. 'EURUSDV1M BGN Curncy'."""
    suffix, _ = TENORS[tenor]
    return f"{pair}{suffix} {source} Curncy"


def spot_ticker(pair: str) -> str:
    return f"{pair} Curncy"


def rv_window(tenor: str) -> int:
    return TENORS[tenor][1]
