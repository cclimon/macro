"""
config.py
---------
Central configuration for the STIR engine.
All API keys, file paths, and constants live here.
Never commit real API keys — use environment variables in production.
"""

from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

ROOT_DIR   = Path(__file__).resolve().parent
DATA_DIR   = ROOT_DIR / "data" / "cache"       # local parquet cache
LOG_DIR    = ROOT_DIR.parent / "output" / "stir-engine"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

# FRED (https://fred.stlouisfed.org/docs/api/api_key.html — free registration)
FRED_API_KEY = os.getenv("FRED_API_KEY", "YOUR_FRED_API_KEY_HERE")
print(f"Using FRED API key: {FRED_API_KEY}")

# CME DataMine (paid) — leave blank for prototype; yfinance is used as fallback
# CME_USERNAME = os.getenv("CME_USERNAME", "")
# CME_PASSWORD = os.getenv("CME_PASSWORD", "")

# ---------------------------------------------------------------------------
# SOFR futures universe
# ---------------------------------------------------------------------------

# CME Globex codes for 3-month SOFR futures quarterly expiries.
# Format: SR3 + month code + 2-digit year
# Month codes: H=Mar, M=Jun, U=Sep, Z=Dec
SOFR_QUARTERLY_CODES = [
    "SR3H25", "SR3M25", "SR3U25", "SR3Z25",
    "SR3H26", "SR3M26", "SR3U26", "SR3Z26",
]

# Front contract ticker via yfinance (continuous proxy for prototyping)
# Note: yfinance coverage for SOFR futures is limited; ZQ=FF futures used
# as a well-established fallback with near-identical signal properties.
SOFR_FRONT_YFTICKER  = "SR3=F"       # 3-month SOFR front month
FF_FRONT_YFTICKER    = "ZQ=F"        # Fed Funds front month (fallback)

# Futures price convention: price = 100 - implied_rate
# e.g. price 95.50 -> implied rate 4.50%
FUTURES_PRICE_OFFSET = 100.0

# ---------------------------------------------------------------------------
# FRED macro series
# ---------------------------------------------------------------------------

FRED_SERIES = {
    # Policy rate
    "fed_funds_target_upper": "DFEDTARU",     # Upper bound of FF target range
    "fed_funds_target_lower": "DFEDTARL",     # Lower bound
    "fed_funds_effective":    "DFF",          # Effective FF rate (daily)
    "sofr_fixing":            "SOFR",         # SOFR daily fixing

    # Inflation
    "cpi_yoy":                "CPIAUCSL",     # CPI all items, SA (monthly)
    "core_cpi_yoy":           "CPILFESL",     # Core CPI ex food & energy
    "pce_yoy":                "PCEPI",        # PCE price index (monthly)
    "core_pce_yoy":           "PCEPILFE",     # Core PCE — Fed's preferred

    # Labour market
    "nonfarm_payrolls":       "PAYEMS",       # NFP (monthly)
    "unemployment_rate":      "UNRATE",       # Unemployment rate (monthly)

    # Activity
    "ism_manufacturing":      "MANEMP",       # Manufacturing employment proxy
    "retail_sales":           "RSAFS",        # Retail & food services sales
}

# ---------------------------------------------------------------------------
# Data fetch settings
# ---------------------------------------------------------------------------

DEFAULT_START_DATE = "2018-01-01"    # SOFR futures launched Oct 2018; buffer for warmup
DEFAULT_END_DATE   = None            # None = today

# Frequency for resampling all series to a common daily spine
RESAMPLE_FREQ = "B"                  # Business days

# Local cache filenames
CACHE_SOFR_FUTURES = DATA_DIR / "sofr_futures.parquet"
CACHE_FRED_MACRO   = DATA_DIR / "fred_macro.parquet"
CACHE_COMBINED     = DATA_DIR / "combined_daily.parquet"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL  = "INFO"
LOG_FILE   = LOG_DIR / "stir_engine.log"

# ---------------------------------------------------------------------------
# OSINT / news signal
# ---------------------------------------------------------------------------
 
# Anthropic API key — used for LLM-based event classification
# Get yours at https://console.anthropic.com
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY_HERE")
 
# GDELT API — no key required, but rate-limit to 1 req/sec to be polite
GDELT_API_BASE    = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_RATE_LIMIT  = 1.0      # seconds between requests
 
# RSS feeds polled for Fed communications and macro news
OSINT_RSS_FEEDS = {
    "fed_speeches":    "https://www.federalreserve.gov/feeds/speeches.xml",
    "fed_press":       "https://www.federalreserve.gov/feeds/press_all.xml",
    "reuters_economy": "https://feeds.reuters.com/reuters/businessNews",
}
 
# Local cache
CACHE_OSINT_EVENTS = DATA_DIR / "osint_events.parquet"
CACHE_OSINT_SCORE  = DATA_DIR / "osint_score.parquet"

# ---------------------------------------------------------------------------
# OSINT event taxonomy
# Each class defines:
#   half_life_fast  : days for the immediate repricing component to decay by half
#   half_life_slow  : days for the lingering uncertainty premium to decay by half
#   weight_fast     : share of initial impulse in the fast component (rest = slow)
#   max_magnitude   : cap on |impact| for this class (prevents outlier domination)
#   crisis_override : if True, a high-severity event of this class triggers
#                     the circuit breaker regardless of HMM state
# ---------------------------------------------------------------------------
 
OSINT_EVENT_TAXONOMY = {
    "geopolitical_shock": {
        "half_life_fast":  3.0,
        "half_life_slow":  30.0,
        "weight_fast":     0.35,
        "max_magnitude":   3.0,
        "crisis_override": True,
    },
    "cb_communication": {
        "half_life_fast":  1.0,
        "half_life_slow":  7.0,
        "weight_fast":     0.70,
        "max_magnitude":   2.0,
        "crisis_override": False,
    },
    "scheduled_release": {
        "half_life_fast":  0.5,
        "half_life_slow":  3.0,
        "weight_fast":     0.85,
        "max_magnitude":   2.5,
        "crisis_override": False,
    },
    "financial_stress": {
        "half_life_fast":  2.0,
        "half_life_slow":  14.0,
        "weight_fast":     0.50,
        "max_magnitude":   2.5,
        "crisis_override": True,
    },
    "noise": {
        "half_life_fast":  0.25,
        "half_life_slow":  1.0,
        "weight_fast":     0.95,
        "max_magnitude":   0.5,
        "crisis_override": False,
    },
}
 
# Severity tiers used by the LLM classifier
# The LLM returns a tier; tier × base_magnitude = initial impulse
OSINT_SEVERITY_TIERS = {
    1: 0.25,   # background noise / minor
    2: 0.75,   # moderate — notable but contained
    3: 1.50,   # significant — market-moving
    4: 2.50,   # major — rare, e.g. surprise 50bp cut
    5: 3.00,   # structural shock — war, pandemic, financial crisis
}
 
# Crisis override threshold: if cumulative OSINT score exceeds this
# magnitude (absolute), force crisis weighting in the ensemble
OSINT_CRISIS_SCORE_THRESHOLD = 2.0