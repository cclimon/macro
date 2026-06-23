# config/pairs.py
# G10 FX Universe & Bloomberg Ticker Mapping

G10_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
    "AUDUSD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY",
    "AUDJPY", "CADJPY", "NZDCHF", "EURCHF", "AUDNZD",
]

# Spot FX tickers — Bloomberg Curncy
SPOT_TICKERS = {pair: f"{pair} Curncy" for pair in G10_PAIRS}

# 3m FX Forward tickers
FORWARD_TICKERS = {pair: f"{pair}3M Curncy" for pair in G10_PAIRS}

# G10 currencies involved
G10_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]

# OIS / Overnight rate tickers per currency
OIS_TICKERS = {
    "USD": "USSOC Index",       # Fed Funds OIS
    "EUR": "EUSWEO Index",      # EUR OIS 1d
    "GBP": "BPSWSC Index",      # GBP SONIA OIS
    "JPY": "JYSWOC Index",      # JPY TONAR OIS
    "CHF": "SFSNT Index",       # CHF SARON
    "CAD": "CORRA Index",       # CAD CORRA
    "AUD": "RBATCTR Index",     # AUD RBA Cash Target
    "NZD": "RBNZOCR Index",     # NZD OCR
}

# 3m money market / T-bill rates per currency
RATE_3M_TICKERS = {
    "USD": "US0003M Index",     # USD 3m LIBOR/SOFR proxy
    "EUR": "EUR003M Index",     # EUR 3m Euribor
    "GBP": "BP0003M Index",     # GBP 3m LIBOR
    "JPY": "JY0003M Index",     # JPY 3m TIBOR
    "CHF": "SF0003M Index",     # CHF 3m
    "CAD": "CD0003M Index",     # CAD 3m CDOR
    "AUD": "AU0003M Index",     # AUD 3m BBSW
    "NZD": "NZ0003M Index",     # NZD 3m
}

# CPI YoY tickers per currency
CPI_TICKERS = {
    "USD": "CPI YOY Index",
    "EUR": "ECCPUPCH Index",
    "GBP": "UKRPIYOY Index",
    "JPY": "JNCPIYOY Index",
    "CHF": "SZCPIYOY Index",
    "CAD": "CACPIYOY Index",
    "AUD": "AUCPIYOY Index",
    "NZD": "NZCPIYOY Index",
}

# Manufacturing PMI tickers per currency
PMI_MFG_TICKERS = {
    "USD": "NAPMPMI Index",
    "EUR": "MPMIEZMA Index",
    "GBP": "PMITMUK Index",
    "JPY": "JMAPMI Index",
    "CHF": "SZWVPSAI Index",
    "CAD": "IVEYSA Index",
    "AUD": "AIMIPMI Index",
    "NZD": "NZPMI Index",
}

# Services PMI tickers per currency
PMI_SVCS_TICKERS = {
    "USD": "NAPMPMS Index",
    "EUR": "MPMIEZMS Index",
    "GBP": "PMITBUK Index",
    "JPY": "JMAPSER Index",
    "CHF": None,
    "CAD": None,
    "AUD": "AISIPMI Index",
    "NZD": None,
}

# Central bank policy rate tickers
POLICY_RATE_TICKERS = {
    "USD": "FDTRMJNK Index",
    "EUR": "EURR002W Index",
    "GBP": "UKBRBASE Index",
    "JPY": "BOJDPBAL Index",
    "CHF": "SZLTTR Index",
    "CAD": "CABROVER Index",
    "AUD": "RBATCTR Index",
    "NZD": "RBNZOCR Index",
}

# Lookback windows
LOOKBACK = {
    "1m": 21,    # ~21 trading days
    "3m": 63,    # ~63 trading days
    "1y": 252,   # ~252 trading days (for vol normalisation)
}

# Historical data pull (trading days)
HIST_DAYS = 400  # enough for 1y + buffer
