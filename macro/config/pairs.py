# config/pairs.py
# G10 FX Universe & Bloomberg Ticker Mapping

G10_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
    "AUDUSD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY",
    "AUDJPY", "CADJPY", "NZDCHF", "EURCHF", "AUDNZD",
    "EURNOK", "USDNOK", "EURSEK", "USDSEK",
]

# Spot FX tickers — Bloomberg Curncy
SPOT_TICKERS = {pair: f"{pair} Curncy" for pair in G10_PAIRS}

# 3m FX Forward tickers
FORWARD_TICKERS = {pair: f"{pair}3M Curncy" for pair in G10_PAIRS}

# G10 currencies involved
G10_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "NOK", "SEK"]

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
    "NOK": "NOWA Index",        # NOK NOWA overnight
    "SEK": "SWESTR Index",      # SEK SWESTR overnight
}

# 3m OIS / IBOR rates per currency
RATE_3M_TICKERS = {
    "USD": "USOSFRC Curncy",    # USD 3m SOFR compounded
    "EUR": "EURI3M BGN Curncy", # EUR 3m Euribor
    "GBP": "GBPI3M BGN Curncy", # GBP 3m SONIA
    "JPY": "JPYI3M BGN Curncy", # JPY 3m TONAR
    "CHF": "CHFI3M BGN Curncy", # CHF 3m SARON
    "CAD": "CADI3M BGN Curncy", # CAD 3m CORRA
    "AUD": "AUDI3M BGN Curncy", # AUD 3m AONIA
    "NZD": "NZDI3M BGN Curncy", # NZD 3m OCR
    "NOK": "NOKI3M BGN Curncy", # NOK 3m NOWA
    "SEK": "SEKI3M BGN Curncy", # SEK 3m SWESTR
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
    "NOK": "NOCPIYOY Index",    # Norway CPI YoY
    "SEK": "SWCPIYOY Index",    # Sweden CPI YoY
}

# Composite PMI tickers per currency (S&P Global Composite)
PMI_TICKERS = {
    "USD": "MPMIUSCA Index",
    "EUR": "MPMIEZCA Index",
    "GBP": "MPMIGBCA Index",
    "JPY": "MPMIJPCA Index",
    "CHF": "SVPMICOM Index",
    "CAD": "IVEYPMIS Index",
    "AUD": "MPMIAUCA Index",
    "NZD": "BNNZPMI Index",
    "NOK": "NOPMISA Index",
    "SEK": "PMISCMPS Index",
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
    "NOK": "NOBRDEPA Index",     # Norges Bank sight deposit rate
    "SEK": "SWRRATMN Index",    # Riksbank repo rate
}

# Lookback windows
LOOKBACK = {
    "1m": 21,    # ~21 trading days
    "3m": 63,    # ~63 trading days
    "1y": 252,   # ~252 trading days (for vol normalisation)
}

# Historical data pull (trading days)
HIST_DAYS = 400  # enough for 1y + buffer
