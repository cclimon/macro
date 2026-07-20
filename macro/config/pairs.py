# config/pairs.py
# G10 FX Universe & Bloomberg Ticker Mapping

G10_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
    "AUDUSD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY",
    "AUDJPY", "CADJPY", "NZDCHF", "EURCHF", "AUDNZD",
    "EURNOK", "USDNOK", "EURSEK", "USDSEK",
]

EM_PAIRS = [
    "USDBRL", "USDMXN", "USDCLP",
    "EURHUF", "USDPLN", "EURCZK",
    "USDZAR", "USDKRW", "USDIDR",
]

ALL_PAIRS = G10_PAIRS + EM_PAIRS

# Spot FX tickers — Bloomberg Curncy
SPOT_TICKERS = {pair: f"{pair} Curncy" for pair in ALL_PAIRS}

# 3m FX Forward tickers
FORWARD_TICKERS = {pair: f"{pair}3M Curncy" for pair in ALL_PAIRS}

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
    # EM
    "BRL": "BZDIOVRA Index",    # Brazil CDI overnight (best proxy)
    "MXN": "MXIBR91D Index",    # Mexico TIIE 91-day
    "CLP": "CLCAM90D Index",    # Chile 90-day deposit rate
    "HUF": "BUBOR3M BGN Curncy", # Hungary BUBOR 3m
    "PLN": "WIBOR3M BGN Curncy", # Poland WIBOR 3m
    "CZK": "PRIBOR3M BGN Curncy", # Czech PRIBOR 3m
    "ZAR": "JIBAS3M BGN Curncy", # South Africa JIBAR 3m
    "KRW": "KOCD91D Index",     # Korea CD 91-day
    "IDR": "IDIBOR3M BGN Curncy", # Indonesia JIBOR 3m
}

# CPI YoY tickers per currency
CPI_TICKERS = {
    "USD": "CPI YOY Index",
    "EUR": "ECCPEMUY Index",
    "GBP": "UKRPCJYR Index",
    "JPY": "JNCPIYOY Index",
    "CHF": "SZCPIYOY Index",
    "CAD": "CACPIYOY Index",
    "AUD": "AUCPIYOY Index",
    "NZD": "NZCPIYOY Index",
    "NOK": "NOCPIYOY Index",
    "SEK": "SWCPIYOY Index",
    # EM
    "BRL": "BZPIIPCA Index",    # Brazil IPCA CPI YoY
    "MXN": "MXCPIYOY Index",    # Mexico CPI YoY
    "CLP": "CLCPIYOY Index",    # Chile CPI YoY
    "HUF": "HUCPIYOY Index",    # Hungary CPI YoY
    "PLN": "POCPIYOY Index",    # Poland CPI YoY
    "CZK": "CZCPIYOY Index",    # Czech CPI YoY
    "ZAR": "SACPIYOY Index",    # South Africa CPI YoY
    "KRW": "KOCPIYOY Index",    # Korea CPI YoY
    "IDR": "IDCPIYOY Index",    # Indonesia CPI YoY
}

# Composite PMI tickers per currency (S&P Global Composite; manufacturing where composite unavailable)
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
    # EM — composite where available, manufacturing otherwise; None = no coverage
    "BRL": "MPMIBZCA Index",    # S&P Global Brazil Composite
    "MXN": "MXPMIMAN Index",    # S&P Global Mexico Manufacturing (no composite)
    "CLP": None,                # No S&P Global coverage
    "HUF": None,                # No S&P Global coverage
    "PLN": "MPMIPLMF Index",    # S&P Global Poland Manufacturing
    "CZK": "MPMICZMA Index",    # S&P Global Czech Manufacturing
    "ZAR": "MPMIZAMA Index",    # S&P Global South Africa
    "KRW": "MPMIKRCA Index",    # S&P Global Korea Composite
    "IDR": "MPMIIDCA Index",    # S&P Global Indonesia
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
    "NOK": "NOBRDEPA Index",
    "SEK": "SWRRATMN Index",
    # EM
    "BRL": "BZSTSETA Index",    # Brazil SELIC target
    "MXN": "MXONBR Index",      # Mexico Banxico overnight
    "CLP": "CHPBREPR Index",    # Chile BCCh repo rate
    "HUF": "HUGBBASE Index",    # Hungary MNB base rate
    "PLN": "POREFINR Index",    # Poland NBP reference rate
    "CZK": "CZTXRBOR Index",    # Czech CNB 2W repo
    "ZAR": "SAREPOPM Index",    # South Africa SARB repo
    "KRW": "KOBASE Index",      # Korea BoK base rate
    "IDR": "IDBIRATE Index",    # Indonesia BI 7-day reverse repo
}

# Lookback windows
LOOKBACK = {
    "1m": 21,    # ~21 trading days
    "3m": 63,    # ~63 trading days
    "1y": 252,   # ~252 trading days (for vol normalisation)
}

# Historical data pull (trading days)
HIST_DAYS = 400  # enough for 1y + buffer
