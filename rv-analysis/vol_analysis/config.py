"""
config.py — TENOR_MAP, plot style constants, and signal thresholds.
"""

TENOR_MAP = {
    "1W":  {"iv_ticker": "{PAIR}V1W Curncy",  "window": 5  },
    "2W":  {"iv_ticker": "{PAIR}V2W Curncy",  "window": 10 },
    "1M":  {"iv_ticker": "{PAIR}V1M Curncy",  "window": 21 },
    "3M":  {"iv_ticker": "{PAIR}V3M Curncy",  "window": 63 },
    "6M":  {"iv_ticker": "{PAIR}V6M Curncy",  "window": 126},
}

# Rolling lookback windows for percentile rank
PERCENTILE_WINDOWS = {
    "1Y": 252,
    "2Y": 504,
}

# Signal thresholds (based on 2Y percentile)
SIGNAL_RICH_THRESHOLD    = 75   # percentile > 75 → IV RICH
SIGNAL_CHEAP_THRESHOLD   = 25   # percentile < 25 → IV CHEAP

# RiskMetrics EWMA decay factor
EWMA_LAMBDA = 0.94

# Annualisation factor
ANNUALISATION_FACTOR = 252

# Forward-fill limit for missing data (days)
FFILL_LIMIT = 2

# ── Plot style ────────────────────────────────────────────────────────────────

PLOT_STYLE = {
    "figure_facecolor":  "white",
    "axes_facecolor":    "white",
    "grid_color":        "#cccccc",
    "grid_alpha":        0.3,
    "spine_color":       "#cccccc",
    "font_family":       "DejaVu Sans",
    "label_fontsize":    10,
    "title_fontsize":    12,
    "dpi":               150,
    "height_ratios":     [3, 2, 2, 1.5],
}

PANEL_COLORS = {
    "yz":      "blue",
    "ewma":    "orange",
    "iv":      "green",
    "zero":    "black",
    "fill_pos": "green",
    "fill_neg": "red",
    "ref_line": "grey",
}

SIGNAL_COLORS = {
    "IV RICH":  "darkgreen",
    "IV CHEAP": "darkred",
    "NEUTRAL":  "grey",
}
