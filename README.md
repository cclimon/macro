# macro

FX and rates analytics — four tools sharing a common output directory.

## Projects

| Folder | Description | Data source |
|---|---|---|
| `fx-vol-monitor/` | G10 FX volatility & carry monitor — HAR-RV forecasts, squeeze flags, heatmap | Bloomberg / synthetic fallback |
| `fx-vol-bands/` | Implied vol bands for FX pairs — Bollinger width regime filter, gamma squeeze detection | Bloomberg / CSV / sample |
| `rv-analysis/` | Realized vs implied vol analysis — Yang-Zhang & EWMA estimators, IV/RV spreads, calibration | Bloomberg |
| `stir-engine/` | Multi-factor SOFR/STIR engine — OU mean reversion, momentum, HMM regimes, walk-forward backtest | yfinance / FRED / CFTC |

## Output

All generated files (charts, CSVs, logs) write to `output/<project>/` which is gitignored.

```
output/
├── fx-vol-monitor/    # per-pair PNG monitors, summary heatmap, CSV
├── fx-vol-bands/      # vol band charts and data CSVs
├── rv-analysis/       # RV/IV dashboard PNGs, calibration charts
└── stir-engine/       # engine logs
```

## Setup

Each project has its own `requirements.txt`. Install per project:

```bash
cd <project>
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Quick start

```bash
# FX vol & carry monitor
python fx-vol-monitor/fx_vol_carry_monitor.py

# FX vol bands (Bloomberg)
python fx-vol-bands/fx_vol_bands.py --blp --ticker EURUSD

# RV vs IV analysis
python -m rv-analysis.vol_analysis.main --pair EURUSD

# STIR engine
python stir-engine/main.py
```
