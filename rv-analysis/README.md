# RV Analysis

Realized vs Implied Volatility analysis for FX pairs using Bloomberg data. Computes Yang-Zhang and EWMA realized volatility estimators, compares against implied vol across five tenors, and generates IV/RV spread dashboards.

## Tools

| Script | Description |
|---|---|
| `vol_analysis/main.py` | Main RV vs IV dashboard — per-tenor PNG charts |
| `calibrate.py` | Calibrates YZ and EWMA estimators against 30-min HF realized vol |
| `run_usdinr_rsi.py` | RSI mean-reversion backtest (configurable pair) |
| `USDINR_RSI_Backtest.ipynb` | Jupyter notebook version of the RSI backtest |

## Usage

```bash
# RV vs IV dashboard (run from repo root)
python -m rv-analysis.vol_analysis.main --pair EURUSD
python -m rv-analysis.vol_analysis.main --pair EURUSD --start 2023-01-01 --end 2025-12-31

# Pairs: EURUSD, GBPUSD, USDJPY, USDCNH, USDMXN, CADJPY, EURGBP, EURMXN, GBPAUD

# Calibration
python rv-analysis/calibrate.py --pair EURUSD

# RSI backtest
python rv-analysis/run_usdinr_rsi.py --pair EURUSD
python rv-analysis/run_usdinr_rsi.py --pair USDJPY --sell-thresh 70 --cover-thresh 50
```

## Output

PNG dashboards → `output/rv-analysis/` (gitignored).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# Bloomberg terminal must be running for live data
```
