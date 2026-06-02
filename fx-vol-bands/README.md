# FX Vol Bands

Computes 1-week implied volatility bands for FX pairs. Adds Bollinger Band width as a regime filter and flags potential gamma squeeze setups.

## Usage

```bash
# Bloomberg (live data)
python fx_vol_bands.py --blp --ticker EURUSD

# From a CSV file (columns: date, spot, implied_vol_1w, risk_reversal_25d)
python fx_vol_bands.py --csv mydata.csv

# Offline sample data — no Bloomberg needed
python fx_vol_bands.py --sample --pair EURUSD
```

## Key options

| Flag | Default | Description |
|---|---|---|
| `--blp` | — | Use Bloomberg terminal as data source |
| `--csv` | — | Load data from a CSV file |
| `--sample` | — | Generate synthetic data for testing |
| `--ticker` | — | Bloomberg ticker (e.g. `EURUSD`) |
| `--pair` | — | Pair label when using `--sample` |
| `--out-dir` | `output/fx-vol-bands/` | Output directory |

## Output

Charts and data CSVs → `output/fx-vol-bands/` (gitignored).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
