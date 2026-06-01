# FX Volatility & Carry Monitor

Single-script tool that fetches one year of daily FX data from Bloomberg, computes volatility and carry metrics across 18 G10 pairs, and writes per-pair charts plus a summary heatmap to `./output/`.

---

## What it does

### Data
- Connects to Bloomberg via `blpapi` (`localhost:8194`).  
- If Bloomberg is unavailable (or `blpapi` is not installed) it falls back to GBM + mean-reverting synthetic data so the script always runs.
- Lookback: 1 calendar year from today.

### Universe
| Group | Pairs |
|---|---|
| G10 Majors | EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD EURGBP EURCHF EURJPY |
| Crosses | GBPJPY AUDJPY CADJPY GBPAUD GBPCAD EURAUD EURCAD EURNZD |

### Module 1 — Volatility monitor
- **20-day realised vol**: rolling std of log returns × √252.
- **Implied vol bands on spot**: spot × exp(±σ × √(7/365)) for σ = 1, 1.5, 2.
- **Term-structure ratio**: 1W ATM IV / 1M ATM IV; flagged inverted when > 1.0.
- **IV/RV ratio**: 1W ATM IV divided by the HAR-RV forecast (see Module 2).

### Module 2 — HAR-RV volatility forecast
Fits a Heterogeneous Autoregressive model on daily spot log returns:

```
RV_next = α + β₁·RV_d + β₂·RV_w + β₃·RV_m
```

where RV_d = daily variance, RV_w = 5-day mean, RV_m = 22-day mean.  
Coefficients estimated via OLS (numpy `lstsq`).  
The current-period annualised forecast replaces the raw 20d RV as the denominator of the IV/RV ratio.

### Module 3 — Carry / vol ratio
```
carry_diff = CCY_3M_yield − USD_3M_yield        (pairs with a USD leg)
carry_diff = leg1_3M_yield − leg2_3M_yield      (pure crosses)

carry_vol_ratio = carry_diff / 1W_ATM_IV
```

| Ratio | Label |
|---|---|
| > 1.5 | **ATTRACTIVE** |
| 0.5 – 1.5 | NEUTRAL |
| 0 – 0.5 | UNATTRACTIVE |
| < 0 | AVOID |

### Squeeze flag
Triggered when all three conditions hold simultaneously:
1. Last-day spot move > 1.2× the 1-week implied σ band
2. IV/HAR-RV ratio > 1.2
3. Term-structure ratio > 1.0 (inversion)

---

## Outputs (`./output/`)

| File | Description |
|---|---|
| `{PAIR}_monitor.png` | 4-panel chart per pair (18 files) |
| `fx_summary_heatmap.png` | All-pairs heatmap with colour-coded metrics |
| `fx_vol_carry_summary.csv` | Latest-date summary table for all pairs |

### Per-pair chart panels
1. Spot price with ±1σ / ±1.5σ / ±2σ implied vol bands; red dots mark squeeze flags.
2. Term-structure ratio (1W/1M) with inversion shading.
3. IV/HAR-RV ratio + twin-axis overlay of 20d RV vs HAR-RV forecast.
4. Carry/vol ratio with coloured ATTRACTIVE / NEUTRAL / UNATTRACTIVE / AVOID zones.

---

## Installation

```bash
pip install -r requirements.txt
```

If you have a Bloomberg terminal, install `blpapi` from the Bloomberg package index (see the comment in `requirements.txt`).

---

## Usage

```bash
python fx_vol_carry_monitor.py
```

The script auto-detects whether Bloomberg is available and falls back to synthetic data if not.

---

## Bloomberg ticker conventions used

| Data | Ticker format | Example |
|---|---|---|
| Spot | `{PAIR} BGN Curncy` | `EURUSD BGN Curncy` |
| 1W ATM IV | `{PAIR}V1W BGN Curncy` | `EURUSDV1W BGN Curncy` |
| 1M ATM IV | `{PAIR}V1M BGN Curncy` | `EURUSDV1M BGN Curncy` |
| 25d RR 1W | `{PAIR}1WRR BGN Curncy` | `EURUSD1WRR BGN Curncy` |
| USD 3M yield | `BICLUSSP Index` | — |
| CCY 3M yield | `{CCY}I3M Curncy` | `EURI3M Curncy` |

All fields: `PX_LAST`, daily frequency.
