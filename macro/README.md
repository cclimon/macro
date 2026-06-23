# G10 FX Signal Dashboard

EOD signal scanner for G10 FX — Technical, Carry, and Macro pillars.
Data source: Bloomberg terminal via `blpapi`.

## Setup

```bash
# 1. Clone into your local GitRepo folder
git clone https://github.com/cclimon/macro

# 2. Install dependencies (Bloomberg terminal must be open)
pip install -r requirements.txt

# Note: blpapi requires a Bloomberg terminal licence.
# Download the Python blpapi SDK from the Bloomberg Developer Portal.

# 3. Run the EOD signal builder
python main.py

# 4. Launch the Streamlit dashboard
streamlit run dashboard/app.py
```

## Project structure

```
macro/
├── config/
│   └── pairs.py          # G10 universe, all Bloomberg tickers
├── data/
│   └── bloomberg.py      # BDP / BDH fetch layer
├── signals/
│   ├── technical.py      # RSI, MACD, SMA, ROC, BB, ADX, Z-score
│   ├── carry.py          # Rate diff, carry/vol, forward carry, real carry
│   └── macro.py          # PMI, CPI, CB bias, policy rate diff
├── dashboard/
│   └── app.py            # Streamlit dashboard
├── main.py               # EOD orchestrator
└── requirements.txt
```

## Signals

### Technical (Momentum)
- RSI(14) with zone classification
- MACD 12/26/9 — histogram direction
- SMA cross 20/50 and 50/200
- Rate of Change — 1m and 3m
- Bollinger Band %B
- ADX(14) trend strength
- 1-year Z-score

### Carry
- 3m nominal rate differential (bps)
- Real carry = nominal − CPI differential
- Forward-implied annualised carry
- Realised vol 1m and 3m (annualised)
- Carry/Vol ratio 1m and 3m
- Vol regime (Low / Normal / High vs 6m avg)

### Macro
- PMI composite differential (Mfg + Services)
- PMI MoM momentum differential
- CPI YoY differential
- CPI MoM momentum differential
- CB bias score (Hawkish / Neutral / Dovish)
- Policy rate differential (bps)

## Next steps
- [ ] Add scoring / ranking layer across pillars
- [ ] Add weight optimisation with learning phase
- [ ] Add EM pairs
- [ ] Add CFTC CoT positioning data
- [ ] Add GDP surprise scores
