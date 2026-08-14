# FX Screener — v1

Cross-sectional ranking of G10 and EM currencies vs USD, plus crosses, with a
suggested trade expression. Core horizon (1–3m), equal-weight pillars.

## Install

```powershell
conda activate FX_BBG
cd "C:\Users\CCM\OneDrive - Centile Partners Advisory Ltd\Documents\06_Models_&RISK\GitRepo"
# drop fx_screener\ into the repo
pip install pandas numpy pyarrow
```

## Run order

```powershell
python bbg.py --validate          # 1. ticker audit — do this first
python run_screener.py --top 10   # 2. full run
```

`--validate` writes `ticker_validation.csv`. Everything marked `PLACEHOLDER` or
`FAIL` needs a real ticker in `config.py` before that input contributes.

## Bloomberg cache

`bbg.py` maintains a persistent parquet cache under `.cache/`. Each series is
stored as a single file keyed on `(ticker, field, start)`. On every call the
cache is checked first:

- **Cache fresh** (`last stored date ≥ requested end`) — returned from disk,
  no Bloomberg call.
- **Cache stale** — only the missing tail (`last_date + 1 → today`) is fetched
  from Bloomberg and appended. A 16-year history series costs one small
  incremental request per day rather than a full re-pull.
- **Cache miss** (first run or new ticker) — full history fetched and stored.

If Bloomberg times out or errors during a delta update, the cached data is
returned rather than an empty Series, so a flaky connection during a top-up
does not blank out a pillar.

### Daily workflow

```powershell
git pull                                  # get teammates' latest cache
python run_screener.py --top 10          # BBG called only for the missing tail
git add .cache && git commit -m "cache $(Get-Date -Format yyyy-MM-dd)" && git push
```

The `.cache/` directory is tracked in git. Committing it after each run means
teammates always pull a warm cache and the screener starts in seconds.

### One-time migration (after first pull of this change)

The old cache files used a different hash (they included the `end` date in the
key) and will never be read. Delete them before the first run:

```powershell
Remove-Item ".cache\*.parquet"
```

The next run will be a cold run (~500 Bloomberg calls), after which the delta
scheme takes over.

## Architecture

| File | Role |
|---|---|
| `config.py` | universe, quote conventions, ticker map, weights |
| `bbg.py` | blpapi wrapper, parquet cache, ticker validation |
| `normalize.py` | winsorise → TS z + XS z → blend |
| `pillars.py` | the seven pillars |
| `score.py` | equal-weight composite, crosses, conviction |
| `expression.py` | IV/RV and skew → structure lookup |
| `run_screener.py` | runner and output |

## The seven pillars

1. **Trend** — 3m/6m/12m momentum, distance from 200d, MACD, all vol-normalised.
2. **Value** — STEER-lite (log spot ~ 2y rate diff + ToT + credit, 2y window)
   and BEER-lite (log REER ~ real rate diff + ToT + C/A, 5y window). Three
   regressors each, refit weekly. Residual sign inverted: rich = negative.
3. **Carry** — forward-implied, not policy differential, so NDF premium and CIP
   basis are captured. Plus curve slope.
4. **Carry-adjusted risk** — carry ÷ implied vol, ÷ downside realised vol,
   ÷ 5y CDS. This is the pillar that distinguishes carry from levered beta.
5. **Vol regime** — Δimplied, Δrealised and term slope, each signed by the
   currency's own rolling beta to credit spreads rather than by a hardcoded
   haven list. Rich/cheap vol and skew are excluded here on purpose.
6. **Positioning** — CoT level and 3m change, inverted; 25d RR skew as the
   proxy where CoT does not exist. Contrarian at extremes.
7. **Fundamental momentum** — Δ2y rate differential, CESI differential, ΔCDS,
   ToT momentum, C/A %GDP. Rate of change, not level: levels are already priced
   through carry and value.

## Design decisions worth knowing

**Orientation is handled once.** `Pair.sign` inverts USD-base pairs. Price
signals use `sign`, rate signals use `-sign`. Nothing else in the codebase
flips a sign.

**A dead ticker degrades one input, not one currency.** `combine()`
renormalises over whatever resolved, so a currency missing 2 of 5 inputs is
scored on the 3 that worked.

**Dual normalisation.** TS z asks "rich or cheap versus its own history"; XS z
asks "best available today". Blended 50/50 in `config.XS_TS_BLEND` to avoid
committing to either failure mode.

**Conviction is reported, never applied.** `agreement` and `dispersion` are
columns, not multipliers. An agreement-weighted score is a free parameter
fitted on the same history that built the pillars.

**Crosses are derived, not scored separately.** `score(base) − score(term)`, so
an RV call can never contradict the outright ranking.

## Open items before this is production

- **ToT is a placeholder** — most currencies point at a single commodity or
  BCOM. Export-weighted baskets are the correct construction; tell me the
  weighting source and I will build them.
- **CoT tickers unverified** — the CFTC Bloomberg tickers in `config.py` are
  marked `VERIFY_`. Send me one working example and I will complete the map.
- **REER unverified** — BIS broad or JPM broad, your preference.
- **No backtest yet.** IC per pillar, per bloc, per horizon, net of bid-offer
  and NDF spread, is the next module. Until it runs, treat the equal weights as
  a prior and the ranks as a starting point for judgement, not an output.
- **Realised vol is close-to-close.** Swapping in your Yang-Zhang estimator
  from `rv-analysis` is a one-function change in `MarketData`.
- **TRY, and to a degree IDR and INR, have structural breaks** that will
  distort a 5y rolling z. Consider a shorter TS window for those names.
