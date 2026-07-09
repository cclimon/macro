# FX Vol Monitor — Implied vs Yang-Zhang Realized

Heatmap of the z-score of ln(IV / RV_YZ) across G10 majors, crosses and key EM,
for tenors 1W / 2W / 1M / 3M / 6M / 1Y. Positive z (red) = implied rich vs its
own history of premium over realized; negative (blue) = implied cheap.

## Run

```bash
pip install -r requirements_vol.txt
streamlit run dashboard/vol_app.py
```

Requires a live Bloomberg session (`blpapi` + `xbbg`). Without one, the app
auto-falls back to labelled mock data so the UI can be developed anywhere.

## Modules

| File | Purpose |
|---|---|
| `config/vol_universe.py` | Universe, tenor→ticker map, RV window per tenor |
| `vol/estimators.py` | Yang-Zhang (+ GK, Parkinson, RS, close-to-close) |
| `vol/data.py` | xbbg pulls (implied history + OHLC) with mock fallback |
| `vol/signals.py` | ln(IV/RV) panels, z-scores, percentiles |
| `dashboard/vol_app.py` | Streamlit heatmap, drill-down, snapshot table |

## Methodology notes

- **Yang-Zhang** = overnight variance + k·open-close variance + (1−k)·Rogers-
  Satchell, with k = 0.34 / (1.34 + (n+1)/(n−1)). Drift-independent, handles
  gaps, ~7–14x more efficient than close-to-close.
- **RV window matched to tenor**: 1W→5d, 2W→10d, 1M→21d, 3M→63d, 6M→126d,
  1Y→252d. Annualization 252, output in vol points to match BBG quoting.
- **Signal**: z of ln(IV/RV) over a user-selected lookback (63–756d slider).
  Log-ratio is symmetric and regime-stable; the constant discretization bias of
  range estimators cancels in the z-score.
- **Percentile toggle**: ln(IV/RV) is fat-tailed; percentile rank is the
  robustness companion to the z-score.
- Implied is ffilled max 3 days to bridge holiday gaps; NaN cells mean
  insufficient history at the chosen lookback.

## Known caveats / v2 candidates

- FX "overnight" in daily bars ≈ weekend/holiday gaps (BBG NY-close bars).
- Event contamination: cells whose implied window spans CB meetings/elections
  will read "rich" without being a signal → planned ECO-calendar overlay.
- HAR-RV forecast comparison (IV vs *forecast* RV over the option life).
- 25d RR z-score skew tab, term-structure slope column.
