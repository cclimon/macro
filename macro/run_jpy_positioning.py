# run_jpy_positioning.py
# Standalone: compute JPY positioning time series and print last 3 years.
# Run from GitRepo root: python macro\run_jpy_positioning.py

import pandas as pd
from data.bloomberg import BloombergSession, fetch_macro_history
from signals.positioning import (
    POSITIONING_TICKERS,
    HIST_YEARS,
    build_positioning_for_currency,
)

FETCH_DAYS = HIST_YEARS * 365  # fetch_macro_history adds 1.5x buffer internally


def main():
    print("Connecting to Bloomberg...")
    with BloombergSession() as bbg:
        print(f"Fetching JPY data ({HIST_YEARS}y history)...")
        jpy_data = {}
        for category, tickers in POSITIONING_TICKERS.items():
            if "JPY" not in tickers:
                continue
            hist = fetch_macro_history(
                bbg, {"JPY": tickers["JPY"]}, days=FETCH_DAYS, periodicity="DAILY"
            )
            if "JPY" in hist.columns:
                jpy_data[category] = hist["JPY"]
            else:
                print(f"  WARNING: no data returned for {category} / JPY")

    spot_index = jpy_data["spot"].index
    nan_series = pd.Series(float("nan"), index=spot_index)

    def get(key, fallback_key=None):
        s = jpy_data.get(key)
        if s is None and fallback_key:
            s = jpy_data.get(fallback_key)
            if s is not None:
                print(f"  INFO: {key} missing, falling back to {fallback_key}")
        return s.reindex(spot_index) if s is not None else nan_series

    rr_1m = get("risk_reversal_1m")
    rr_6m = get("risk_reversal_6m", fallback_key="risk_reversal_1m")
    cftc_long = get("cftc_long")
    cftc_short = get("cftc_short")

    print("\nData coverage (non-NaN rows):")
    for label, s in [("spot", jpy_data["spot"]), ("rr_1m", rr_1m),
                     ("rr_6m", rr_6m), ("cftc_long", cftc_long), ("cftc_short", cftc_short)]:
        print(f"  {label:15s}: {s.notna().sum()} rows")

    print("\nComputing positioning score (CFTC percentile may take a few seconds)...")
    score = build_positioning_for_currency(
        "JPY",
        spot=jpy_data["spot"],
        rr_1m=rr_1m,
        rr_6m=rr_6m,
        cftc_long=cftc_long,
        cftc_short=cftc_short,
    )

    cutoff = pd.Timestamp.today() - pd.DateOffset(years=3)
    recent = score[score.index >= cutoff].dropna()

    print(
        f"\nJPY Positioning — last 3 years "
        f"({recent.index[0].date()} to {recent.index[-1].date()})"
    )
    print(f"Range: {recent.min():.1f} to {recent.max():.1f}  |  "
          f"Current: {recent.iloc[-1]:.1f}")
    print()
    print(recent.resample("W").last().to_string())  # weekly to keep output readable

    out = "macro/data/cache/jpy_positioning.csv"
    recent.to_csv(out)
    print(f"\nFull daily series saved to {out}")


if __name__ == "__main__":
    main()
