"""
macro_charts.py
---------------
Plots Real GDP (GDPC1) normalised by Total Nonfarm Payrolls (PAYEMS).

  GDPC1  : Billions of Chained 2017 Dollars, SAAR (quarterly)
  PAYEMS : All Employees Total Nonfarm, Thousands of Persons (monthly)

Normalised series = GDPC1 (bn) / PAYEMS (k persons)
                  = GDP per worker in $millions (Chained 2017)

Output -> output/macro/gdp_per_worker.png
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from fredapi import Fred

# ── Config ─────────────────────────────────────────────────────────────────
FRED_API_KEY = "fe767a7edad74cf231e42a1c60e39851"
START        = "1950-01-01"
PLOT_FROM    = 1990
OUT_DIR      = Path(__file__).resolve().parent / "output" / "macro"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Fetch ───────────────────────────────────────────────────────────────────
fred = Fred(api_key=FRED_API_KEY)

print("Fetching GDPC1 (Real GDP)...")
gdp = fred.get_series("GDPC1", observation_start=START)          # quarterly, bn$

print("Fetching PAYEMS (Total Nonfarm Payrolls)...")
emp = fred.get_series("PAYEMS", observation_start=START)         # monthly, k persons

# ── Resample to annual ──────────────────────────────────────────────────────
# Use annual mean so SAAR GDP stays comparable across quarters
gdp_annual = gdp.resample("YE").mean()
emp_annual = emp.resample("YE").mean()

# Align on common years
df = pd.DataFrame({"gdp": gdp_annual, "emp": emp_annual}).dropna()
df.index = df.index.year                                         # integer year labels

# Normalise: bn$ / k persons = $m per worker (Chained 2017 Dollars)
df["gdp_per_worker"] = df["gdp"] / df["emp"] * 1_000            # → thousands of $

# ── Filter to plot window ───────────────────────────────────────────────────
plot = df[df.index >= PLOT_FROM].copy()

# ── Linear trend on plot window ─────────────────────────────────────────────
x     = np.arange(len(plot))
coeffs = np.polyfit(x, plot["gdp_per_worker"].values, 1)
trend  = np.polyval(coeffs, x)
slope_per_yr = coeffs[0]                                          # $k per year
n_years      = plot.index[-1] - plot.index[0]
cagr         = (plot["gdp_per_worker"].iloc[-1] /
                plot["gdp_per_worker"].iloc[0]) ** (1 / n_years) - 1

# ── Plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})
fig.suptitle(
    "Real GDP per Nonfarm Worker\n"
    "GDPC1 (Chained 2017 $) ÷ PAYEMS — Annual",
    fontsize=14, fontweight="bold", y=0.98
)

# Panel 1: GDP per worker + trend
ax1 = axes[0]
ax1.fill_between(plot.index, plot["gdp_per_worker"], alpha=0.15, color="#1a6fbd")
ax1.plot(plot.index, plot["gdp_per_worker"], color="#1a6fbd", lw=2, label="GDP / worker")
ax1.plot(plot.index, trend, color="#c94a2a", lw=1.8, ls="--",
         label=f"Trend  +${slope_per_yr:,.1f}k/yr  |  CAGR {cagr*100:.2f}%")
ax1.set_ylabel("Thousands of Chained 2017 $ per Worker", fontsize=10)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}k"))
ax1.set_ylim(bottom=80)
ax1.grid(alpha=0.3)
ax1.set_title("GDP per Worker (productivity proxy)", fontsize=11, loc="left")
ax1.legend(fontsize=10, loc="upper left")

# Slope annotation box
ax1.annotate(
    f"Trend slope: +${slope_per_yr:,.1f}k per year\nCAGR ({plot.index[0]}–{plot.index[-1]}): {cagr*100:.2f}%",
    xy=(0.98, 0.08), xycoords="axes fraction",
    ha="right", va="bottom", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e7", edgecolor="#ccaa00", alpha=0.9)
)

# Panel 2: YoY % change
yoy = plot["gdp_per_worker"].pct_change() * 100
colors = ["#217a3c" if v >= 0 else "#cc2222" for v in yoy]
ax2 = axes[1]
ax2.bar(plot.index, yoy, color=colors, width=0.8, alpha=0.85)
ax2.axhline(0, color="black", lw=0.8)
ax2.axhline(yoy.mean(), color="#c94a2a", lw=1.2, ls="--",
            label=f"Avg {yoy.mean():+.2f}%/yr")
ax2.set_ylabel("YoY Change (%)", fontsize=10)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
ax2.grid(alpha=0.3, axis="y")
ax2.set_title("Annual Change", fontsize=11, loc="left")
ax2.legend(fontsize=10, loc="lower left")

ax2.set_xlabel("Year", fontsize=10)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

out = OUT_DIR / "gdp_per_worker.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nChart saved -> {out}")

# ── Quick summary ───────────────────────────────────────────────────────────
latest = df.iloc[-1]
prev   = df.iloc[-2]
chg    = (latest["gdp_per_worker"] / prev["gdp_per_worker"] - 1) * 100
print(f"\nLatest year  : {df.index[-1]}")
print(f"GDP          : ${latest['gdp']:,.1f}bn")
print(f"Employment   : {latest['emp']:,.0f}k workers")
print(f"GDP/worker   : ${latest['gdp_per_worker']:,.1f}k (Chained 2017$)")
print(f"YoY change   : {chg:+.2f}%")
