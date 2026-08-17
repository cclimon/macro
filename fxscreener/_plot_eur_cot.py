import sys
sys.path.insert(0, ".")
import bbg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print("Fetching EUR CoT longs...")
l = bbg.history("IMMBENCL Index", start="2010-01-01")
print(f"  longs : {len(l)} obs, last {l.index[-1].date()}")

print("Fetching EUR CoT shorts...")
s = bbg.history("IMMBENCS Index", start="2010-01-01")
print(f"  shorts: {len(s)} obs, last {s.index[-1].date()}")

common = l.index.intersection(s.index)
l, s = l.loc[common], s.loc[common]
net  = l - s
oi   = l + s
skew = net / oi.replace(0, float("nan"))

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
fig.suptitle("EUR CoT — IMMBENCL / IMMBENCS Index", fontsize=13)

axes[0].bar(l.index, l.values, width=5, color="steelblue", label="Longs")
axes[0].bar(s.index, -s.values, width=5, color="salmon", label="Shorts (inverted)")
axes[0].axhline(0, color="black", lw=0.5)
axes[0].set_ylabel("Contracts")
axes[0].legend(fontsize=8)
axes[0].set_title("Gross Longs vs Shorts")

axes[1].plot(net.index, net.values, color="darkblue", lw=1)
axes[1].axhline(0, color="black", lw=0.5)
axes[1].fill_between(net.index, net.values, 0,
                     where=net.values > 0, alpha=0.2, color="steelblue")
axes[1].fill_between(net.index, net.values, 0,
                     where=net.values < 0, alpha=0.2, color="salmon")
axes[1].set_ylabel("Contracts")
axes[1].set_title("Net (Long - Short)")

ROLL_WEEKS = 104   # 2-year rolling window for percentile bands
roll = skew.rolling(ROLL_WEEKS, min_periods=ROLL_WEEKS // 2)
r80 = roll.quantile(0.80)
r20 = roll.quantile(0.20)

axes[2].plot(skew.index, skew.values, color="darkgreen", lw=1, label="Skew")
axes[2].plot(r80.index, r80.values, color="red",  lw=0.9, ls="--", label="80th pct (2y roll)")
axes[2].plot(r20.index, r20.values, color="blue", lw=0.9, ls="--", label="20th pct (2y roll)")
axes[2].fill_between(skew.index, r20.values, r80.values, alpha=0.08, color="grey")
axes[2].axhline(0, color="black", lw=0.5)
axes[2].fill_between(skew.index, skew.values, 0,
                     where=skew.values > 0, alpha=0.15, color="green")
axes[2].fill_between(skew.index, skew.values, 0,
                     where=skew.values < 0, alpha=0.15, color="red")
axes[2].set_ylabel("Ratio")
axes[2].set_title("Skew = Net / OI  (dashed = 2y rolling 20/80th pct)")
axes[2].legend(fontsize=8)

axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[2].xaxis.set_major_locator(mdates.YearLocator(2))
fig.autofmt_xdate()
plt.tight_layout()

out = "eur_cot_skew.png"
plt.savefig(out, dpi=130)
print(f"\nSaved -> {out}")

last = skew.iloc[-1]
pct  = (skew <= last).mean() * 100
print(f"Current skew : {last:.3f}  ({pct:.0f}th percentile since 2010)")
print(f"Current net  : {int(net.iloc[-1]):,} contracts")
print(f"Current OI   : {int(oi.iloc[-1]):,} contracts")
