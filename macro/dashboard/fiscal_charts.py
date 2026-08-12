# dashboard/fiscal_charts.py
# Matplotlib rendering for the sovereign fiscal dashboard.
#
# Design (per the dataviz method):
#   - DIVERGING red<->gray<->green for signed metrics (balance, primary, r-g),
#     centered at 0, oriented so RED = weaker fiscal position everywhere.
#   - SEQUENTIAL gray->red for magnitude metrics (debt, interest) where more = worse.
#   - No in-plot text annotations on bar charts (value read from the axis); the
#     heatmap IS the numeric table, so cell values are the data, not annotations.
#   - Rendered to PNG at high DPI so it stays sharp pasted into Outlook/Word.
from __future__ import annotations

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize

# ── theme (matches .streamlit/config.toml) ───────────────────────────────────
BG = "#0d1117"
PANEL = "#161b22"
INK = "#e6edf3"
MUTED = "#8b949e"
GRID = "#30363d"
RED = "#f85149"
GREEN = "#3fb950"
NEUTRAL = "#6e7681"

DIVERGING = LinearSegmentedColormap.from_list("fiscal_div", [RED, NEUTRAL, GREEN])
SEQ_WORSE = LinearSegmentedColormap.from_list("fiscal_seq", ["#2d333b", "#8f3d3d", RED])

DPI = 220  # 2-3x display; crisp when pasted larger

# ── metric metadata ──────────────────────────────────────────────────────────
# key -> (short label, unit, kind, higher_is_better)
#   kind: "div" = diverging about 0 ; "seq" = sequential (higher = worse)
METRICS = {
    "fiscal_balance_pct_gdp":  ("Fiscal balance", "% GDP", "div", True),
    "primary_balance_pct_gdp": ("Primary balance", "% GDP", "div", True),
    "debt_pct_gdp":            ("Gross debt", "% GDP", "seq", False),
    "interest_pct_gdp":        ("Interest cost", "% GDP", "seq", False),
    "interest_pct_revenue":    ("Interest cost", "% revenue", "seq", False),
    "r_minus_g":               ("r − g", "pp", "div", False),
}
METRIC_ORDER = list(METRICS)


def _apply_theme(fig, *axes):
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(INK)


# ── data shaping ───────────────────────────────────────────────────────────────
def to_dataframe(payload: dict) -> pd.DataFrame:
    rows = []
    for c in payload["countries"]:
        row = {"country": c["country"], "iso2": c["iso2"], "group": c["group"]}
        for m in METRIC_ORDER:
            row[m] = c.get(m)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("iso2")
    return df


def _goodness(df: pd.DataFrame, metric: str) -> pd.Series:
    """Signed 'strength' with weaker = negative (red), stronger = positive (green)."""
    _, _, _, hib = METRICS[metric]
    v = df[metric].astype(float)
    return v if hib else -v


def _robust_lim(values) -> float:
    """Symmetric colour limit robust to outliers (e.g. Turkey's inflation-driven r-g):
    clip at the 85th percentile of |value| so mid-range keeps contrast and extremes saturate."""
    a = np.abs(np.asarray(values, dtype=float))
    a = a[~np.isnan(a)]
    if a.size == 0:
        return 1.0
    return max(float(np.percentile(a, 85)), 1e-6)


# ── charts ───────────────────────────────────────────────────────────────────
def render_ranked_bar(df: pd.DataFrame, metric: str, ascending_best_first=True) -> bytes:
    label, unit, kind, hib = METRICS[metric]
    s = df[metric].dropna().astype(float)
    # sort so the strongest position is on top
    good = _goodness(df, metric).reindex(s.index)
    order = good.sort_values(ascending=False).index
    s = s.reindex(order)

    # colours
    if kind == "div":
        lim = _robust_lim(_signed_for_color(s, hib))
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        colors = DIVERGING(np.clip(norm(_signed_for_color(s, hib)), 0, 1))
    else:
        gv = _goodness(df, metric).reindex(s.index)  # higher=better; worse=red
        norm = Normalize(vmin=float(np.nanmin(gv)), vmax=float(np.nanmax(gv)))
        colors = SEQ_WORSE(1 - norm(gv.values))       # invert: worse -> deep red

    fig, ax = plt.subplots(figsize=(6.2, max(3.2, 0.34 * len(s) + 1.0)))
    y = np.arange(len(s))[::-1]
    ax.barh(y, s.values, color=colors, height=0.66, edgecolor=BG, linewidth=0.8)
    ax.axvline(0, color=MUTED, lw=1.0, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(s.index, fontsize=9, color=INK)
    ax.set_xlabel(f"{label} ({unit})")
    ax.set_title(f"{label} — {unit}", fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    _apply_theme(fig, ax)
    ax.margins(y=0.01)
    fig.tight_layout()
    return _to_png(fig)


def _signed_for_color(s: pd.Series, higher_is_better: bool) -> np.ndarray:
    # For diverging colouring we want red on the "weak" side. If higher_is_better,
    # the raw value already has surplus>0 (green); else flip so worse>0 maps to red.
    return s.values if higher_is_better else -s.values


def render_heatmap(df: pd.DataFrame, metrics=None, sort_by="debt_pct_gdp") -> bytes:
    """Colored numeric table: rows=countries, cols=metrics. Cell text = value."""
    metrics = metrics or METRIC_ORDER
    d = df.copy()
    if sort_by in d.columns:
        d = d.reindex(_goodness(d, sort_by).sort_values(ascending=False).index)

    ncol, nrow = len(metrics), len(d)
    fig, ax = plt.subplots(figsize=(1.35 * ncol + 1.6, 0.34 * nrow + 1.2))

    # per-column normalized colour matrix
    rgba = np.zeros((nrow, ncol, 4))
    for j, m in enumerate(metrics):
        _, _, kind, hib = METRICS[m]
        col = d[m].astype(float)
        if kind == "div":
            lim = _robust_lim(_signed_for_color(col, hib))
            norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
            rgba[:, j, :] = DIVERGING(np.clip(norm(_signed_for_color(col, hib)), 0, 1))
        else:
            g = (col if hib else -col)
            n = Normalize(vmin=float(np.nanmin(g)), vmax=float(np.nanmax(g)))
            rgba[:, j, :] = SEQ_WORSE(1 - n(g.values))
        for i in range(nrow):
            if pd.isna(col.values[i]):
                rgba[i, j, :] = matplotlib.colors.to_rgba(PANEL)

    ax.imshow(rgba, aspect="auto")
    # cell values (data, not annotations)
    for i in range(nrow):
        for j, m in enumerate(metrics):
            v = d[m].values[i]
            txt = "—" if pd.isna(v) else f"{v:.1f}"
            lum = 0.299 * rgba[i, j, 0] + 0.587 * rgba[i, j, 1] + 0.114 * rgba[i, j, 2]
            ax.text(j, i, txt, ha="center", va="center",
                    color="#0b0d10" if lum > 0.6 else INK, fontsize=8)

    ax.set_xticks(np.arange(ncol))
    ax.set_xticklabels([f"{METRICS[m][0]}\n{METRICS[m][1]}" for m in metrics],
                       fontsize=8, color=INK)
    ax.set_yticks(np.arange(nrow))
    ax.set_yticklabels([f"{iso}" for iso in d.index], fontsize=8, color=INK)
    ax.set_xticks(np.arange(-.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nrow, 1), minor=True)
    ax.grid(which="minor", color=BG, lw=1.4)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    ax.set_title("Sovereign fiscal scorecard — red = weaker position",
                 fontsize=12, fontweight="bold", loc="left", pad=10, color=INK)
    _apply_theme(fig, ax)
    fig.tight_layout()
    return _to_png(fig)


def _to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=BG, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
