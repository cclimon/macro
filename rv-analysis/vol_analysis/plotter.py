"""
plotter.py — 4-panel realized-vs-implied-vol dashboard (one per tenor).
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from vol_analysis.config import (
    PANEL_COLORS,
    PLOT_STYLE,
    SIGNAL_COLORS,
    SIGNAL_CHEAP_THRESHOLD,
    SIGNAL_RICH_THRESHOLD,
)

matplotlib.use("Agg")   # non-interactive backend for file output


# ── Public entry point ────────────────────────────────────────────────────────

def plot_dashboard(
    pair: str,
    tenor: str,
    data: pd.DataFrame,
    output_dir: str = ".",
) -> str:
    """Build and save a 4-panel vol analysis dashboard as PNG.

    Parameters
    ----------
    pair:       Currency pair string (e.g. 'EURUSD').
    tenor:      Tenor label (e.g. '3M').
    data:       DataFrame returned by spread_analysis.compute_percentile_ranks().
                Required columns: IV, YZ_RV, EWMA_RV, Spread_YZ, Spread_EWMA,
                YZ_pct_1Y, YZ_pct_2Y, EWMA_pct_1Y, EWMA_pct_2Y.
    output_dir: Directory in which to save the PNG.

    Returns
    -------
    Absolute path to the saved PNG file.
    """
    fig, axes = plt.subplots(
        4, 1,
        figsize=(14, 16),
        gridspec_kw={"height_ratios": PLOT_STYLE["height_ratios"]},
        sharex=False,
    )
    fig.patch.set_facecolor(PLOT_STYLE["figure_facecolor"])

    _apply_style(axes)

    _panel1_vol_history(axes[0], data, pair, tenor)
    _panel2_spread(axes[1], data)
    _panel3_percentile(axes[2], data)
    _panel4_snapshot(axes[3], data, pair, tenor)

    fig.tight_layout(pad=2.0)

    filename = f"{pair}_{tenor}_vol_analysis.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(
        filepath,
        dpi=PLOT_STYLE["dpi"],
        bbox_inches="tight",
        facecolor=PLOT_STYLE["figure_facecolor"],
    )
    plt.close(fig)
    return os.path.abspath(filepath)


# ── Panel renderers ───────────────────────────────────────────────────────────

def _panel1_vol_history(ax: plt.Axes, data: pd.DataFrame, pair: str, tenor: str) -> None:
    """Panel 1 — Vol History: YZ RV, EWMA RV, Implied Vol."""
    ax.plot(data.index, data["YZ_RV"],   color=PANEL_COLORS["yz"],   lw=1.4, label="Yang-Zhang RV")
    ax.plot(data.index, data["EWMA_RV"], color=PANEL_COLORS["ewma"], lw=1.4, label="EWMA RV")
    ax.plot(data.index, data["IV"],      color=PANEL_COLORS["iv"],   lw=1.8, label="Implied Vol")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.set_title(
        f"{pair} {tenor} — Realized vs Implied Volatility",
        fontsize=PLOT_STYLE["title_fontsize"],
        fontfamily=PLOT_STYLE["font_family"],
    )
    ax.legend(loc="upper right", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_ylabel("Volatility", fontsize=PLOT_STYLE["label_fontsize"],
                  fontfamily=PLOT_STYLE["font_family"])


def _panel2_spread(ax: plt.Axes, data: pd.DataFrame) -> None:
    """Panel 2 — Spread to Implied: IV − RV with fill."""
    for col, color, label in [
        ("Spread_YZ",   PANEL_COLORS["yz"],   "Spread YZ (IV − YZ RV)"),
        ("Spread_EWMA", PANEL_COLORS["ewma"], "Spread EWMA (IV − EWMA RV)"),
    ]:
        s = data[col].dropna()
        ax.plot(s.index, s.values, color=color, lw=1.4, label=label)
        ax.fill_between(
            s.index, s.values, 0,
            where=s.values > 0,
            color=PANEL_COLORS["fill_pos"], alpha=0.15,
        )
        ax.fill_between(
            s.index, s.values, 0,
            where=s.values < 0,
            color=PANEL_COLORS["fill_neg"], alpha=0.15,
        )

    ax.axhline(0, color=PANEL_COLORS["zero"], lw=1.2)
    ax.set_ylabel("Vol Points (%)", fontsize=PLOT_STYLE["label_fontsize"],
                  fontfamily=PLOT_STYLE["font_family"])
    ax.legend(loc="upper right", fontsize=PLOT_STYLE["label_fontsize"])


def _panel3_percentile(ax: plt.Axes, data: pd.DataFrame) -> None:
    """Panel 3 — Percentile Rank of Spread (1Y and 2Y lookbacks)."""
    last = data.iloc[-1]

    specs = [
        ("YZ_pct_1Y",   PANEL_COLORS["yz"],   "solid",   "1Y"),
        ("YZ_pct_2Y",   PANEL_COLORS["yz"],   "dashed",  "2Y"),
        ("EWMA_pct_1Y", PANEL_COLORS["ewma"], "solid",   "1Y"),
        ("EWMA_pct_2Y", PANEL_COLORS["ewma"], "dashed",  "2Y"),
    ]
    label_map = {
        "YZ_pct_1Y":   "YZ Spread 1Y",
        "YZ_pct_2Y":   "YZ Spread 2Y",
        "EWMA_pct_1Y": "EWMA Spread 1Y",
        "EWMA_pct_2Y": "EWMA Spread 2Y",
    }

    for col, color, ls, lookback in specs:
        s       = data[col].dropna()
        cur_val = last[col] if not np.isnan(last[col]) else float("nan")
        if not np.isnan(cur_val):
            lbl = f"{label_map[col]} [{cur_val:.0f}th]"
        else:
            lbl = label_map[col]
        ax.plot(s.index, s.values, color=color, linestyle=ls, lw=1.4, label=lbl)

    for level in [25, 50, 75]:
        ax.axhline(level, color=PANEL_COLORS["ref_line"], lw=0.8, linestyle="--", alpha=0.6)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentile Rank", fontsize=PLOT_STYLE["label_fontsize"],
                  fontfamily=PLOT_STYLE["font_family"])
    ax.legend(loc="upper right", fontsize=PLOT_STYLE["label_fontsize"])


def _panel4_snapshot(ax: plt.Axes, data: pd.DataFrame, pair: str, tenor: str) -> None:
    """Panel 4 — Text snapshot as of the last available date (no axes)."""
    ax.axis("off")
    ax.set_facecolor(PLOT_STYLE["axes_facecolor"])

    last      = data.iloc[-1]
    last_date = data.index[-1].strftime("%Y-%m-%d")

    iv_val      = last["IV"]
    yz_val      = last["YZ_RV"]
    ewma_val    = last["EWMA_RV"]
    spr_yz      = last["Spread_YZ"]
    spr_ewma    = last["Spread_EWMA"]
    pct_yz_2y   = last["YZ_pct_2Y"]
    pct_ewma_2y = last["EWMA_pct_2Y"]

    def _signal(pct: float) -> str:
        if np.isnan(pct):
            return "NEUTRAL"
        if pct > SIGNAL_RICH_THRESHOLD:
            return "IV RICH"
        if pct < SIGNAL_CHEAP_THRESHOLD:
            return "IV CHEAP"
        return "NEUTRAL"

    sig_yz   = _signal(pct_yz_2y)
    sig_ewma = _signal(pct_ewma_2y)

    sep = "─" * 45
    x0, y0 = 0.02, 0.95
    lh      = 0.09   # line height in axes-fraction units
    fs      = 9
    mono    = "monospace"

    def row(y: float, text: str, color: str = "black") -> None:
        ax.text(x0, y, text, transform=ax.transAxes,
                fontsize=fs, fontfamily=mono, color=color,
                verticalalignment="top")

    row(y0,        f"SNAPSHOT — {pair} {tenor}   as of {last_date}")
    row(y0 - lh,   sep)
    row(y0 - 2*lh, f"  Implied Vol        :   {iv_val:>6.2f}%")
    row(y0 - 3*lh, f"  Yang-Zhang RV      :   {yz_val:>6.2f}%")
    row(y0 - 4*lh, f"  EWMA RV            :   {ewma_val:>6.2f}%")
    row(y0 - 5*lh,
        f"  Spread YZ          :   {spr_yz:>+6.2f} vol pts   "
        f"[{pct_yz_2y:.0f}th pct / 2Y]" if not np.isnan(pct_yz_2y) else
        f"  Spread YZ          :   {spr_yz:>+6.2f} vol pts   [N/A]")
    row(y0 - 6*lh,
        f"  Spread EWMA        :   {spr_ewma:>+6.2f} vol pts   "
        f"[{pct_ewma_2y:.0f}th pct / 2Y]" if not np.isnan(pct_ewma_2y) else
        f"  Spread EWMA        :   {spr_ewma:>+6.2f} vol pts   [N/A]")

    # Signal YZ label + coloured value on same row using two text() calls
    lbl_y  = y0 - 7*lh
    ax.text(x0, lbl_y, "  Signal YZ          :   ",
            transform=ax.transAxes, fontsize=fs, fontfamily=mono,
            color="black", verticalalignment="top")
    ax.text(0.35, lbl_y, sig_yz,
            transform=ax.transAxes, fontsize=fs, fontfamily=mono,
            color=SIGNAL_COLORS[sig_yz], verticalalignment="top", fontweight="bold")

    lbl_y2 = y0 - 8*lh
    ax.text(x0, lbl_y2, "  Signal EWMA        :   ",
            transform=ax.transAxes, fontsize=fs, fontfamily=mono,
            color="black", verticalalignment="top")
    ax.text(0.35, lbl_y2, sig_ewma,
            transform=ax.transAxes, fontsize=fs, fontfamily=mono,
            color=SIGNAL_COLORS[sig_ewma], verticalalignment="top", fontweight="bold")

    row(y0 - 9*lh, sep)


# ── Style helpers ─────────────────────────────────────────────────────────────

def _apply_style(axes) -> None:
    """Apply the shared chart style to all axes."""
    for ax in axes:
        ax.set_facecolor(PLOT_STYLE["axes_facecolor"])
        ax.yaxis.grid(True, color=PLOT_STYLE["grid_color"], alpha=PLOT_STYLE["grid_alpha"])
        ax.xaxis.grid(False)
        for spine in ax.spines.values():
            spine.set_edgecolor(PLOT_STYLE["spine_color"])
        ax.tick_params(labelsize=PLOT_STYLE["label_fontsize"])
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily(PLOT_STYLE["font_family"])
