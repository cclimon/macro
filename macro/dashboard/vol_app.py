# dashboard/vol_app.py
# FX Vol Monitor — implied vs Yang-Zhang realized, z-score heatmap
#
# Run from repo root:
#   streamlit run dashboard/vol_app.py
#
# Requires a live Bloomberg session (xbbg/blpapi). Falls back to mock data
# automatically if blpapi is not importable, and labels the source clearly.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.vol_universe import PAIR_GROUP, TENORS, UNIVERSE
from vol import data as vol_data
from vol import signals

st.set_page_config(page_title="FX Vol Monitor", layout="wide")

# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.title("FX Vol Monitor")

sel_tenors = st.sidebar.multiselect(
    "Tenors", list(TENORS), default=list(TENORS),
    help="Implied ATM tenor; realized window is matched (1W→5d … 1Y→252d)",
)

lookback = st.sidebar.slider(
    "Z-score lookback (business days)",
    min_value=63, max_value=756, value=252, step=21,
    help="History used for the mean/std of ln(IV/RV). 252 ≈ 1y, 504 ≈ 2y.",
)

groups = st.sidebar.multiselect(
    "Universe", ["G10 Majors", "Crosses", "EM"],
    default=["G10 Majors", "Crosses", "EM"],
)
pairs = [p for p in UNIVERSE if PAIR_GROUP[p] in groups]

metric = st.sidebar.radio(
    "Cell metric", ["Z-score", "Percentile"], horizontal=True,
    help="Z assumes ~normal ln(IV/RV); percentile is robust to fat tails.",
)

force_mock = st.sidebar.checkbox("Force mock data", value=False)

# ── Data ─────────────────────────────────────────────────────────────────────
# History needed: lookback + longest RV window + buffer
history_days = lookback + max(TENORS[t][1] for t in (sel_tenors or ["1Y"])) + 30


@st.cache_data(ttl=15 * 60, show_spinner="Fetching data…")
def get_data(pairs_key: tuple, tenors_key: tuple, hist: int, mock: bool):
    return vol_data.load(list(pairs_key), list(tenors_key), hist,
                         use_mock=mock or None)


if not sel_tenors or not pairs:
    st.info("Select at least one tenor and one universe group.")
    st.stop()

implied, ohlc, source = get_data(tuple(pairs), tuple(sel_tenors),
                                 history_days, force_mock)

snap = signals.build_snapshot(implied, ohlc, sel_tenors, lookback)
zdf = snap["z"].reindex(index=pairs, columns=sel_tenors)
pdf = snap["pct"].reindex(index=pairs, columns=sel_tenors)
ivdf = snap["iv"].reindex(index=pairs, columns=sel_tenors)
rvdf = snap["rv"].reindex(index=pairs, columns=sel_tenors)

# ── Header ───────────────────────────────────────────────────────────────────
left, right = st.columns([3, 1])
with left:
    st.title("Implied vs Yang-Zhang Realized")
    st.caption(
        f"Cell = {metric.lower()} of ln(IV/RV) · lookback {lookback}d · "
        f"RV window matched to tenor · source: {source}"
    )
with right:
    if "MOCK" in source:
        st.warning("Mock data — run on a Bloomberg-connected machine.")

# ── Heatmap ──────────────────────────────────────────────────────────────────
if metric == "Z-score":
    vals = zdf
    zmin, zmax, zmid = -3.0, 3.0, 0.0
    cbar_title = "z"
else:
    vals = pdf
    zmin, zmax, zmid = 0.0, 100.0, 50.0
    cbar_title = "%ile"

hover = np.empty(vals.shape, dtype=object)
for i, p in enumerate(vals.index):
    for j, t in enumerate(vals.columns):
        hover[i, j] = (
            f"<b>{p} {t}</b><br>"
            f"IV: {ivdf.iloc[i, j]:.2f}  ·  RV(YZ): {rvdf.iloc[i, j]:.2f}<br>"
            f"z: {zdf.iloc[i, j]:+.2f}  ·  pctile: {pdf.iloc[i, j]:.0f}"
        )

annot = vals.map(lambda v: f"{v:+.1f}" if metric == "Z-score" and pd.notna(v)
                 else (f"{v:.0f}" if pd.notna(v) else ""))

fig = go.Figure(
    go.Heatmap(
        z=vals.values,
        x=list(vals.columns),
        y=list(vals.index),
        zmin=zmin, zmax=zmax, zmid=zmid,
        colorscale="RdBu_r",   # red = implied rich, blue = implied cheap
        text=annot.values,
        texttemplate="%{text}",
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        colorbar=dict(title=cbar_title, thickness=12),
        xgap=2, ygap=2,
    )
)
fig.update_layout(
    height=max(420, 26 * len(pairs)),
    yaxis=dict(autorange="reversed"),
    margin=dict(l=10, r=10, t=10, b=10),
    font=dict(size=12),
)
st.plotly_chart(fig, use_container_width=True)

# ── Drill-down ───────────────────────────────────────────────────────────────
st.divider()
c1, c2 = st.columns([1, 3])
with c1:
    dd_pair = st.selectbox("Pair", pairs)
    dd_tenor = st.selectbox("Tenor", sel_tenors)

ratio = snap["ratio_hist"][dd_tenor].get(dd_pair)
with c2:
    if ratio is not None and ratio.notna().sum() > lookback:
        zs = signals.zscore_series(
            snap["ratio_hist"][dd_tenor][[dd_pair]], lookback
        )[dd_pair].dropna()
        fig2 = go.Figure()
        fig2.add_scatter(x=zs.index, y=zs.values, mode="lines",
                         line=dict(width=1.5), name="z(ln IV/RV)")
        for lvl, dash in [(2, "dot"), (-2, "dot"), (0, "dash")]:
            fig2.add_hline(y=lvl, line_width=1, line_dash=dash, opacity=0.4)
        fig2.update_layout(
            title=f"{dd_pair} {dd_tenor} — z-score history",
            height=300, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Not enough history for this pair/tenor at current lookback.")

# ── Snapshot table ───────────────────────────────────────────────────────────
with st.expander("Snapshot table (IV / RV / z / pctile)"):
    flat = []
    for t in sel_tenors:
        for p in pairs:
            flat.append({
                "Pair": p, "Tenor": t,
                "IV": round(float(ivdf.loc[p, t]), 2),
                "RV (YZ)": round(float(rvdf.loc[p, t]), 2),
                "z": round(float(zdf.loc[p, t]), 2),
                "pctile": round(float(pdf.loc[p, t]), 1),
            })
    st.dataframe(pd.DataFrame(flat), use_container_width=True, height=400)
