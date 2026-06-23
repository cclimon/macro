# dashboard/app.py
# Streamlit G10 FX Signal Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.pairs import G10_PAIRS
from data.cache import load_signals, CACHE_DIR
import pytz

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="G10 FX Signal Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme palette ────────────────────────────────────────────────────────

BULL_COLOR  = "#00e676"   # green
BEAR_COLOR  = "#ff1744"   # red
NEUT_COLOR  = "#9e9e9e"   # grey
BG_COLOR    = "#0d1117"
CARD_COLOR  = "#161b22"
ACCENT      = "#58a6ff"

st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        background-color: {BG_COLOR};
        color: #e6edf3;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}
    .metric-card {{
        background: {CARD_COLOR};
        border-radius: 8px;
        padding: 14px 18px;
        border-left: 3px solid {ACCENT};
        margin-bottom: 8px;
    }}
    .bull {{ color: {BULL_COLOR}; font-weight: 600; }}
    .bear {{ color: {BEAR_COLOR}; font-weight: 600; }}
    .neut {{ color: {NEUT_COLOR}; font-weight: 600; }}
    .section-header {{
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {ACCENT};
        margin: 18px 0 8px 0;
        border-bottom: 1px solid #30363d;
        padding-bottom: 4px;
    }}
    div[data-testid="stDataFrame"] {{ border-radius: 8px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Colour helpers ────────────────────────────────────────────────────────────

def color_val(val, good_positive: bool = True):
    """Return styled HTML span for a numeric value."""
    if pd.isna(val):
        return '<span class="neut">N/A</span>'
    if isinstance(val, str):
        return val
    color = BULL_COLOR if (val > 0) == good_positive else BEAR_COLOR
    return f'<span style="color:{color};font-weight:600">{val}</span>'


SIGNAL_COLORS = {
    "Bull": BULL_COLOR, "Bullish ↑": BULL_COLOR, "Bullish ~": BULL_COLOR,
    "Bear": BEAR_COLOR, "Bearish ↓": BEAR_COLOR, "Bearish ~": BEAR_COLOR,
    "Strong Bull": BULL_COLOR, "Mild Bull": "#69f0ae",
    "Strong Bear": BEAR_COLOR, "Mild Bear": "#ff6e6e",
    "Overbought": BEAR_COLOR, "Oversold": BULL_COLOR,
    "Neutral": NEUT_COLOR, "N/A": NEUT_COLOR,
    "Hawkish 🦅": BULL_COLOR, "Dovish 🕊️": BEAR_COLOR,
    "High": BEAR_COLOR, "Low": BULL_COLOR, "Normal": NEUT_COLOR,
    "Trending": BULL_COLOR, "Strong": BULL_COLOR, "Weak": NEUT_COLOR,
    "Flat": NEUT_COLOR,
}

def style_label(val):
    color = SIGNAL_COLORS.get(str(val), NEUT_COLOR)
    return f'background-color: transparent; color: {color}; font-weight: 600'


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"### 📡 G10 FX Signals")
    st.divider()

    pair_filter = st.multiselect(
        "Filter pairs",
        options=G10_PAIRS,
        default=G10_PAIRS,
    )
    st.divider()

    # Data timestamp
    ts_file = CACHE_DIR / "last_updated.txt"
    if ts_file.exists():
        from datetime import datetime as _dt
        ts_raw = ts_file.read_text().strip()
        ts_utc = _dt.fromisoformat(ts_raw).replace(tzinfo=pytz.utc)
        ts_london = ts_utc.astimezone(pytz.timezone("Europe/London"))
        st.caption(f"Data as of: **{ts_london.strftime('%d %b %Y %H:%M')} London**")
    else:
        st.caption("Data as of: —")
    st.divider()
    st.caption("Signals are isolated. Scoring / weighting in next phase.")


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading cached signals …")
def _load():
    return load_signals()

try:
    data  = _load()
    tech  = data["technical"].loc[data["technical"].index.isin(pair_filter)]
    carry = data["carry"].loc[data["carry"].index.isin(pair_filter)]
    macro = data["macro"].loc[data["macro"].index.isin(pair_filter)]
    spot  = data["spot"]
    as_of = data["as_of"]
    load_ok = True
except FileNotFoundError:
    st.warning("No data found. Run `python main.py` first to generate the EOD data.")
    st.stop()
except Exception as e:
    st.error(f"Error loading cached signals: {e}")
    st.stop()


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(f"## 📡 G10 FX Signal Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("Universe", f"{len(pair_filter)} pairs")
col2.metric("Pillars", "3 (Tech · Carry · Macro)")
col3.metric("As of", as_of.strftime("%d %b %Y %H:%M") if hasattr(as_of, "strftime") else str(as_of))
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_tech, tab_carry, tab_macro, tab_chart = st.tabs(
    ["📈 Technical", "💰 Carry", "🌍 Macro", "📊 Charts"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TECHNICAL
# ══════════════════════════════════════════════════════════════════════════════

with tab_tech:
    st.markdown('<p class="section-header">Momentum & Price Action</p>', unsafe_allow_html=True)

    display_cols = {
        "rsi_14":      "RSI(14)",
        "rsi_zone":    "RSI Zone",
        "macd_signal": "MACD",
        "sma_20_50":   "SMA 20/50",
        "sma_50_200":  "SMA 50/200",
        "roc_1m":      "ROC 1m %",
        "roc_3m":      "ROC 3m %",
        "bb_pct_b":    "BB %B",
        "adx_14":      "ADX(14)",
        "adx_strength":"Trend",
        "zscore_1m":   "Z-score 1m",
        "zscore_3m":   "Z-score 3m",
        "zscore_1y":   "Z-score 1Y",
    }
    t_disp = tech.rename(columns=display_cols)[list(display_cols.values())]

    # Styler
    def style_tech(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for col in ["RSI Zone", "MACD", "SMA 20/50", "SMA 50/200", "Trend"]:
            if col in df.columns:
                styles[col] = df[col].apply(style_label)
        for col in ["ROC 1m %", "ROC 3m %", "Z-score 1m", "Z-score 3m", "Z-score 1Y"]:
            if col in df.columns:
                styles[col] = df[col].apply(
                    lambda v: f"color:{BULL_COLOR};font-weight:600" if (not pd.isna(v) and v > 0)
                    else (f"color:{BEAR_COLOR};font-weight:600" if not pd.isna(v) else "")
                )
        return styles

    st.dataframe(
        t_disp.style.apply(style_tech, axis=None).format(
            {
                "RSI(14)":  "{:.1f}",
                "ROC 1m %": "{:+.2f}%",
                "ROC 3m %": "{:+.2f}%",
                "BB %B":    "{:.3f}",
                "ADX(14)":    "{:.1f}",
                "Z-score 1m": "{:+.2f}",
                "Z-score 3m": "{:+.2f}",
                "Z-score 1Y": "{:+.2f}",
            },
            na_rep="N/A",
        ),
        use_container_width=True,
        height=560,
    )

    with st.expander("ℹ️ Signal definitions"):
        st.markdown("""
        | Signal | Description |
        |--------|-------------|
        | **RSI(14)** | Relative Strength Index. >70 = Overbought, <30 = Oversold |
        | **MACD** | 12/26/9 EMA. Histogram direction shows momentum |
        | **SMA Cross** | Short MA vs long MA — Bull if short above long |
        | **ROC** | Rate of change over 1m / 3m lookback |
        | **BB %B** | Position within Bollinger Bands (0=lower, 1=upper, 0.5=mid) |
        | **ADX** | Trend strength. >25 = trending, >40 = strong. Uses real High/Low |
        | **Z-score 1m** | Current price vs 21-day rolling mean in standard deviations |
        | **Z-score 3m** | Current price vs 63-day rolling mean in standard deviations |
        | **Z-score 1Y** | Current price vs 252-day rolling mean in standard deviations |
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CARRY
# ══════════════════════════════════════════════════════════════════════════════

with tab_carry:
    st.markdown('<p class="section-header">Carry & Volatility Metrics</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Carry metrics**")
        carry_cols = {
            "rate_diff_bps":    "Rate Diff (bps)",
            "real_carry_bps":   "Real Carry (bps)",
            "fwd_carry_ann_pct":"Fwd Carry Ann %",
            "cpi_diff_pct":     "CPI Diff %",
        }
        c_disp = carry.rename(columns=carry_cols)[list(carry_cols.values())]

        def style_carry(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for col in ["Rate Diff (bps)", "Real Carry (bps)", "Fwd Carry Ann %"]:
                if col in df.columns:
                    styles[col] = df[col].apply(
                        lambda v: f"color:{BULL_COLOR};font-weight:600" if (not pd.isna(v) and v > 0)
                        else (f"color:{BEAR_COLOR};font-weight:600" if not pd.isna(v) else "")
                    )
            return styles

        st.dataframe(
            c_disp.style.apply(style_carry, axis=None).format(
                {
                    "Rate Diff (bps)":   "{:+.1f}",
                    "Real Carry (bps)":  "{:+.1f}",
                    "Fwd Carry Ann %":   "{:+.3f}%",
                    "CPI Diff %":        "{:+.2f}%",
                },
                na_rep="N/A",
            ),
            use_container_width=True,
            height=560,
        )

    with c2:
        st.markdown("**Volatility & risk-adjusted carry**")
        vol_cols = {
            "rvol_1m_pct":  "RVol 1m %",
            "rvol_3m_pct":  "RVol 3m %",
            "carry_vol_1m": "Carry/Vol 1m",
            "carry_vol_3m": "Carry/Vol 3m",
            "vol_regime":   "Vol Regime",
        }
        v_disp = carry.rename(columns=vol_cols)[list(vol_cols.values())]

        def style_vol(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            if "Vol Regime" in df.columns:
                styles["Vol Regime"] = df["Vol Regime"].apply(style_label)
            for col in ["Carry/Vol 1m", "Carry/Vol 3m"]:
                if col in df.columns:
                    styles[col] = df[col].apply(
                        lambda v: f"color:{BULL_COLOR};font-weight:600" if (not pd.isna(v) and v > 0)
                        else (f"color:{BEAR_COLOR};font-weight:600" if not pd.isna(v) else "")
                    )
            return styles

        st.dataframe(
            v_disp.style.apply(style_vol, axis=None).format(
                {
                    "RVol 1m %":    "{:.2f}%",
                    "RVol 3m %":    "{:.2f}%",
                    "Carry/Vol 1m": "{:+.3f}",
                    "Carry/Vol 3m": "{:+.3f}",
                },
                na_rep="N/A",
            ),
            use_container_width=True,
            height=560,
        )

    # Carry/Vol bar chart
    st.markdown('<p class="section-header">Risk-adjusted carry ranking</p>', unsafe_allow_html=True)
    cv_data = carry["carry_vol_1m"].dropna().sort_values(ascending=False)
    fig_cv = go.Figure(go.Bar(
        x=cv_data.index,
        y=cv_data.values,
        marker_color=[BULL_COLOR if v > 0 else BEAR_COLOR for v in cv_data.values],
        text=[f"{v:+.3f}" for v in cv_data.values],
        textposition="outside",
    ))
    fig_cv.update_layout(
        title="Carry / Realised Vol (1m) — ranked",
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        font_color="#e6edf3",
        yaxis_title="Carry/Vol ratio",
        height=350,
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_cv, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MACRO
# ══════════════════════════════════════════════════════════════════════════════

with tab_macro:
    st.markdown('<p class="section-header">Macro Economic Differentials</p>', unsafe_allow_html=True)

    m1, m2 = st.columns(2)

    with m1:
        st.markdown("**PMI & growth**")
        pmi_cols = {
            "pmi_base":       "PMI Base",
            "pmi_quote":      "PMI Quote",
            "pmi_diff":       "PMI Diff",
            "pmi_signal":     "PMI Signal",
            "pmi_mom_diff":   "PMI MoM Δ",
        }
        p_disp = macro.rename(columns=pmi_cols)[list(pmi_cols.values())]

        def style_pmi(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            if "PMI Signal" in df.columns:
                styles["PMI Signal"] = df["PMI Signal"].apply(style_label)
            for col in ["PMI Diff", "PMI MoM Δ"]:
                if col in df.columns:
                    styles[col] = df[col].apply(
                        lambda v: f"color:{BULL_COLOR};font-weight:600" if (not pd.isna(v) and v > 0)
                        else (f"color:{BEAR_COLOR};font-weight:600" if not pd.isna(v) else "")
                    )
            return styles

        st.dataframe(
            p_disp.style.apply(style_pmi, axis=None).format(
                {"PMI Base":"{:.1f}", "PMI Quote":"{:.1f}",
                 "PMI Diff":"{:+.1f}", "PMI MoM Δ":"{:+.2f}"},
                na_rep="N/A",
            ),
            use_container_width=True, height=560,
        )

    with m2:
        st.markdown("**Inflation & central bank**")
        cb_cols = {
            "cpi_base_yoy":         "CPI Base YoY",
            "cpi_quote_yoy":        "CPI Quote YoY",
            "cpi_diff_pct":         "CPI Diff %",
            "cb_bias_base":         "CB Base",
            "cb_bias_quote":        "CB Quote",
            "cb_diff_score":        "CB Diff",
            "policy_rate_diff_bps": "Policy Δ (bps)",
        }
        cb_disp = macro.rename(columns=cb_cols)[list(cb_cols.values())]

        def style_cb(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for col in ["CB Base", "CB Quote"]:
                if col in df.columns:
                    styles[col] = df[col].apply(style_label)
            for col in ["CPI Diff %", "CB Diff", "Policy Δ (bps)"]:
                if col in df.columns:
                    styles[col] = df[col].apply(
                        lambda v: f"color:{BULL_COLOR};font-weight:600" if (not pd.isna(v) and v > 0)
                        else (f"color:{BEAR_COLOR};font-weight:600" if not pd.isna(v) else "")
                    )
            return styles

        st.dataframe(
            cb_disp.style.apply(style_cb, axis=None).format(
                {"CPI Base YoY":"{:.2f}%", "CPI Quote YoY":"{:.2f}%",
                 "CPI Diff %":"{:+.2f}%", "CB Diff":"{:+d}",
                 "Policy Δ (bps)":"{:+.1f}"},
                na_rep="N/A",
            ),
            use_container_width=True, height=560,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CHARTS
# ══════════════════════════════════════════════════════════════════════════════

with tab_chart:
    st.markdown('<p class="section-header">Price Charts with Signals</p>', unsafe_allow_html=True)

    selected_pair = st.selectbox("Select pair", options=pair_filter, index=0)

    if selected_pair in spot.columns:
        px_series = spot[selected_pair].dropna().iloc[-126:]   # 6m window

        # ── Price + Bollinger chart ───────────────────────────────────────────
        from signals.technical import compute_bollinger, compute_rsi, compute_macd

        bb = compute_bollinger(px_series)
        rsi_s = compute_rsi(px_series)
        macd_df = compute_macd(px_series)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=px_series.index, y=bb["upper"],
            name="BB Upper", line=dict(color="#444", dash="dash", width=1),
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=px_series.index, y=bb["lower"],
            name="BB Lower", line=dict(color="#444", dash="dash", width=1),
            fill="tonexty", fillcolor="rgba(88,166,255,0.05)",
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=px_series.index, y=px_series.values,
            name=selected_pair, line=dict(color=ACCENT, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=px_series.index, y=bb["mid"],
            name="SMA20", line=dict(color="#f0a500", width=1, dash="dot"),
        ))
        fig.update_layout(
            paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
            font_color="#e6edf3",
            title=f"{selected_pair} — 6m with Bollinger Bands",
            height=380, margin=dict(t=40, b=10),
            legend=dict(orientation="h", y=1.08),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#21262d"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── RSI ───────────────────────────────────────────────────────────────
        c_rsi, c_macd = st.columns(2)
        with c_rsi:
            fig_rsi = go.Figure()
            fig_rsi.add_hline(y=70, line_dash="dash", line_color=BEAR_COLOR, opacity=0.5)
            fig_rsi.add_hline(y=30, line_dash="dash", line_color=BULL_COLOR, opacity=0.5)
            fig_rsi.add_hline(y=50, line_dash="dot", line_color=NEUT_COLOR, opacity=0.3)
            fig_rsi.add_trace(go.Scatter(
                x=rsi_s.index, y=rsi_s.values,
                name="RSI(14)", line=dict(color="#ffd700", width=2),
            ))
            fig_rsi.update_layout(
                paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
                font_color="#e6edf3", title="RSI (14)",
                height=250, margin=dict(t=40, b=10),
                yaxis=dict(range=[0, 100], showgrid=True, gridcolor="#21262d"),
                xaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

        with c_macd:
            fig_macd = go.Figure()
            colors = [BULL_COLOR if v >= 0 else BEAR_COLOR for v in macd_df["histogram"]]
            fig_macd.add_trace(go.Bar(
                x=macd_df.index, y=macd_df["histogram"],
                name="Histogram", marker_color=colors, opacity=0.7,
            ))
            fig_macd.add_trace(go.Scatter(
                x=macd_df.index, y=macd_df["macd"],
                name="MACD", line=dict(color=ACCENT, width=1.5),
            ))
            fig_macd.add_trace(go.Scatter(
                x=macd_df.index, y=macd_df["signal"],
                name="Signal", line=dict(color="#f0a500", width=1.5),
            ))
            fig_macd.update_layout(
                paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
                font_color="#e6edf3", title="MACD (12/26/9)",
                height=250, margin=dict(t=40, b=10),
                yaxis=dict(showgrid=True, gridcolor="#21262d"),
                xaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_macd, use_container_width=True)
