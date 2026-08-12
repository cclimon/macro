# dashboard/fiscal_app.py
# Sovereign Fiscal Dashboard — local dev / testing (Streamlit).
#
# Run from the macro/ project root:
#   streamlit run dashboard/fiscal_app.py
#
# Consumes the static snapshot produced by data/fetch_fiscal_metrics.py
# (data/fiscal_metrics_latest.json). Charts are matplotlib PNGs so they paste
# sharp into Outlook/Word; each has a one-click "Copy to clipboard" (Clipboard API)
# plus a "Download PNG" fallback.
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard import fiscal_charts as fc  # noqa: E402

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "fiscal_metrics_latest.json"

st.set_page_config(page_title="Sovereign Fiscal Dashboard", layout="wide")


@st.cache_data(ttl=15 * 60)
def load_payload():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def copy_download(png: bytes, key: str, filename: str):
    """Copy-to-clipboard (Clipboard API) + Download PNG, side by side."""
    b64 = base64.b64encode(png).decode("ascii")
    html = f"""
    <div style="display:flex;gap:8px;align-items:center;font-family:sans-serif;">
      <button id="cp_{key}" style="background:#238636;color:#fff;border:0;border-radius:6px;
        padding:6px 12px;font-size:13px;cursor:pointer;">📋 Copy to clipboard</button>
      <span id="st_{key}" style="color:#8b949e;font-size:12px;"></span>
    </div>
    <script>
      const b64_{key} = "{b64}";
      function b2blob_{key}(b64){{
        const bin = atob(b64); const len = bin.length; const u8 = new Uint8Array(len);
        for (let i=0;i<len;i++) u8[i]=bin.charCodeAt(i);
        return new Blob([u8], {{type:'image/png'}});
      }}
      document.getElementById('cp_{key}').addEventListener('click', async () => {{
        const status = document.getElementById('st_{key}');
        try {{
          const blob = b2blob_{key}(b64_{key});
          await navigator.clipboard.write([new ClipboardItem({{'image/png': blob}})]);
          status.textContent = '✓ copied — paste into Outlook/Word';
        }} catch (e) {{
          status.textContent = '✗ copy failed ('+e.name+') — use Download PNG';
        }}
      }});
    </script>
    """
    c1, c2 = st.columns([3, 2])
    with c1:
        components.html(html, height=44)
    with c2:
        st.download_button("⬇ Download PNG", data=png, file_name=filename,
                           mime="image/png", key=f"dl_{key}")


# ── load ─────────────────────────────────────────────────────────────────────
payload = load_payload()
df = fc.to_dataframe(payload)

st.title("Sovereign Fiscal Dashboard")
st.caption(
    f"General-government basis · reference year {payload['reference_year']} · "
    f"snapshot {payload['generated_at']} · "
    "sources: Bloomberg Economics (headline balance) + IMF WEO via Bloomberg (backbone). "
    "Colour: red = weaker fiscal position. r−g uses r−g > 0 = adverse (red)."
)

# ── sidebar ──────────────────────────────────────────────────────────────────
groups = sorted(df["group"].unique())
sel_groups = st.sidebar.multiselect("Group", groups, default=groups)
sort_by = st.sidebar.selectbox(
    "Sort / rank by", fc.METRIC_ORDER, index=fc.METRIC_ORDER.index("debt_pct_gdp"),
    format_func=lambda m: f"{fc.METRICS[m][0]} ({fc.METRICS[m][1]})",
)
view = st.sidebar.radio("View", ["Scorecard (table)", "Ranked bars"], index=0)

dff = df[df["group"].isin(sel_groups)].copy()
if dff.empty:
    st.info("Select at least one group.")
    st.stop()

# ── interactive sortable table (native) ──────────────────────────────────────
with st.expander("Interactive table (click a column header to sort)", expanded=False):
    show = dff.copy()
    show.columns = [c if c in ("country", "group") else f"{fc.METRICS[c][0]} ({fc.METRICS[c][1]})"
                    if c in fc.METRICS else c for c in show.columns]
    st.dataframe(show, use_container_width=True)

# ── main view ────────────────────────────────────────────────────────────────
if view == "Scorecard (table)":
    st.subheader("Fiscal scorecard")
    png = fc.render_heatmap(dff, sort_by=sort_by)
    st.image(png, use_container_width=False)
    copy_download(png, key="heatmap", filename="fiscal_scorecard.png")
else:
    st.subheader(f"Ranked: {fc.METRICS[sort_by][0]} ({fc.METRICS[sort_by][1]})")
    png = fc.render_ranked_bar(dff, sort_by)
    st.image(png, use_container_width=False)
    copy_download(png, key=f"bar_{sort_by}", filename=f"fiscal_{sort_by}.png")

    st.divider()
    st.caption("All five metrics:")
    cols = st.columns(2)
    for i, m in enumerate(fc.METRIC_ORDER):
        with cols[i % 2]:
            p = fc.render_ranked_bar(dff, m)
            st.image(p, use_container_width=False)
            copy_download(p, key=f"barall_{m}", filename=f"fiscal_{m}.png")

st.sidebar.caption(
    "Data via data/fetch_fiscal_metrics.py (Bloomberg). "
    "Charts render as PNG for clean copy-paste into client notes."
)
