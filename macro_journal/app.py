"""app.py

Macro FX Journal — Streamlit front end.

Workflow:
1. Paste raw color into the INFO box -> filed (anonymized/tagged) into today's draft.
2. Ask a question / request a summary -> answered strictly from today's filed INFO,
   structured as MACRO THESIS / FLOWS SEEN / ECONOMICS / PRICE ACTION.
3. Hit "Close EOD" -> compiles the day's draft into a compressed Markdown report,
   commits and pushes it to your GitHub repo.
"""
import streamlit as st
from dotenv import load_dotenv

import storage
import claude_client
import git_utils

load_dotenv()

st.set_page_config(page_title="Macro FX Journal", layout="wide")
st.title("Macro FX Journal")

# --- Day selector -----------------------------------------------------------
if "current_day" not in st.session_state:
    st.session_state.current_day = storage.today_str()

col_day, col_status = st.columns([1, 3])
with col_day:
    st.session_state.current_day = st.text_input(
        "Trading day (YYYY-MM-DD)", value=st.session_state.current_day
    )

day = st.session_state.current_day
draft = storage.load_draft(day)

with col_status:
    if storage.is_eod_done(day):
        st.warning(f"EOD already closed for {day}. New INFO will file under this day unless you change it above.")
    else:
        st.info(f"Filing INFO under {day}. {len(draft)} entries stored so far today.")

st.divider()

# --- INFO ingestion ----------------------------------------------------------
st.subheader("File INFO")
info_text = st.text_area("Paste raw INFO / colour here", height=200, key="info_input")

if st.button("File INFO", type="primary"):
    if not info_text.strip():
        st.error("Nothing to file.")
    else:
        with st.spinner("Filing (anonymizing, tagging, translating if needed)..."):
            entry = claude_client.file_info(info_text, draft)
            draft = storage.append_entry(day, entry)
        st.success("Filed silently. No summary produced (per filing rules).")
        st.rerun()

with st.expander(f"View {len(draft)} raw filed entries for {day}"):
    for i, e in enumerate(draft, 1):
        st.markdown(f"**Entry {i}:**")
        st.text(e.get("filed", ""))

st.divider()

# --- Question / summary prompt -----------------------------------------------
st.subheader("Ask a question / request a summary")
question = st.text_input("e.g. 'Summary GBP' or 'Comment on JPY'", key="question_input")

if st.button("Get answer"):
    if not draft:
        st.error(f"No INFO filed yet for {day}.")
    elif not question.strip():
        st.error("Enter a question first.")
    else:
        with st.spinner("Answering from today's filed INFO..."):
            answer = claude_client.answer_question(question, draft)
        st.markdown("### Answer")
        st.markdown(answer)

st.divider()

# --- EOD ----------------------------------------------------------------------
st.subheader("Close of Day (EOD)")
st.caption("Compiles today's INFO into a compressed, by-currency Markdown report and pushes it to GitHub.")

if st.button("Close EOD and push to GitHub"):
    if not draft:
        st.error(f"No INFO filed for {day} — nothing to compile.")
    else:
        with st.spinner("Compiling EOD report..."):
            eod_markdown = claude_client.compile_eod(day, draft)
            eod_file = storage.save_eod_markdown(day, eod_markdown)
        st.success(f"EOD report written to {eod_file}")
        with st.spinner("Committing and pushing to GitHub..."):
            status = git_utils.commit_and_push_eod(eod_file, day)
        st.info(status)
        st.markdown("### EOD Report")
        st.markdown(eod_markdown)

# --- Browse past EOD reports ---------------------------------------------------
st.divider()
st.subheader("Past EOD reports")
past_days = storage.list_eod_days()
if past_days:
    chosen = st.selectbox("Select a day", options=list(reversed(past_days)))
    if chosen:
        st.markdown(storage.eod_path(chosen).read_text(encoding="utf-8"))
else:
    st.caption("No EOD reports closed yet.")
