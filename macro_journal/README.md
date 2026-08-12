# Macro FX Journal

Local Streamlit app for filing intraday FX/rates/macro "colour" (tagged INFO),
answering questions from it in a structured format, and closing the day (EOD)
into a compressed Markdown report that gets committed and pushed to GitHub.

## Setup

1. Copy this whole `macro_journal/` folder into your GitRepo:
   `C:\Users\CCM\OneDrive - Centile Partners Advisory Ltd\Documents\06_Models_&RISK\GitRepo`

2. Open that folder in Claude Code (or a terminal) and create a virtual env:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and fill in:
   - `ANTHROPIC_API_KEY` — your API key
   - `GITHUB_PAT` — your GitHub Personal Access Token (repo scope)
   - `GITHUB_REPO` — `cclimon/macro`
   - `LOCAL_REPO_PATH` — the full path to this folder once it's inside your git repo clone

4. Make sure this folder is actually a git repo clone of `cclimon/macro` withBCNI3M BGN Curncy
   `origin` reachable (or let `git_utils.py` create the remote using your PAT).

5. Run the app:
   ```
   streamlit run app.py
   ```

## Workflow

- **File INFO**: paste raw colour, hit "File INFO". It gets anonymized
  (named sell-side banks -> US BANK / EURO BANK / ASIAN BANK), translated if
  needed, and tagged by currency — then stored silently in
  `data/drafts/YYYY-MM-DD.json`. No summary is shown at this stage.

- **Ask a question**: e.g. "Summary GBP", "Comment on JPY". Answered strictly
  from today's filed INFO, structured as MACRO THESIS / FLOWS SEEN / ECONOMICS
  / PRICE ACTION.

- **Close EOD**: compiles the full day into a compressed by-currency Markdown
  report, writes it to `data/eod/YYYY-MM-DD.md`, commits, and pushes to GitHub.
  Any INFO filed after this automatically belongs to the next day.

## Editing the rules

All behavior rules (anonymization, formatting, structure, context-only mode)
live in `config/system_prompt.md` — edit that file directly rather than the
Python code to adjust behavior.

The currency/country tag table lives in `config/currency_codes.json` and
should be extended as new `{XX}` codes appear in your INFO feed.

## Next steps (not yet wired in)

- Live Bloomberg API pull for the ECONOMICS category (currently the app relies
  on whatever INFO/data you paste in; add a `bloomberg_client.py` module and
  call it from `claude_client.compile_eod` / `answer_question` when
  `BLOOMBERG_ENABLED=true`).
- PDF export of EOD reports (the "journal" PDF workflow) — can be added with
  `reportlab` or by converting the Markdown with `pandoc`.
- Auth/multi-user if this ever needs to be shared beyond one desk.
