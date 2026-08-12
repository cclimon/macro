# Macro FX Journal

Sell-side FX desk tool. Captures unstructured market colour through the
trading day, files it under a trading date, answers ad-hoc questions from
that day's filings, and compiles an end-of-day report.

## Stack

- Backend: Python 3.9+, standard library only. One file, `server.py`.
- Frontend: vanilla HTML/CSS/JS in `public/`. No build step, no frameworks.
- Port 3170, loopback only (127.0.0.1 and ::1 — Safari resolves `localhost`
  to IPv6, so a v4-only bind gives "Load failed" there).

## Run

```
python3 server.py
```

Then open http://127.0.0.1:3170 (or http://localhost:3170).

`MOCK_MODE=true` is the default — no API key or `anthropic` package needed.
Everything works end to end in mock mode: filing, tagging, summaries, EOD,
and (if this folder is a git repo) commit and push.

To switch to real Claude calls:

```
pip install anthropic
export MOCK_MODE=false
export ANTHROPIC_API_KEY=sk-...
python3 server.py
```

Optional overrides: `CLAUDE_MODEL`, `ANON_MODEL`, `SUMMARY_MODEL` (default
`claude-opus-4-8`), `EOD_PUSH=false` to commit without pushing.

## Data model

One JSON file per trading day: `data/YYYY-MM-DD.json`

```json
{
  "date": "2026-07-20",
  "status": "open",
  "entries": [
    {"seq": 1, "ts": "2026-07-20T09:12:03Z", "tags": ["GBP"], "raw": "...", "filed": "..."}
  ]
}
```

Every filing re-reads the file, appends, and writes atomically (temp file +
`os.replace`), so a crash mid-write never loses the day. `raw` is the paste
exactly as received (only leading/trailing whitespace trimmed — interior
line breaks, tables, accents, emoji pass through untouched). `filed` is the
anonymised version, identical to `raw` in mock mode.

## Currency tagging

Rule-based, no model call, in `tagging.py`. Priority order: explicit `{XX}`
bracket codes, then pairs (compact and slashed), then standalone ISO codes,
then an alias dictionary, then the case-sensitive `BI` rule, then currency
symbols. Run the test battery:

```
python3 test_tagging.py
```

28 cases, including the deliberate traps ("the bulls **won** the session"
must not tag NOK; "**TRY** TO KEEP POWDER DRY" must not tag TRY; `MPC`,
`real`, `peso` alone must not tag anything).

## Output contract

Every summary and every EOD section uses exactly these four headers, in
this order, plain text:

```
*MACRO THESIS*
*FLOWS SEEN*
*ECONOMICS*
*PRICE ACTION*
```

Prose under each, never bullets. Currency pairs without a slash (`GBPUSD`,
not `GBP/USD`). Numbers reproduced exactly as filed. Content drawn only
from that day's filings — never outside knowledge.

## API

- `GET /api/config` → `{"mockMode": bool}`
- `GET /api/day?date=YYYY-MM-DD` → the day object
- `POST /api/file` `{date, text, allowClosed?}` → 400 empty text, 409 closed
  day without the flag, else `{entry, count, status}`
- `POST /api/ask` `{date, query}` → `{answer, query, date, currency}`.
  `summary GBP` or bare `GBP` → that currency. `summary`/`all` → whole day.
  Anything else → a free question over the day's filings.
- `POST /api/eod` `{date}` → compiles, writes `reports/YYYY-MM-DD.md`,
  closes the day, commits (and pushes unless `EOD_PUSH=false`) →
  `{report, path, status, git, count}`. 409 if already closed or empty.

## Acceptance checks

1. A long note with accents, emoji and a pipe table files, survives a page
   reload byte-identical, and is readable UTF-8 on disk (not escaped).
2. A 20,000-character paste files without freezing or truncation.
3. Empty submit → quiet refusal (red border flash), no dialog.
4. Server down → error shown in the paste panel, text stays in the box.
5. Switching to yesterday and back swaps the entries list correctly.
6. Four filings in ninety seconds, keyboard only (⌘/Ctrl+Enter).
7. Closed day: red banner, confirm before filing, server 409 without
   `allowClosed`, a second EOD 409s.
8. `summary GBP` → four headers in order, numbers verbatim, no slashed
   pairs, copies as plain text. `summary CHF` with nothing filed → the
   single "Nothing filed for CHF on <date>." line, no headers.
9. Entries list shows exactly 3 collapsed rows, the rest via scroll.
10. `python3 test_tagging.py` passes all 28 cases.
11. All of the above with `MOCK_MODE=true` and no API key present.

## Out of scope (by design)

Calendars / next-day data previews, authentication, multi-user support,
persisting ask-console history, editing or deleting filed entries.
