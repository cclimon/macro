"""discover_fiscal_fields.py  --  Bloomberg fiscal-field discovery / sanity check.

CHECKPOINT TOOL (run before building the full puller).

For the 5 sovereign-fiscal metrics, across a handful of test countries, this:
  1. Searches Bloomberg field mnemonics  (//blp/apiflds  -> the "BFIELDS" path)
  2. Searches candidate security tickers  (//blp/instruments  -> security lookup)
  3. BDP-probes every candidate ticker to confirm it actually resolves to a value,
     and reports the as-of date / frequency Bloomberg attaches to it.
  4. Prints a country x metric coverage matrix so the mnemonics can be
     eyeballed and sanity-checked before anything gets hardcoded.

Nothing here is hardcoded as "the answer" -- it prints what Bloomberg returns so a
human can confirm conventions (they vary by country) before we scale the puller.

Run:  conda run -n FX_BBG --no-capture-output python data/discover_fiscal_fields.py
Requires a live Bloomberg Terminal (localhost:8194).
"""
from __future__ import annotations

import datetime as _dt
import blpapi

# ── Test universe ────────────────────────────────────────────────────────────
# (kept small on purpose -- this is the pre-scale sanity check)
COUNTRIES = [
    {"iso": "US", "name": "United States", "aliases": ["US", "United States"]},
    {"iso": "DE", "name": "Germany",       "aliases": ["Germany", "German"]},
    {"iso": "JP", "name": "Japan",         "aliases": ["Japan", "Japanese"]},
    {"iso": "BR", "name": "Brazil",        "aliases": ["Brazil", "Brazilian"]},
    {"iso": "ZA", "name": "South Africa",  "aliases": ["South Africa"]},
]

# ── The 5 metrics + the words we search Bloomberg with ───────────────────────
# metric key -> (human label, [field-search terms], [security-search term templates])
# "{c}" in a security template is filled with each country alias.
METRICS = {
    "fiscal_balance_pct_gdp": (
        "Fiscal balance % GDP (headline, general govt)",
        ["budget balance", "net lending", "fiscal balance", "government balance"],
        ["{c} budget balance % GDP",
         "{c} general government net lending",
         "{c} government budget balance GDP"],
    ),
    "primary_balance_pct_gdp": (
        "Primary balance % GDP (ex-interest)",
        ["primary balance", "primary budget"],
        ["{c} primary balance % GDP",
         "{c} general government primary balance"],
    ),
    "debt_pct_gdp": (
        "Gross general govt debt % GDP",
        ["government debt", "gross debt", "public debt", "debt to GDP"],
        ["{c} government debt % GDP",
         "{c} general government gross debt",
         "{c} gross debt GDP"],
    ),
    "interest_cost": (
        "Interest servicing cost (% GDP and % revenue)",
        ["interest payments", "interest expense", "debt service", "interest paid"],
        ["{c} government interest payments",
         "{c} government interest expense GDP",
         "{c} general government interest"],
    ),
    "nominal_gdp_growth": (
        "Nominal GDP growth (for r-g; actual + consensus fcst)",
        ["nominal GDP", "GDP nominal", "nominal gross domestic"],
        ["{c} nominal GDP",
         "{c} nominal GDP year",
         "{c} nominal GDP forecast"],
    ),
}

# Fields we ask for when probing a candidate ticker.  A candidate "resolves" if
# PX_LAST comes back numeric.  The rest give us the as-of date / frequency /
# description Bloomberg attaches, which is what we actually need to eyeball.
PROBE_FLOAT_FIELDS = ["PX_LAST"]
PROBE_META_FIELDS = [
    "NAME", "LONG_COMP_NAME", "SECURITY_DES", "COUNTRY", "INDX_FREQ",
    "LAST_UPDATE_DT", "ECO_RELEASE_DT", "LATEST_ANNOUNCEMENT_PERIOD",
]

MAX_SECURITY_RESULTS = 12
MAX_FIELD_RESULTS = 10


# ── blpapi plumbing ──────────────────────────────────────────────────────────
def open_session():
    opts = blpapi.SessionOptions()
    opts.setServerHost("localhost")
    opts.setServerPort(8194)
    s = blpapi.Session(opts)
    if not s.start():
        raise ConnectionError("Bloomberg session failed to start -- is the Terminal open/logged in?")
    for svc in ("//blp/refdata", "//blp/apiflds", "//blp/instruments"):
        if not s.openService(svc):
            raise ConnectionError(f"Could not open {svc}")
    return s


def _drain(session):
    """Yield every message of a request until the final RESPONSE event."""
    while True:
        ev = session.nextEvent(2000)
        for msg in ev:
            yield msg
        if ev.eventType() == blpapi.Event.RESPONSE:
            break


def search_fields(session, term: str):
    """//blp/apiflds FieldSearchRequest -> [(mnemonic, datatype, category, desc)]."""
    svc = session.getService("//blp/apiflds")
    req = svc.createRequest("FieldSearchRequest")
    req.set("searchSpec", term)
    session.sendRequest(req)
    out = []
    for msg in _drain(session):
        if not msg.hasElement("fieldData"):
            continue
        arr = msg.getElement("fieldData")
        for i in range(arr.numValues()):
            fi = arr.getValueAsElement(i)
            if not fi.hasElement("fieldInfo"):
                continue
            info = fi.getElement("fieldInfo")
            g = lambda k: info.getElementAsString(k) if info.hasElement(k) else ""
            out.append((g("mnemonic"), g("datatype"), g("categoryName"), g("description")))
    # de-dup, keep order
    seen, uniq = set(), []
    for r in out:
        if r[0] and r[0] not in seen:
            seen.add(r[0]); uniq.append(r)
    return uniq[:MAX_FIELD_RESULTS]


_YK = {
    "govt": "Govt", "corp": "Corp", "mtge": "Mtge", "muni": "Muni",
    "pfd": "Pfd", "equity": "Equity", "comdty": "Comdty", "index": "Index",
    "curncy": "Curncy", "mmkt": "M-Mkt",
}
import re as _re
_YK_RE = _re.compile(r"<([a-z\-]+)>\s*$", _re.IGNORECASE)


def normalize_ticker(sec: str) -> str:
    """Instrument search returns 'IGD%USA<index>'; BDP needs 'IGD%USA Index'."""
    m = _YK_RE.search(sec.strip())
    if not m:
        return sec.strip()
    root = sec[: m.start()].strip()
    yk = _YK.get(m.group(1).lower(), m.group(1).capitalize())
    return f"{root} {yk}"


def search_securities(session, query: str):
    """//blp/instruments instrumentListRequest -> [(normalized_security, description)]."""
    svc = session.getService("//blp/instruments")
    req = svc.createRequest("instrumentListRequest")
    req.set("query", query)
    req.set("maxResults", MAX_SECURITY_RESULTS)
    session.sendRequest(req)
    out = []
    for msg in _drain(session):
        if not msg.hasElement("results"):
            continue
        arr = msg.getElement("results")
        for i in range(arr.numValues()):
            r = arr.getValueAsElement(i)
            sec = r.getElementAsString("security") if r.hasElement("security") else ""
            des = r.getElementAsString("description") if r.hasElement("description") else ""
            if sec:
                out.append((normalize_ticker(sec), des))
    return out


def _fmt_val(el, name):
    """Extract a field value as float-or-string, tolerating datatype."""
    try:
        dt = el.getElement(name).datatype()
    except Exception:
        return None
    try:
        if dt in (blpapi.DataType.FLOAT64, blpapi.DataType.FLOAT32,
                  blpapi.DataType.INT32, blpapi.DataType.INT64):
            return float(el.getElementAsFloat(name))
        if dt == blpapi.DataType.DATE:
            return str(el.getElementAsDatetime(name))
        return el.getElementAsString(name)
    except Exception:
        try:
            return el.getElementAsString(name)
        except Exception:
            return None


def bdp_probe(session, tickers):
    """BDP a batch of tickers for float + meta fields.
    Returns {ticker: {field: value, '_ok': bool, '_error': str|None}}."""
    if not tickers:
        return {}
    svc = session.getService("//blp/refdata")
    req = svc.createRequest("ReferenceDataRequest")
    for t in tickers:
        req.getElement("securities").appendValue(t)
    for f in PROBE_FLOAT_FIELDS + PROBE_META_FIELDS:
        req.getElement("fields").appendValue(f)
    session.sendRequest(req)

    res = {t: {"_ok": False, "_error": None} for t in tickers}
    for msg in _drain(session):
        if not msg.hasElement("securityData"):
            continue
        sd = msg.getElement("securityData")
        for i in range(sd.numValues()):
            row = sd.getValueAsElement(i)
            tkr = row.getElementAsString("security")
            rec = res.setdefault(tkr, {"_ok": False, "_error": None})
            if row.hasElement("securityError"):
                err = row.getElement("securityError")
                rec["_error"] = err.getElementAsString("message") if err.hasElement("message") else "securityError"
                continue
            fd = row.getElement("fieldData")
            for f in PROBE_FLOAT_FIELDS + PROBE_META_FIELDS:
                if fd.hasElement(f):
                    rec[f] = _fmt_val(fd, f)
            rec["_ok"] = isinstance(rec.get("PX_LAST"), (int, float))
    return res


# ── Driver ───────────────────────────────────────────────────────────────────
REPORT: list[str] = []


def rec(line: str = "", echo: bool = False):
    """Append to the full report; optionally echo to stdout (stdout truncates)."""
    REPORT.append(line)
    if echo:
        print(line)


def main():
    import os
    header = [
        "=" * 96,
        "BLOOMBERG SOVEREIGN-FISCAL FIELD DISCOVERY  --  pre-scale sanity check",
        f"run: {_dt.datetime.now().isoformat(timespec='seconds')}",
        f"countries: {', '.join(c['iso'] for c in COUNTRIES)}",
        "=" * 96,
    ]
    for h in header:
        rec(h, echo=True)

    session = open_session()

    # 1) Field-mnemonic search -- documented, but expected to be a dead end.
    #    Fiscal aggregates on BBG are ECONOMIC SECURITIES (pull PX_LAST), not BDP
    #    reference fields on a country object. One term per metric to prove that.
    rec("\n\n########## (1) FIELD MNEMONIC SEARCH  (//blp/apiflds) -- expected noise ##########")
    for key, (label, field_terms, _sec) in METRICS.items():
        rec(f"\n----- {label}  [{key}] -----")
        term = field_terms[0]
        rows = search_fields(session, term)
        rec(f'  search "{term}"  ->  {len(rows)} field(s)')
        for mnem, dtype, cat, desc in rows[:6]:
            rec(f"      {mnem:<34} {dtype:<8} {desc[:56]}")

    # 2) Security search + 3) BDP probe, per country x metric -----------------
    rec("\n\n########## (2/3) SECURITY SEARCH + BDP PROBE  (//blp/instruments -> //blp/refdata) ##########")
    coverage = {}  # (iso, metric) -> list of resolved ticker dicts (best first)
    for country in COUNTRIES:
        iso, name = country["iso"], country["name"]
        rec(f"\n\n==================== {iso}  ({name}) ====================")
        for key, (label, _ft, sec_templates) in METRICS.items():
            rec(f"\n  ---- {label}  [{key}] ----")
            candidates = []
            for tmpl in sec_templates:
                for alias in country["aliases"]:
                    q = tmpl.format(c=alias)
                    for sec, des in search_securities(session, q):
                        candidates.append((sec, des, q))
            seen, uniq = set(), []
            for sec, des, q in candidates:
                if sec not in seen:
                    seen.add(sec); uniq.append((sec, des, q))
            if not uniq:
                rec("     (no security matches from instrument search)")
                coverage[(iso, key)] = []
                continue

            probe = bdp_probe(session, [s for s, _, _ in uniq])
            resolved = []
            for sec, des, q in uniq:
                r = probe.get(sec, {})
                ok = r.get("_ok")
                px = r.get("PX_LAST")
                asof = r.get("ECO_RELEASE_DT") or r.get("LATEST_ANNOUNCEMENT_PERIOD") or r.get("LAST_UPDATE_DT") or ""
                freq = r.get("INDX_FREQ") or ""
                flag = "OK " if ok else "-- "
                errtxt = "" if ok else f"  [{r.get('_error') or 'no PX_LAST'}]"
                rec(f"     {flag}{sec:<24} px={str(px):<11} asof={str(asof):<12} freq={str(freq):<9} | {des[:44]}{errtxt}")
                if ok:
                    resolved.append({"ticker": sec, "px_last": px, "asof": asof,
                                     "freq": freq, "desc": des})
            coverage[(iso, key)] = resolved

    # 4) Coverage matrix (echoed to stdout) -----------------------------------
    metric_keys = list(METRICS.keys())
    matrix = ["\n\n########## (4) COVERAGE MATRIX  (resolved = BDP returned numeric PX_LAST) ##########\n"]
    hdr = f'{"country":<16}' + "".join(f"{k[:15]:<17}" for k in metric_keys)
    matrix.append(hdr)
    matrix.append("-" * len(hdr))
    for country in COUNTRIES:
        iso = country["iso"]
        cells = []
        for k in metric_keys:
            res = coverage.get((iso, k)) or []
            cells.append((res[0]["ticker"][:15] if res else "EMPTY").ljust(17))
        matrix.append(f'{iso + " " + country["name"][:12]:<16}' + "".join(cells))
    matrix += [
        "",
        "Legend: ticker shown = BBG resolved a live numeric value (best candidate).",
        "        EMPTY = no instrument-search candidate returned a numeric PX_LAST",
        "        -> those cells route to the IMF WEO / Eurostat fallback in the puller.",
    ]
    # Also list every resolved ticker per country/metric (the actual deliverable)
    detail = ["\n\n########## (5) RESOLVED TICKERS PER COUNTRY x METRIC ##########"]
    for country in COUNTRIES:
        iso = country["iso"]
        detail.append(f"\n{iso} ({country['name']}):")
        for k in metric_keys:
            res = coverage.get((iso, k)) or []
            if res:
                toks = ", ".join(f'{r["ticker"]} (={r["px_last"]}, {r["asof"]})' for r in res[:4])
            else:
                toks = "EMPTY -> fallback"
            detail.append(f"    {k:<24}: {toks}")

    for block in (matrix, detail):
        for line in block:
            rec(line, echo=True)
    rec("=" * 96, echo=True)

    session.stop()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "discover_fiscal_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))
    print(f"\n[full detail written to] {out_path}")


if __name__ == "__main__":
    main()
