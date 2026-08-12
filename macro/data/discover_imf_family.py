"""discover_imf_family.py -- map the IMF WEO / Fiscal Monitor family ON Bloomberg.

The IMF Data Mapper REST API is WAF-blocked from this environment (403), but
Bloomberg mirrors the same IMF WEO / Fiscal Monitor series as economic securities
(e.g. IGS%USA Index = "IMF US General Government Gross Debt % of GDP").

This finds the ticker PREFIX per fiscal concept and confirms the '<PREFIX>%<ISO3>'
pattern, so the puller can generate IMF-basis tickers for the whole universe.

Run:  conda run -n FX_BBG --no-capture-output python data/discover_imf_family.py
"""
from __future__ import annotations
import datetime as _dt
import re as _re
import blpapi

TEST = [("USA", "United States"), ("DEU", "Germany"), ("JPN", "Japan"),
        ("BRA", "Brazil"), ("ZAF", "South Africa")]

# concept -> instrument-search queries (we only keep hits whose description
# starts with "IMF" and mentions general government / the concept)
CONCEPTS = {
    "overall_balance": ["IMF {n} general government net lending", "IMF {n} net lending borrowing"],
    "primary_balance": ["IMF {n} general government primary", "IMF {n} primary net lending"],
    "gross_debt":      ["IMF {n} general government gross debt", "IMF {n} gross debt"],
    "revenue":         ["IMF {n} general government revenue", "IMF {n} government revenue"],
    "expenditure":     ["IMF {n} general government total expenditure", "IMF {n} government expenditure"],
    "interest":        ["IMF {n} general government interest", "IMF {n} interest expense"],
    "nominal_gdp":     ["IMF {n} gross domestic product current prices", "IMF {n} nominal GDP"],
}

_YK = {"index": "Index", "curncy": "Curncy", "govt": "Govt", "comdty": "Comdty",
       "equity": "Equity", "corp": "Corp", "mtge": "Mtge"}
_YK_RE = _re.compile(r"<([a-z\-]+)>\s*$", _re.I)


def norm(sec):
    m = _YK_RE.search(sec.strip())
    if not m:
        return sec.strip()
    return f"{sec[:m.start()].strip()} {_YK.get(m.group(1).lower(), m.group(1).capitalize())}"


def open_session():
    o = blpapi.SessionOptions(); o.setServerHost("localhost"); o.setServerPort(8194)
    s = blpapi.Session(o)
    if not s.start():
        raise ConnectionError("no terminal")
    for svc in ("//blp/refdata", "//blp/instruments"):
        s.openService(svc)
    return s


def drain(s):
    while True:
        ev = s.nextEvent(3000)
        for m in ev:
            yield m
        if ev.eventType() == blpapi.Event.RESPONSE:
            break


def isearch(s, q, n=15):
    svc = s.getService("//blp/instruments")
    r = svc.createRequest("instrumentListRequest"); r.set("query", q); r.set("maxResults", n)
    s.sendRequest(r)
    out = []
    for m in drain(s):
        if m.hasElement("results"):
            arr = m.getElement("results")
            for i in range(arr.numValues()):
                e = arr.getValueAsElement(i)
                sec = e.getElementAsString("security") if e.hasElement("security") else ""
                des = e.getElementAsString("description") if e.hasElement("description") else ""
                if sec:
                    out.append((norm(sec), des))
    return out


def bdp(s, tickers, fields):
    if not tickers:
        return {}
    svc = s.getService("//blp/refdata")
    r = svc.createRequest("ReferenceDataRequest")
    for t in tickers:
        r.getElement("securities").appendValue(t)
    for f in fields:
        r.getElement("fields").appendValue(f)
    s.sendRequest(r)
    res = {t: {} for t in tickers}
    for m in drain(s):
        if not m.hasElement("securityData"):
            continue
        sd = m.getElement("securityData")
        for i in range(sd.numValues()):
            row = sd.getValueAsElement(i); t = row.getElementAsString("security")
            if row.hasElement("securityError"):
                continue
            fd = row.getElement("fieldData")
            for f in fields:
                if fd.hasElement(f):
                    try:
                        res[t][f] = fd.getElementAsFloat(f)
                    except Exception:
                        try:
                            res[t][f] = str(fd.getElementAsDatetime(f))
                        except Exception:
                            res[t][f] = fd.getElementAsString(f)
    return res


def main():
    s = open_session()
    print("IMF-on-Bloomberg family discovery @", _dt.datetime.now().isoformat(timespec="seconds"))
    # prefix -> {iso3: (ticker, value, desc)}   to reveal the <PREFIX>%<ISO3> pattern
    prefix_map = {}
    for iso3, name in TEST:
        print(f"\n================ {iso3} ({name}) ================")
        for concept, queries in CONCEPTS.items():
            hits = []
            for q in queries:
                hits += isearch(s, q.format(n=name))
            # keep IMF general-government series only, de-dup
            seen, imf = set(), []
            for sec, des in hits:
                if sec in seen:
                    continue
                seen.add(sec)
                if des.upper().startswith("IMF"):
                    imf.append((sec, des))
            probe = bdp(s, [t for t, _ in imf], ["PX_LAST", "INDX_FREQ", "LAST_UPDATE_DT"])
            print(f"  -- {concept} --")
            for sec, des in imf[:8]:
                v = probe.get(sec, {})
                px = v.get("PX_LAST")
                if px is None:
                    continue
                print(f"     {sec:<18} = {str(px):<10} freq={str(v.get('INDX_FREQ','')):<10} | {des[:52]}")
                # record prefix (token before '%')
                mm = _re.match(r"^([A-Za-z0-9]+)%([A-Z]{3})\b", sec)
                if mm:
                    pref = mm.group(1)
                    prefix_map.setdefault((concept, pref), {})[iso3] = (sec, px, des[:46])
    # summarise candidate <PREFIX>%<ISO3> patterns that resolved for >=3 countries
    print("\n\n================ CANDIDATE  <PREFIX>%<ISO3>  PATTERNS ================")
    for (concept, pref), d in sorted(prefix_map.items()):
        if len(d) >= 2:
            iso_list = ",".join(sorted(d.keys()))
            example = next(iter(d.values()))
            print(f"  {concept:<16} {pref+'%<ISO3>':<12} covers[{iso_list}]  e.g. {example[0]}={example[1]}  ({example[2]})")
    s.stop()


if __name__ == "__main__":
    main()
