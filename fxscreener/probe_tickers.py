"""
Ticker probe.

For every role that failed or came back stale, test a list of candidate
conventions and report which resolve with a usable history. Run this instead of
guessing; it costs one Bloomberg call per candidate and settles the map.

    python probe_tickers.py                 # all roles
    python probe_tickers.py --role swap2y   # one role
    python probe_tickers.py --out probe_results.csv

Output columns: role, ccy, candidate, status, obs, last_date, last_value.
Pick the winner per currency and paste it into config.py.
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

import bbg

START = "2023-01-01"
MIN_OBS_DAILY = 500       # a daily series should have ~940 over this window


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------
# These are CANDIDATES, not assertions. The LIBOR-referenced *SW2 roots are
# dead or dying; the RFR/OIS replacements below are the conventions to test.
# Where a swap market is thin (PEN, TWD, THB) the generic government bond is
# included as a fallback.

SWAP2Y_CANDIDATES: dict[str, list[str]] = {
    # --- confirmed dead or stale, need replacing
    "USD": ["USOSFR2 Curncy", "USSOC Curncy", "USSA2 Curncy", "USSW2 Curncy"],
    "GBP": ["BPSWS2 Curncy", "BPSWSC Curncy", "BPSO2 Curncy", "GTGBP2Y Govt"],
    "CHF": ["SFSNT2 Curncy", "SFSOC Curncy", "SFSWO2 Curncy", "GTCHF2Y Govt"],
    "JPY": ["JYSO2 Curncy", "JYSOC Curncy", "JYSWO2 Curncy", "GTJPY2Y Govt"],
    "CAD": ["CDSO2 Curncy", "CDSWO2 Curncy", "CDOR02 Index", "GTCAD2Y Govt"],
    # --- outright failures
    "BRL": ["BCSWPF2 Curncy", "BCSWNI2 Curncy", "PREDI360 Index", "GTBRL2Y Govt",
            "ODF2 Comdty"],
    "IDR": ["IHSWNI2 Curncy", "IHSWN2 Curncy", "IDSWNI2 Curncy", "GTIDR2Y Govt"],
    "PEN": ["PSSW2 Curncy", "PNSW2 Curncy", "GTPEN2Y Govt", "PERU2Y Index"],
    "THB": ["THSW2 Curncy", "TBSW2 Curncy", "TBSO2 Curncy", "GTTHB2Y Govt"],
    "TRY": ["TYSO2 Curncy", "TRSW2 Curncy", "TYSW2 Curncy", "GTTRY2Y Govt"],
    "TWD": ["NTSW2 Curncy", "TWSW2 Curncy", "NTSWO2 Curncy", "GTTWD2Y Govt"],
    # --- working, probed only to confirm the RFR version is not better
    "EUR": ["EESWE2 Curncy", "EUSA2 Curncy"],
    "AUD": ["ADSWAP2 Curncy", "ADSO2 Curncy", "ADSW2 Curncy"],
    "MXN": ["MPSW2C Curncy", "MPSWF2 Curncy", "MPSW2B Curncy"],
    "ILS": ["ISSW2 Curncy", "ISSO2 Curncy", "GTILS2Y Govt"],
}

# NDF forward points. The BGN outright convention worked for COP, KRW, CNH but
# failed for these six. The classic NDF roots are a 2-letter code plus N.
FWD_CANDIDATES: dict[str, list[str]] = {
    "BRL": ["BCN3M BGN Curncy", "BCN3M Curncy", "USDBRL3M NDF Curncy", "BRL3M Curncy"],
    "CLP": ["CHN3M BGN Curncy", "CHN3M Curncy", "USDCLP3M NDF Curncy", "CLP3M Curncy"],
    "IDR": ["IHN3M BGN Curncy", "IHN3M Curncy", "USDIDR3M NDF Curncy", "IDR3M Curncy"],
    "INR": ["IRN3M BGN Curncy", "IRN3M Curncy", "USDINR3M NDF Curncy", "INR3M Curncy"],
    "PEN": ["PSN3M BGN Curncy", "PSN3M Curncy", "USDPEN3M NDF Curncy", "PEN3M Curncy"],
    "TWD": ["NTN3M BGN Curncy", "NTN3M Curncy", "USDTWD3M NDF Curncy", "TWD3M Curncy"],
}

CPI_CANDIDATES: dict[str, list[str]] = {
    "ILS": ["ISCPIYOY Index", "ISCPIYY Index", "ISCPYOY Index", "ISCPIYYP Index"],
}

# BIS broad real effective exchange rate. Pattern is BISB + 3-letter code + R.
REER_CANDIDATES: dict[str, list[str]] = {
    ccy: [f"BISB{code}R Index", f"BIS{code}R Index", f"{code}REER Index",
          f"CTOT{code} Index"]
    for ccy, code in {
        "EUR": "EUR", "JPY": "JPY", "GBP": "GBP", "CHF": "CHF", "CAD": "CAD",
        "AUD": "AUD", "NZD": "NZD", "NOK": "NOK", "SEK": "SEK",
        "BRL": "BRL", "MXN": "MXN", "CLP": "CLP", "COP": "COP", "PEN": "PEN",
        "PLN": "PLN", "HUF": "HUF", "CZK": "CZK", "ZAR": "ZAR", "TRY": "TRY",
        "ILS": "ILS", "KRW": "KRW", "CNH": "CNY", "TWD": "TWD", "INR": "INR",
        "IDR": "IDR", "THB": "THB",
    }.items()
}

# 5y USD senior sovereign CDS. Common Bloomberg forms.
CDS_CANDIDATES: dict[str, list[str]] = {
    ccy: [f"C{code}1U5 CBIL Curncy", f"C{code}1U5 Curncy", f"C{code}1U5 CBIN Curncy"]
    for ccy, code in {
        "BRL": "BRA", "MXN": "MEX", "CLP": "CHI", "COP": "COL", "PEN": "PER",
        "PLN": "POL", "HUF": "HUN", "CZK": "CZE", "ZAR": "SOAF", "TRY": "TURKEY",
        "ILS": "ISRAEL", "KRW": "KOREA", "CNH": "CHINA", "INR": "INDIA",
        "IDR": "INDON", "THB": "THAI",
    }.items()
}

# CFTC non-commercial net positions. Several conventions coexist on the
# terminal; probe rather than assume.
COT_CANDIDATES: dict[str, list[str]] = {
    "EUR": ["IMM0EURN Index", "CFTCNCNE Index", "IMM1EURN Index"],
    "JPY": ["IMM0JPYN Index", "CFTCNCNJ Index", "IMM1JPYN Index"],
    "GBP": ["IMM0GBPN Index", "CFTCNCNB Index", "IMM1GBPN Index"],
    "CHF": ["IMM0CHFN Index", "CFTCNCNS Index", "IMM1CHFN Index"],
    "CAD": ["IMM0CADN Index", "CFTCNCNC Index", "IMM1CADN Index"],
    "AUD": ["IMM0AUDN Index", "CFTCNCNA Index", "IMM1AUDN Index"],
    "NZD": ["IMM0NZDN Index", "CFTCNCNN Index", "IMM1NZDN Index"],
    "MXN": ["IMM0MXNN Index", "CFTCNCNM Index", "IMM1MXNN Index"],
    "BRL": ["IMM0BRLN Index", "CFTCNCNR Index", "IMM1BRLN Index"],
    "ZAR": ["IMM0ZARN Index", "CFTCNCNZ Index", "IMM1ZARN Index"],
}

# CDS roots are now confirmed in config.py; only the suffix is open, handled
# by probe_cds_suffix(). Forward tickers are confirmed too.
CANDIDATE_SETS = {
    "swap2y": SWAP2Y_CANDIDATES,
    "reer": REER_CANDIDATES,
    "cot": COT_CANDIDATES,
}


def probe(role: str, candidates: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for ccy, cands in candidates.items():
        won = False
        for tk in cands:
            s = bbg.history(tk, start=START, use_cache=False)
            ok = not s.empty
            rows.append({
                "role": role, "ccy": ccy, "candidate": tk,
                "status": "OK" if ok else "FAIL",
                "obs": len(s),
                "last_date": s.index[-1].date() if ok else None,
                "last_value": round(float(s.iloc[-1]), 4) if ok else None,
                "pick": "",
            })
            mark = "OK  " if ok else "fail"
            print(f"  {role:<11} {ccy:<4} {mark} {tk:<26} obs={len(s)}", flush=True)
            # Take the first candidate with a usable history and stop probing.
            if ok and (len(s) >= MIN_OBS_DAILY or role in ("cpi_yoy", "cot", "reer")):
                rows[-1]["pick"] = "<== USE"
                won = True
                break
        if not won:
            print(f"  {role:<11} {ccy:<4} NO CANDIDATE RESOLVED", flush=True)
    return pd.DataFrame(rows)


def probe_cds_suffix() -> None:
    """Settle the CDS source/yellow-key suffix against confirmed roots.

    The roots are confirmed; only the tail is uncertain. Test a few roots
    against each candidate suffix and take the one that resolves broadly —
    testing several roots rather than one guards against a single sovereign
    being illiquid on an otherwise correct suffix.
    """
    # CIGB1U5 (India) is included deliberately: the root is confirmed but it
    # did not resolve on the suffix that works for fifteen other sovereigns,
    # so either India uses a different tail or it simply is not quoted. This
    # probe distinguishes the two.
    roots = ["CBRZ1U5", "CMEX1U5", "CSOAF1U5", "CPOLD1U5", "CKREA1U5", "CIGB1U5"]
    suffixes = ["CBIL Curncy", "Curncy", "CBIN Curncy", "CMAN Curncy",
                "CBGN Curncy", "Corp"]
    print("\n=== CDS SUFFIX PROBE ===")
    best, best_n = None, -1
    for suf in suffixes:
        hits, sample = 0, None
        for r in roots:
            s = bbg.history(f"{r} {suf}", start=START, use_cache=False)
            if not s.empty:
                hits += 1
                sample = sample or (r, len(s), round(float(s.iloc[-1]), 1))
        print(f"  {suf:<14} {hits}/{len(roots)} resolved"
              + (f"   e.g. {sample[0]} obs={sample[1]} last={sample[2]}" if sample else ""))
        if hits > best_n:
            best, best_n = suf, hits
    if best_n > 0:
        print(f"\n  ==> set CDS_SUFFIX = \"{best}\" in config.py")
        print("      (check the sample level looks like a SPREAD in bp, not a price)")
    else:
        print("\n  ==> nothing resolved; the roots may need a different yellow key")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default=None,
                    choices=list(CANDIDATE_SETS) + ["cds_suffix"])
    ap.add_argument("--out", default="probe_results.csv")
    a = ap.parse_args()

    if not bbg.preflight():
        raise SystemExit("Bloomberg unreachable.")

    if a.role == "cds_suffix":
        probe_cds_suffix()
        raise SystemExit(0)

    sets = {a.role: CANDIDATE_SETS[a.role]} if a.role else CANDIDATE_SETS
    frames = [probe(r, c) for r, c in sets.items()]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(a.out, index=False)

    print("\n=== WINNERS ===")
    w = df[df["pick"] != ""]
    print(w[["role", "ccy", "candidate", "obs", "last_value"]].to_string(index=False))
    missing = sorted(set(zip(df.role, df.ccy)) - set(zip(w.role, w.ccy)))
    if missing:
        print("\n=== NOTHING RESOLVED (needs a manual look on the terminal) ===")
        for r, c in missing:
            print(f"  {r:<11} {c}")
    print(f"\nFull probe -> {a.out}")
