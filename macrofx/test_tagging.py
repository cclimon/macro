"""test_tagging.py — run with: python3 test_tagging.py"""
from tagging import tag_text

CASES = [
    # (input, expected_tags, description)
    ("the bulls won the session", [], "won as verb must not tag NOK/anything"),
    ("TRY TO KEEP POWDER DRY", [], "shouty headline, TRY is a word not a tag"),
    ("EUR/USD", ["EUR", "USD"], "slashed pair tags both legs"),
    ("USDJPY heavy, dollar offered", ["JPY", "USD"], "compact pair + dollar alias for USD"),
    ("{BZ} Brazil", ["BRL"], "bracket code maps to currency"),
    ("{GB} GDP data due", ["GBP"], "bracket code GB -> GBP"),
    ("{JN} JPY article", ["JPY"], "bracket + standalone same currency dedupes"),
    ("Cable is bid this morning", ["GBP"], "cable alias"),
    ("The BoJ kept rates unchanged", ["JPY"], "BoJ alias"),
    ("real money was a buyer", [], "real excluded as alias trap"),
    ("MXN peso weakened", ["MXN"], "peso alone excluded but MXN standalone still tags"),
    ("MPC decision awaited", [], "MPC excluded as generic alias"),
    ("BI held rates steady", ["IDR"], "BI case-sensitive match -> IDR"),
    ("bi-weekly CPI due Thursday", [], "bi-weekly must not fire BI rule"),
    ("bi held rates steady", [], "lowercase bi must not match (case-sensitive)"),
    ("£ higher against the dollar", ["GBP", "USD"], "GBP symbol + dollar alias"),
    ("USDCNH held in a tight range", ["USD", "CNY"], "CNH normalises to CNY"),
    ("EURCHF broke higher", ["EUR", "CHF"], "compact pair both legs valid"),
    ("Sterling extends gains, gilts rally", ["GBP"], "two GBP aliases dedupe to one tag"),
    ("USD index steady", ["USD"], "USD standalone with DXY/dollar-style evidence via 'USD'"),
    ("just USD alone with no other evidence word", ["USD"], "bare USD counts as its own evidence"),
    ("COP started slightly weaker amid outflows", ["COP"], "COP standalone tags when not shouty"),
    ("PESO CRASHES AS COP HITS RECORD LOW", [], "shouty guard suppresses COP as standalone"),
    ("PBOC injected liquidity, yuan steady", ["CNY"], "PBoC + yuan alias, single CNY tag"),
    ("KOSPI rallied on Samsung earnings", ["KRW"], "KOSPI alias -> KRW"),
    ("{SA} ZAR view", ["ZAR"], "SA bracket maps to ZAR"),
    ("{CH} CNH update", ["CHF", "CNY"], "ambiguous CH bracket (CHF) plus CNH text (normalises to CNY) both tag"),
    ("no currency content here at all", [], "no false positives on generic text"),
]


def run():
    passed = 0
    failed = 0
    for text, expected, desc in CASES:
        result = tag_text(text)
        # order-insensitive comparison since we only guarantee membership + a primary
        ok = set(result) == set(expected)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"[{status}] {desc}")
        if not ok:
            print(f"       input:    {text!r}")
            print(f"       expected: {expected}")
            print(f"       got:      {result}")
    print(f"\n{passed} passed, {failed} failed out of {len(CASES)}")
    return failed == 0


if __name__ == "__main__":
    import sys
    ok = run()
    sys.exit(0 if ok else 1)
