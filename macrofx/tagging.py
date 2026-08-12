"""tagging.py

Rule-based currency tagging for filed INFO text. No model calls — pure
string/regex logic, deterministic, testable.

Priority order (a match at an earlier stage does not prevent later stages
from also contributing tags; all matches across all stages are collected
and deduped, but the FIRST stage that produces any tag determines which
tag is considered "primary" for EOD grouping — specifically the first
tag discovered in document order is used as the primary/grouping tag).
"""
import re

# ---------------------------------------------------------------------------
# ISO currency universe this tool cares about (~34 codes)
# ---------------------------------------------------------------------------
ISO_CODES = {
    "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD",
    "CNY", "CNH", "KRW", "INR", "IDR", "BRL", "MXN", "ZAR",
    "TRY", "HUF", "PLN", "CZK", "SEK", "NOK", "THB", "TWD",
    "MYR", "PHP", "SGD", "HKD", "ILS", "RUB", "COP", "PEN",
    "UYU", "ARS", "CLP",
}

# ---------------------------------------------------------------------------
# {XX} bracket code -> currency map (Bloomberg-style + ISO mixed, 2-letter)
# ---------------------------------------------------------------------------
BRACKET_MAP = {
    "US": "USD", "GB": "GBP", "UK": "GBP", "JN": "JPY", "JP": "JPY",
    "EC": "EUR", "EU": "EUR", "GE": "EUR", "FR": "EUR", "IT": "EUR",
    "SP": "EUR", "AU": "AUD", "NZ": "NZD", "CA": "CAD",
    "SZ": "CHF", "CH": "CHF",  # NOTE: {CH} is ambiguous (Switzerland vs China)
    "CN": "CNY", "KO": "KRW", "KR": "KRW", "IN": "INR", "ID": "IDR",
    "BZ": "BRL", "BR": "BRL", "MX": "MXN", "SA": "ZAR",
    "TU": "TRY", "TR": "TRY", "HU": "HUF", "PO": "PLN", "PL": "PLN",
    "CZ": "CZK", "SW": "SEK", "SE": "SEK", "NO": "NOK",
    "TH": "THB", "TA": "TWD", "TW": "TWD", "MA": "MYR", "MY": "MYR",
    "PH": "PHP", "SI": "SGD", "SG": "SGD", "HK": "HKD",
    "IS": "ILS", "RU": "RUB",
    "PE": "PEN", "AR": "ARS", "UR": "UYU", "CL": "CLP", "CO": "COP",
}

BRACKET_RE = re.compile(r"\{([A-Za-z]{2})\}")

# ---------------------------------------------------------------------------
# Pair patterns
# ---------------------------------------------------------------------------
# Compact: EURUSD, eurusd (case-insensitive) — 6 letters, both legs valid ISO
COMPACT_PAIR_RE = re.compile(r"\b([A-Za-z]{3})([A-Za-z]{3})\b")
# Slash/hyphen: EUR/USD, EUR-USD
SLASHED_PAIR_RE = re.compile(r"\b([A-Za-z]{3})[/\-]([A-Za-z]{3})\b")

# Standalone ISO codes — UPPERCASE ONLY (avoids "try", "cad", "cop" as words)
STANDALONE_RE = re.compile(r"\b([A-Z]{3})\b")

# USD "noise" evidence words — needed for USD to tag when it's only a pair leg
USD_EVIDENCE_RE = re.compile(r"\bUSD\b|\bDXY\b|\bdollar\b|\bFed\b|\bSOFR\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Alias dictionary — case-insensitive, word-boundary
# ---------------------------------------------------------------------------
ALIASES = {
    "GBP": ["cable", "sterling", "gilts?", "SONIA", "BoE", "pound"],
    "EUR": ["euro", "ECB", "Bund", "Euribor", "eurozone"],
    "JPY": ["yen", "BoJ", "MoF", "Tokyo fix"],
    "USD": ["dollar", "greenback", "DXY", "Fed", "FOMC", "SOFR", "treasur(?:y|ies)"],
    "CHF": ["swiss franc", "swissy", "SNB"],
    "KRW": ["korea", "seoul", "BoK", "KOSPI"],
    "CNY": ["china", "yuan", "renminbi", "PBoC"],
    "AUD": ["aussie", "RBA"],
    "NZD": ["kiwi", "RBNZ"],
    "CAD": ["loonie"],
    "SEK": ["riksbank", "krona"],
    "NOK": ["norges bank"],
    "MXN": ["banxico"],
    "BRL": ["brazil"],
    "INR": ["RBI", "rupee"],
    "TRY": ["CBRT", "lira"],
    "ZAR": ["SARB", "rand"],
    "PLN": ["zloty"],
    "HUF": ["forint"],
    "CZK": ["koruna"],
    "SGD": ["MAS", "singapore"],
    "HKD": ["HKMA", "hong kong"],
    "THB": ["baht"],
    "MYR": ["ringgit"],
    "PHP": ["BSP"],
    "ILS": ["shekel"],
    "TWD": ["taiwan"],
    "IDR": ["indonesia"],
}

# Deliberately excluded (documented, not implemented): "won", "real", "peso", "MPC"

_ALIAS_PATTERNS = {
    ccy: re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)
    for ccy, words in ALIASES.items()
}

BI_RE = re.compile(r"\bBI\b")  # case-sensitive on purpose (Bank Indonesia)
BIWEEKLY_RE = re.compile(r"\bbi-weekly\b", re.IGNORECASE)

SYMBOL_MAP = {"£": "GBP", "€": "EUR", "¥": "JPY"}


def _is_shouty(text: str) -> bool:
    """True if >80% of letters are uppercase, with a minimum letter count.

    Threshold set to 10 (not 20) so that short all-caps trading-desk
    headlines like "TRY TO KEEP POWDER DRY" (18 letters) are correctly
    guarded against misfiring as a currency tag.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 10:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return (upper / len(letters)) > 0.8


def tag_text(text: str) -> list[str]:
    """
    Returns a deduped, order-preserving list of currency tags found in text.
    The first tag in the returned list is the "primary" tag for EOD grouping.
    """
    found: list[str] = []

    def add(ccy: str):
        if ccy and ccy not in found:
            found.append(ccy)

    # ---- Stage 1: explicit {XX} bracket codes — always win, always first ----
    for m in BRACKET_RE.finditer(text):
        code = m.group(1).upper()
        if code in BRACKET_MAP:
            add(BRACKET_MAP[code])
        else:
            add(code)  # unknown bracket code kept verbatim

    # ---- Stage 2: pairs (compact + slashed), both legs valid ISO ----
    pair_span_mask = [False] * len(text)

    def mask(span):
        for i in range(span[0], span[1]):
            if i < len(pair_span_mask):
                pair_span_mask[i] = True

    for m in SLASHED_PAIR_RE.finditer(text):
        a, b = m.group(1).upper(), m.group(2).upper()
        if a in ISO_CODES and b in ISO_CODES:
            add(a)
            add(b)
            mask(m.span())

    for m in COMPACT_PAIR_RE.finditer(text):
        if pair_span_mask[m.start()] if m.start() < len(pair_span_mask) else False:
            continue
        a, b = m.group(1).upper(), m.group(2).upper()
        if a in ISO_CODES and b in ISO_CODES:
            add(a)
            add(b)
            mask(m.span())

    # ---- Stage 3: standalone ISO codes, uppercase only ----
    shouty = _is_shouty(text)
    word_like_traps = {"TRY", "COP", "PEN", "PHP"}
    for m in STANDALONE_RE.finditer(text):
        if pair_span_mask[m.start()] if m.start() < len(pair_span_mask) else False:
            continue
        code = m.group(1)
        if code not in ISO_CODES:
            continue
        if shouty and code in word_like_traps:
            continue  # guard against ALL-CAPS headlines misfiring
        if code == "USD":
            # USD noise rule: needs standalone evidence, or being the only ccy found
            if not USD_EVIDENCE_RE.search(text):
                continue
        add(code)

    # ---- Stage 4: alias dictionary ----
    for ccy, pattern in _ALIAS_PATTERNS.items():
        if pattern.search(text):
            add(ccy)

    # ---- Stage 5: BI (case-sensitive), excluding "bi-weekly" ----
    text_no_biweekly = BIWEEKLY_RE.sub("", text)
    if BI_RE.search(text_no_biweekly):
        add("IDR")

    # ---- Stage 6: currency symbols ($ deliberately NOT evidence) ----
    for sym, ccy in SYMBOL_MAP.items():
        if sym in text:
            add(ccy)

    # ---- Stage 7: normalise CNH -> CNY ----
    if "CNH" in found:
        idx = found.index("CNH")
        found.pop(idx)
        add("CNY")

    return found
