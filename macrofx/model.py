"""model.py

The two model seams. Both are gated on MOCK_MODE and must flip together:
in mock mode neither needs an API key or the anthropic SDK; in real mode
both make one Claude call each.
"""
import os
import re

MOCK_MODE = os.environ.get("MOCK_MODE", "true").lower() != "false"
DEFAULT_MODEL = "claude-opus-4-8"


def _get_model(role_env: str) -> str:
    return os.environ.get(role_env) or os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)


def _client():
    """Lazy import so mock mode never needs the SDK installed."""
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "The 'anthropic' package is required in real mode. "
            "Install it with: pip install anthropic"
        ) from e
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Required in real mode (MOCK_MODE=false)."
        )
    return anthropic.Anthropic(api_key=api_key)


ANON_SYSTEM_PROMPT = """You anonymise financial market colour before it is stored.

Replace any NAMED SELL-SIDE BANK (e.g. a named investment bank or broker-dealer)
with exactly one of: US BANK, EURO BANK, ASIAN BANK — chosen by that bank's home
region. Do NOT anonymise: central banks, regulators, exchanges, corporates, or
named people. Leave everything else byte-for-byte identical: numbers, line
breaks, tables, accents, emoji, punctuation, formatting.

Output ONLY the transformed text. No preamble, no explanation, no markdown
fencing."""


def anonymise(text: str) -> str:
    """
    Mock: identity passthrough (nothing to anonymise without a model).
    Real: one Claude call per the system prompt above.
    """
    if MOCK_MODE:
        return text

    client = _client()
    model = _get_model("ANON_MODEL")
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=ANON_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    if getattr(resp, "stop_reason", None) == "refusal":
        raise RuntimeError("Model refused the anonymisation request.")
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


COMPOSE_SYSTEM_PROMPT = """You are a senior sell-side FX analyst producing a
structured summary from the day's filed market colour, for one currency only.

Use ONLY the filed entries given to you. Do not invent, assume, or bring in
outside knowledge. Preserve every number and level exactly as given.

Output EXACTLY four headers, in this order, asterisks literal, nothing else
around them:

*MACRO THESIS*
*FLOWS SEEN*
*ECONOMICS*
*PRICE ACTION*

Under each header write plain prose (never bullet points), roughly 4-5
sentences. Currency pairs are written without a slash (e.g. GBPUSD, not
GBP/USD). If a section has nothing relevant in the filed entries, say so
plainly in one short sentence rather than omitting the header."""

_SECTION_HEADERS = ["*MACRO THESIS*", "*FLOWS SEEN*", "*ECONOMICS*", "*PRICE ACTION*"]

_KEYWORDS = {
    "*FLOWS SEEN*": [
        "bought", "sold", "buying", "selling", "flow", "flows", "positioning",
        "position", "hedge", "hedging", "inflow", "outflow", "demand", "offer",
        "bid", "real money", "hedge fund", "hf ", "corporates", "corp ",
    ],
    "*ECONOMICS*": [
        "cpi", "inflation", "gdp", "rate decision", "hike", "cut", "policy rate",
        "unemployment", "payroll", "pmi", "retail sales", "central bank",
        "meeting", "data", "release", "forecast", "consensus", "expected",
    ],
    "*PRICE ACTION*": [
        "closed", "close", "trading", "traded", "level", "resistance", "support",
        "high", "low", "spot", "range", "bps", "bp)", "%", "pips",
    ],
}


def _split_sentences(text: str) -> list[str]:
    """Splits on sentence-ending punctuation while keeping pipe-table rows whole."""
    lines = text.split("\n")
    out = []
    for line in lines:
        if "|" in line:
            out.append(line.strip())
            continue
        # simple sentence splitter; avoids splitting on decimal points / bp figures
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z{])", line.strip())
        out.extend(p.strip() for p in parts if p.strip())
    return out


def _bucket(sentence: str) -> str:
    lower = sentence.lower()
    for header in ["*FLOWS SEEN*", "*ECONOMICS*", "*PRICE ACTION*"]:
        if any(kw in lower for kw in _KEYWORDS[header]):
            return header
    return "*MACRO THESIS*"


def _strip_markers(sentence: str) -> str:
    sentence = re.sub(r"\{[A-Za-z\^0-9]{1,3}\}", "", sentence)
    sentence = re.sub(r"^[\*\-\u2022]\s*", "", sentence)
    return sentence.strip()


def compose(currency: str, entries: list[dict], date: str) -> str:
    """
    Mock: deterministic stub — bucket sentences by keyword into the four
    sections, strip {XX} markers and bullets, dedupe, cap ~4-5 sentences
    per section, reproduce surviving text verbatim so numbers stay exact.
    Real: one Claude call enforcing the output contract.
    """
    if not entries:
        return f"Nothing filed for {currency} on {date}."

    if MOCK_MODE:
        buckets = {h: [] for h in _SECTION_HEADERS}
        seen = set()
        for e in entries:
            filed = e.get("filed", "")
            for sentence in _split_sentences(filed):
                cleaned = _strip_markers(sentence)
                if not cleaned or cleaned in seen:
                    continue
                seen.add(cleaned)
                buckets[_bucket(cleaned)].append(cleaned)

        lines = []
        for header in _SECTION_HEADERS:
            lines.append(header)
            content = buckets[header][:5]
            if content:
                lines.append(" ".join(content))
            else:
                lines.append(f"Nothing relevant filed for this section on {date}.")
            lines.append("")
        return "\n".join(lines).strip()

    client = _client()
    model = _get_model("SUMMARY_MODEL")
    filed_blob = "\n\n---\n\n".join(e.get("filed", "") for e in entries)
    user_msg = f"Currency: {currency}\nDate: {date}\n\nFiled entries:\n\n{filed_blob}"
    resp = client.messages.create(
        model=model,
        max_tokens=1500,
        system=COMPOSE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    if getattr(resp, "stop_reason", None) == "refusal":
        raise RuntimeError("Model refused the summary request.")
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
