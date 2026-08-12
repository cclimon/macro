"""claude_client.py

Thin wrapper around the Anthropic Messages API. Loads the versioned system
prompt from config/system_prompt.md so behavior rules live in the repo, not
hardcoded in Python.
"""
import os
import json
import requests
from pathlib import Path

ROOT = Path(__file__).parent
SYSTEM_PROMPT_PATH = ROOT / "config" / "system_prompt.md"
CURRENCY_CODES_PATH = ROOT / "config" / "currency_codes.json"

API_URL = "https://api.anthropic.com/v1/messages"


def _load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _load_currency_codes() -> dict:
    if CURRENCY_CODES_PATH.exists():
        return json.loads(CURRENCY_CODES_PATH.read_text(encoding="utf-8"))
    return {}


def _headers() -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in.")
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def _call_claude(user_message: str, extra_context: str = "", max_tokens: int = 2000) -> str:
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    system = _load_system_prompt()
    codes = _load_currency_codes()
    system += "\n\n## Current known currency/country tags\n" + json.dumps(codes, indent=2)
    if extra_context:
        system += "\n\n## Today's already-filed INFO (context for this request)\n" + extra_context

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_message}],
    }
    resp = requests.post(API_URL, headers=_headers(), json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")


def file_info(raw_text: str, existing_draft: list[dict]) -> dict:
    """
    Sends raw INFO text to Claude for anonymization/translation/tagging.
    Returns a dict entry to append to the day's draft.
    Does NOT produce a structured summary (per filing rules).
    """
    context = json.dumps(existing_draft[-10:], indent=2) if existing_draft else ""
    prompt = (
        "The following text is tagged INFO and must be filed, not answered.\n"
        "Apply anonymization, translation, and tagging rules from the system prompt.\n"
        "Return ONLY the cleaned/anonymized/translated text, ready to store. "
        "Do not add commentary, headers, or a structured summary.\n\n"
        f"INFO:\n{raw_text}"
    )
    cleaned = _call_claude(prompt, extra_context=context)
    return {"raw": raw_text, "filed": cleaned}


def answer_question(question: str, todays_draft: list[dict]) -> str:
    """
    Answers a question/summary request using only today's filed INFO,
    structured under the four required headers.
    """
    context = json.dumps(todays_draft, indent=2)
    prompt = f"Question/request: {question}\n\nAnswer strictly from today's filed INFO above."
    return _call_claude(prompt, extra_context=context, max_tokens=1500)


def compile_eod(date_str: str, todays_draft: list[dict]) -> str:
    """
    Compiles the full day's INFO into a compressed, by-currency Markdown report.
    """
    context = json.dumps(todays_draft, indent=2)
    prompt = (
        f"Compile the EOD report for {date_str}. Structure it by currency, each with "
        "MACRO THESIS / FLOWS SEEN / ECONOMICS / PRICE ACTION. Compress narrative language "
        "but preserve every number, level, and price action point precisely. "
        "Output as clean Markdown starting with a top-level heading with the date."
    )
    return _call_claude(prompt, extra_context=context, max_tokens=3000)
