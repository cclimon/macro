"""storage.py

Manages per-day draft storage (data/drafts/YYYY-MM-DD.json) and the
committed EOD reports (data/eod/YYYY-MM-DD.md).
"""
import json
from pathlib import Path
from datetime import date

ROOT = Path(__file__).parent
DRAFTS_DIR = ROOT / "data" / "drafts"
EOD_DIR = ROOT / "data" / "eod"

DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
EOD_DIR.mkdir(parents=True, exist_ok=True)


def today_str() -> str:
    return date.today().isoformat()


def draft_path(day: str) -> Path:
    return DRAFTS_DIR / f"{day}.json"


def eod_path(day: str) -> Path:
    return EOD_DIR / f"{day}.md"


def load_draft(day: str) -> list[dict]:
    p = draft_path(day)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return []


def save_draft(day: str, entries: list[dict]) -> None:
    draft_path(day).write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def append_entry(day: str, entry: dict) -> list[dict]:
    entries = load_draft(day)
    entries.append(entry)
    save_draft(day, entries)
    return entries


def save_eod_markdown(day: str, markdown_text: str) -> Path:
    p = eod_path(day)
    p.write_text(markdown_text, encoding="utf-8")
    return p


def is_eod_done(day: str) -> bool:
    return eod_path(day).exists()


def list_eod_days() -> list[str]:
    return sorted(p.stem for p in EOD_DIR.glob("*.md"))
