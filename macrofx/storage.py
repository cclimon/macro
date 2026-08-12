"""storage.py

One JSON file per trading day: data/YYYY-MM-DD.json
{"date", "status": "open"|"closed", "entries": [{"seq", "ts", "tags", "raw", "filed"}]}

Every filing re-reads the file, appends, writes atomically (temp file +
os.replace) so a crash mid-write never loses the day.
"""
import json
import os
import re
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def is_valid_date(date_str: str) -> bool:
    if not DATE_RE.match(date_str or ""):
        return False
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _day_path(date_str: str) -> str:
    # date_str is validated by callers; still guard against traversal
    safe = os.path.basename(date_str)
    return os.path.join(DATA_DIR, f"{safe}.json")


def load_day(date_str: str) -> dict:
    path = _day_path(date_str)
    if not os.path.exists(path):
        return {"date": date_str, "status": "open", "entries": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write(path: str, obj: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def save_day(date_str: str, day_obj: dict) -> None:
    _atomic_write(_day_path(date_str), day_obj)


def append_entry(date_str: str, raw: str, filed: str, tags: list[str]) -> dict:
    """Re-reads, appends, writes atomically. Returns the new entry."""
    day = load_day(date_str)
    seq = len(day["entries"]) + 1
    entry = {
        "seq": seq,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "tags": tags,
        "raw": raw,
        "filed": filed,
    }
    day["entries"].append(entry)
    save_day(date_str, day)
    return entry


def close_day(date_str: str) -> None:
    day = load_day(date_str)
    day["status"] = "closed"
    save_day(date_str, day)


def entries_for_currency(day_obj: dict, currency: str) -> list[dict]:
    ccy = currency.upper()
    return [e for e in day_obj["entries"] if ccy in [t.upper() for t in e.get("tags", [])]]


def group_by_primary_tag(day_obj: dict) -> dict:
    """Groups entries by their first (primary) tag; untagged -> GENERAL."""
    groups: dict[str, list[dict]] = {}
    for e in day_obj["entries"]:
        tags = e.get("tags", [])
        key = tags[0].upper() if tags else "GENERAL"
        groups.setdefault(key, []).append(e)
    return groups


def report_path(date_str: str) -> str:
    safe = os.path.basename(date_str)
    return os.path.join(REPORTS_DIR, f"{safe}.md")


def write_report(date_str: str, markdown_text: str) -> str:
    path = report_path(date_str)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    return path
