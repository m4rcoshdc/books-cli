"""What the user has read: one small JSON file, edited from the shelf page (`books site --serve`).

reading.json (project root, gitignored: it's personal) maps book id -> state:

  {"breakneck": {"status": "reading", "since": "2026-11-23"}}

Three states: unread (absent from the file), "reading" and "read". It's user
state, not derived data, so it lives outside data/ (which can be rebuilt).
Only the shelf page uses it; the agent tools (library.py) don't.
"""

import datetime
import json

from .config import READING_FILE

STATES = ("unread", "reading", "read")


def load() -> dict[str, dict]:
    if not READING_FILE.exists():
        return {}
    return json.loads(READING_FILE.read_text(encoding="utf-8"))


def status(book_id: str) -> str:
    return load().get(book_id, {}).get("status", "unread")


def mark(book_id: str, state: str) -> dict[str, dict]:
    """Set one book's state and return the whole map. Written via a temp file, so never half-written."""
    if state not in STATES:
        raise ValueError(f"status must be one of {', '.join(STATES)}")
    data = load()
    if state == "unread":
        data.pop(book_id, None)
    elif data.get(book_id, {}).get("status") != state:
        data[book_id] = {"status": state, "since": datetime.date.today().isoformat()}
    tmp = READING_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(sorted(data.items())), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(READING_FILE)
    return data
