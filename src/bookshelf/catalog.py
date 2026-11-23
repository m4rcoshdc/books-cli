"""Phase 0: read Calibre's metadata.db (read-only) into data/catalog.json.

Each book gets a short, stable `id` (a slug of its title). Once assigned, an id
is kept across runs (matched by Calibre uuid) so later layers can rely on it.
"""

import json
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

from .config import CATALOG_FILE, LIBRARY, OVERRIDES_FILE

# Calibre stores "unknown date" as year 101.
_UNKNOWN_DATE_YEAR = 101


def slugify(text: str, max_words: int = 6) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.split(r":| - | — |_ ", text)[0]  # drop subtitles
    words = re.findall(r"[a-z0-9]+", text.lower().replace("'", ""))
    return "-".join(words[:max_words]) or "untitled"


def _connect(library: Path) -> sqlite3.Connection:
    uri = (library / "metadata.db").resolve().as_uri() + "?mode=ro"
    db = sqlite3.connect(uri, uri=True)
    db.row_factory = sqlite3.Row
    return db


def _list(db, sql: str, book_id: int) -> list[str]:
    return [r[0] for r in db.execute(sql, (book_id,))]


def _date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return None if dt.year <= _UNKNOWN_DATE_YEAR else dt.date().isoformat()


def _clean_author(name: str) -> str:
    # Calibre stores commas inside author names as "|"; some imports also
    # append a bracketed duplicate: "Horowitz, Ben [Horowitz, Ben]".
    name = re.sub(r"\s*\[[^\]]*\]?$", "", name.replace("|", ","))
    if name.count(",") == 1:  # "Last, First" -> "First Last"
        last, first = (p.strip() for p in name.split(","))
        name = f"{first} {last}"
    return name.strip()


def _html_to_text(html: str | None) -> str | None:
    if not html:
        return None
    return " ".join(BeautifulSoup(html, "html.parser").get_text(" ").split()) or None


def read_calibre(library: Path = LIBRARY, previous: list[dict] | None = None) -> list[dict]:
    known_ids = {b["uuid"]: b["id"] for b in previous or []}
    used: set[str] = set(known_ids.values())
    db = _connect(library)
    books = []
    for row in db.execute("SELECT * FROM books ORDER BY id"):
        bid = row["id"]
        authors = [_clean_author(a) for a in _list(
            db,
            "SELECT a.name FROM books_authors_link l JOIN authors a ON a.id = l.author "
            "WHERE l.book = ? ORDER BY l.id",
            bid,
        )]
        tags = _list(
            db,
            "SELECT t.name FROM books_tags_link l JOIN tags t ON t.id = l.tag WHERE l.book = ?",
            bid,
        )
        languages = _list(
            db,
            "SELECT g.lang_code FROM books_languages_link l JOIN languages g ON g.id = l.lang_code "
            "WHERE l.book = ? ORDER BY l.item_order",
            bid,
        )
        publisher = _list(
            db,
            "SELECT p.name FROM books_publishers_link l JOIN publishers p ON p.id = l.publisher "
            "WHERE l.book = ?",
            bid,
        )
        series = _list(
            db,
            "SELECT s.name FROM books_series_link l JOIN series s ON s.id = l.series WHERE l.book = ?",
            bid,
        )
        identifiers = {
            r["type"]: r["val"]
            for r in db.execute("SELECT type, val FROM identifiers WHERE book = ?", (bid,))
        }
        comment = db.execute("SELECT text FROM comments WHERE book = ?", (bid,)).fetchone()

        book_dir = library / row["path"]
        formats = {}
        for f in db.execute("SELECT format, name FROM data WHERE book = ?", (bid,)):
            file = book_dir / f"{f['name']}.{f['format'].lower()}"
            if file.exists():
                formats[f["format"]] = file.relative_to(library).as_posix()
        cover = book_dir / "cover.jpg"

        uuid = row["uuid"]
        book_id = known_ids.get(uuid)
        if book_id is None:
            base = book_id = slugify(row["title"])
            n = 2
            while book_id in used:
                book_id, n = f"{base}-{n}", n + 1
            used.add(book_id)

        books.append(
            {
                "id": book_id,
                "calibre_id": bid,
                "uuid": uuid,
                "title": row["title"],
                "authors": authors,
                "series": series[0] if series else None,
                "series_index": row["series_index"] if series else None,
                "tags": tags,
                "languages": languages,
                "publisher": publisher[0] if publisher else None,
                "pubdate": _date(row["pubdate"]),
                "added": _date(row["timestamp"]),
                "identifiers": identifiers,
                "description": _html_to_text(comment["text"] if comment else None),
                "formats": formats,
                "cover": cover.relative_to(library).as_posix() if cover.exists() else None,
            }
        )
    db.close()
    return books


def load_catalog() -> list[dict]:
    if not CATALOG_FILE.exists():
        return []
    return json.loads(CATALOG_FILE.read_text(encoding="utf-8"))["books"]


def apply_overrides(books: list[dict]) -> list[dict]:
    """Layer hand-curated fixes (metadata_overrides.json) over the Calibre data."""
    if not OVERRIDES_FILE.exists():
        return books
    overrides = json.loads(OVERRIDES_FILE.read_text(encoding="utf-8"))
    known = {b["id"] for b in books}
    unknown = [k for k in overrides if not k.startswith("_") and k not in known]
    if unknown:  # a typo, or a fork still carrying someone else's fixes: say so, don't fail
        import sys

        print(f"metadata_overrides.json: ignoring ids not in this library: {', '.join(unknown)}", file=sys.stderr)
    for b in books:
        fix = overrides.get(b["id"], {})
        b.update(fix)
        b["overridden"] = sorted(fix)
    return books


def build_catalog() -> list[dict]:
    books = apply_overrides(read_calibre(previous=load_catalog()))
    CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "library": str(LIBRARY),
        "books": books,
    }
    CATALOG_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return books


def find_book(book_id: str) -> dict:
    for b in load_catalog():
        if b["id"] == book_id or str(b["calibre_id"]) == book_id:
            return b
    raise KeyError(book_id)
