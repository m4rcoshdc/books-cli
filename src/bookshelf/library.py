"""Phase 5: the library as an agent sees it.

One set of read-only functions. The CLI, the warm server (`books serve`) and,
later, an MCP server are thin layers over these, so every interface gives the
same answers.

The tools zoom in, like a researcher in a library:

  books()            the shelf: every book with its thesis and what it is good for
  book(id)           one book: card plus the chapter list with one-line summaries
  chapter(id, sec)   one chapter: summary, key points with chunk ids, concepts, data, figures
  search(question)   passages anywhere, each labelled with the chapter it sits in
  read(chunk_id)     the exact text (plus neighbours) and a ready-made citation
  related(concept)   where an idea appears across books (concept map)
  figure(ref)        what a chart, table or diagram shows, as data

Cards and figure descriptions were written by Claude Haiku (Phase 4): use them
to navigate. Quotes and claims must come from read(), which returns the book's
own words. Every function returns plain dicts (JSON-ready).
"""

import functools
import json
import re
import socket
import urllib.request

from . import embed as emb
from . import index as ix
from .catalog import load_catalog
from .concepts import load as load_concepts
from .config import BOOKS_DIR
from .enrich import load_results
from .extract import load_manifest

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765
CARD_NOTE = "Summaries, key points and figure descriptions are machine-written (Claude Haiku). " \
            "Navigate with them; quote only text returned by read()."


class NotFound(LookupError):
    """A book, section, chunk or figure that does not exist. The message says what does."""


# ---------------------------------------------------------------- lookups (cached per process)


@functools.cache
def _catalog() -> dict[str, dict]:
    return {b["id"]: b for b in load_catalog() if load_manifest(b["id"])}


def resolve_book(ref: str) -> str:
    """Exact id, Calibre number, or an unambiguous part of the id, title or an author's name."""
    books = _catalog()
    if ref in books:
        return ref
    ref_l = ref.lower().strip()
    for b in books.values():
        if str(b["calibre_id"]) == ref_l:
            return b["id"]
    found = [b["id"] for b in books.values()
             if ref_l in b["id"] or ref_l in b["title"].lower() or any(ref_l in a.lower() for a in b["authors"])]
    if len(found) == 1:
        return found[0]
    if found:
        raise NotFound(f"'{ref}' matches several books: {', '.join(found)}")
    raise NotFound(f"No book '{ref}'. Run `books library` for the list.")


@functools.cache
def _manifest(book_id: str) -> dict:
    return load_manifest(book_id)


@functools.cache
def _book_card(book_id: str) -> dict | None:
    r = load_results(book_id, "book").get("book")
    return r["card"] if r else None


@functools.cache
def _chapter_cards(book_id: str) -> dict[int, dict]:
    """section -> card. Long sections were carded in parts (007p1, 007p2): merged here."""
    merged: dict[int, dict] = {}
    for key, r in sorted(load_results(book_id, "chapter").items()):
        sec = int(r["meta"]["section"])
        card = r["card"]
        if sec not in merged:
            merged[sec] = {**card, "summary": [card["summary"]]}
            continue
        m = merged[sec]
        m["summary"].append(card["summary"])
        for field in ("key_points", "concepts", "entities", "data_points", "questions"):
            m[field] = m[field] + card[field]
    for m in merged.values():
        m["summary"] = " ".join(m["summary"])
        seen = set()
        m["concepts"] = [c for c in m["concepts"] if not (c["name"].lower() in seen or seen.add(c["name"].lower()))]
    return merged


@functools.cache
def _figure_cards(book_id: str) -> dict[str, dict]:
    return {k: r["card"] for k, r in load_results(book_id, "figure").items()}


@functools.cache
def _figures_meta(book_id: str) -> dict[str, dict]:
    f = BOOKS_DIR / book_id / "figures.json"
    return {x["id"]: x for x in json.loads(f.read_text(encoding="utf-8"))} if f.exists() else {}


@functools.cache
def _chunk_counts() -> dict[tuple[str, int], int]:
    db = ix._connect(ix.INDEX_FILE, vectors=False)
    rows = db.execute("SELECT book, section, COUNT(*) FROM chunks GROUP BY book, section").fetchall()
    db.close()
    return {(b, s): n for b, s, n in rows}


def _first_sentence(text: str, max_words: int = 30) -> str:
    s = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    words = s.split()
    return s if len(words) <= max_words else " ".join(words[:max_words]) + " …"


def _one_lines(book_id: str) -> dict[int, str]:
    """Chapter one-liners: the book card's, else the first sentence of the chapter card."""
    out = {s: _first_sentence(c["summary"]) for s, c in _chapter_cards(book_id).items()}
    card = _book_card(book_id)
    for s in (card or {}).get("structure", []):
        if s.get("one_line"):
            out[s["section"]] = s["one_line"]
    return out


def _section(book_id: str, n: int) -> dict:
    for s in _manifest(book_id)["sections"]:
        if s["n"] == n:
            return s
    raise NotFound(f"{book_id} has no section {n}. Run `books book {book_id}` for its chapters.")


def _surname(book_id: str) -> str:
    authors = _catalog()[book_id]["authors"]
    if not authors:
        return ""
    name = authors[0].split()[-1]
    return name + (" et al." if len(authors) > 1 else "")


def _pages(pages) -> str:
    if not pages:
        return ""
    a, b = pages
    return f"p. {a}" if a == b else f"pp. {a}-{b}"


def citation(book_id: str, path: list[str], pages, chunk_id: str) -> str:
    b = _catalog()[book_id]
    parts = [p for p in (_surname(book_id), f"*{b['title']}*", f"“{path[-1]}”" if path else "", _pages(pages)) if p]
    return ", ".join(parts) + f" [{chunk_id}]"


# ---------------------------------------------------------------- tools


def books() -> dict:
    out = []
    for b in _catalog().values():
        m = _manifest(b["id"])
        card = _book_card(b["id"]) or {}
        out.append({
            "id": b["id"], "title": b["title"], "authors": b["authors"],
            "language": (b["languages"] or ["?"])[0], "words": m["stats"]["words"],
            "chapters": sum(1 for s in m["sections"] if s["kind"] == "body"),
            "thesis": card.get("thesis", ""), "use_for": card.get("use_for", []),
        })
    out.sort(key=lambda x: x["title"].lower())
    return {"books": out, "note": CARD_NOTE}


def book(ref: str) -> dict:
    book_id = resolve_book(ref)
    b, m, card = _catalog()[book_id], _manifest(book_id), _book_card(book_id) or {}
    one = _one_lines(book_id)
    counts = _chunk_counts()
    chapters = [{
        "section": s["n"], "title": " > ".join(s["path"]), "depth": s["depth"], "kind": s["kind"],
        "words": s["words"], "chunks": counts.get((book_id, s["n"]), 0), "one_line": one.get(s["n"], ""),
    } for s in m["sections"] if s["kind"] != "front"]
    return {
        "id": book_id, "title": b["title"], "authors": b["authors"], "language": (b["languages"] or ["?"])[0],
        "words": m["stats"]["words"],
        "summary": card.get("summary", ""), "thesis": card.get("thesis", ""),
        "top_concepts": card.get("top_concepts", []), "use_for": card.get("use_for", []),
        "chapters": chapters, "note": CARD_NOTE,
    }


def chapter(ref: str, section: int | None = None) -> dict:
    """ref is BOOK:SECTION (e.g. thinking-in-systems:8), or a book with section given separately."""
    if section is None:
        if ":" not in ref:
            raise NotFound("Give a section: BOOK:SECTION, e.g. thinking-in-systems:8")
        ref, sec = ref.split(":", 1)
        section = int(sec.split(":")[0])
    book_id = resolve_book(ref)
    s = _section(book_id, int(section))
    card = _chapter_cards(book_id).get(s["n"])
    db = ix._connect(ix.INDEX_FILE, vectors=False)
    rows = db.execute("SELECT id, pages, figures FROM chunks WHERE book = ? AND section = ? ORDER BY seq",
                      (book_id, s["n"])).fetchall()
    db.close()
    chunk_ids = [r["id"] for r in rows]
    pages = [p for r in rows if (p := json.loads(r["pages"]))]
    fig_ids = [f for r in rows for f in json.loads(r["figures"])]
    fcards = _figure_cards(book_id)
    figures = [{"ref": f"{book_id}:{f}", "type": fcards[f]["type"], "title": fcards[f]["title"]}
               for f in fig_ids if f in fcards]
    numbers = [x["n"] for x in _manifest(book_id)["sections"] if x["kind"] != "front"]
    i = numbers.index(s["n"]) if s["n"] in numbers else -1
    out = {
        "book": book_id, "book_title": _catalog()[book_id]["title"], "section": s["n"],
        "title": " > ".join(s["path"]), "kind": s["kind"], "words": s["words"],
        "pages": [pages[0][0], pages[-1][1]] if pages else None,
        "chunks": chunk_ids, "figures": figures,
        "prev": f"{book_id}:{numbers[i - 1]}" if i > 0 else None,
        "next": f"{book_id}:{numbers[i + 1]}" if 0 <= i < len(numbers) - 1 else None,
        "card": None, "note": CARD_NOTE,
    }
    if card:
        out["card"] = {k: card[k] for k in ("summary", "content_type", "key_points", "concepts",
                                            "entities", "data_points", "questions")}
    elif s["words"] < 250 and chunk_ids:
        out["hint"] = f"Short section with no card: read it directly with `books read {chunk_ids[0]}`."
    return out


def _model_ready() -> bool:
    return emb._model is not None


def server_up(timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=timeout):
            return True
    except OSError:
        return False


def _remote(tool: str, args: dict) -> dict | None:
    """Call the warm server (`books serve`) if it is running, else None."""
    if not server_up():
        return None
    req = urllib.request.Request(f"http://{SERVER_HOST}:{SERVER_PORT}/tool/{tool}",
                                 data=json.dumps(args).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read().decode("utf-8"))
    if "error" in out:
        raise NotFound(out["error"])
    return out


# Max hits per book when searching the whole library (0 = no cap). Phase 6 evals: a cap of 3
# only cost answers on single-target questions (hit@10 0.86 -> 0.91 without it, nothing broke),
# so it's off by default. Agents pass per_book=2 for comparative, cross-book questions.
PER_BOOK = 0


def search(query: str, k: int = 8, book: str | None = None, mode: str = "hybrid", notes: bool = False,
           per_book: int = PER_BOOK) -> dict:
    """Passages that answer a question. Hybrid (keyword + meaning) when the model is
    loaded here or in `books serve`; keyword-only otherwise, with a note saying so.

    per_book caps hits per book (whole-library searches only), so one book can't crowd
    out the others on a comparative question; the next-best passages from other books move up."""
    if mode not in ("hybrid", "keyword", "semantic"):
        raise NotFound(f"Unknown mode '{mode}': use hybrid, keyword or semantic.")
    book_id = resolve_book(book) if book else None
    args = {"query": query, "k": k, "book": book_id, "mode": mode, "notes": notes, "per_book": per_book}
    note = None
    if mode != "keyword" and not _model_ready():
        remote = _remote("search", args)
        if remote is not None:
            remote["via"] = "server"
            return remote
        if mode == "hybrid":
            mode = "keyword"
            note = ("Keyword search only: meaning search needs the warm server. Start it with `books serve` "
                    "(about a minute to load). Rephrasing with the book's own words helps meanwhile.")
    cap = per_book if (per_book and not book_id) else 0
    hits = ix.search(query, mode, k * 4 if cap else k, book_id, ("body", "back") if notes else ("body",))
    if cap:
        taken: dict[str, int] = {}
        kept = []
        for h in hits:
            if taken.get(h.book, 0) < cap:
                taken[h.book] = taken.get(h.book, 0) + 1
                kept.append(h)
        hits = kept[:k]
    one = {}
    out = []
    for h in hits:
        if h.book not in one:
            one[h.book] = _one_lines(h.book)
        sec = int(h.id.split(":")[1])
        out.append({
            "id": h.id, "book": h.book, "title": h.title, "chapter": " > ".join(h.path),
            "chapter_ref": f"{h.book}:{sec}", "chapter_gist": one[h.book].get(sec, ""),
            "pages": h.pages, "snippet": h.snippet, "found_by": h.ranks,
        })
    return {"query": query, "mode": mode, "results": out, "note": note, "via": "local"}


_IMG_LINK = re.compile(r"!\[(fig-\d+)[^\]]*\]\([^)]*\)")


def read(chunk_id: str, before: int = 0, after: int = 0) -> dict:
    """Exact text of a passage, plus `before`/`after` neighbouring passages in the same book."""
    db = ix._connect(ix.INDEX_FILE, vectors=False)
    row = db.execute("SELECT rowid, book FROM chunks WHERE id = ?", (chunk_id.strip(),)).fetchone()
    if not row:
        db.close()
        raise NotFound(f"No passage '{chunk_id}'. Ids look like thinking-in-systems:008:001 (book:section:seq).")
    before, after = max(0, min(before, 5)), max(0, min(after, 5))
    rows = db.execute("SELECT * FROM chunks WHERE book = ? AND rowid BETWEEN ? AND ? ORDER BY rowid",
                      (row["book"], row["rowid"] - before, row["rowid"] + after)).fetchall()
    prev_id = db.execute("SELECT id FROM chunks WHERE book = ? AND rowid = ?", (row["book"], rows[0]["rowid"] - 1)).fetchone()
    next_id = db.execute("SELECT id FROM chunks WHERE book = ? AND rowid = ?", (row["book"], rows[-1]["rowid"] + 1)).fetchone()
    db.close()
    passages = []
    for r in rows:
        path, pages = json.loads(r["path"]), json.loads(r["pages"])
        text = _IMG_LINK.sub(lambda m: f"[image {m.group(1)}]", r["text"])
        passages.append({
            "id": r["id"], "chapter": " > ".join(path), "pages": pages, "words": r["words"],
            "figures": [f"{r['book']}:{f}" for f in json.loads(r["figures"])],
            "citation": citation(r["book"], path, pages, r["id"]), "text": text,
        })
    note = None
    if any(p["figures"] for p in passages):
        note = "Blockquotes starting “[fig-NNN · type]” are machine-written figure descriptions, not the author's words."
    return {"book": row["book"], "book_title": _catalog()[row["book"]]["title"], "passages": passages,
            "prev": prev_id["id"] if prev_id else None, "next": next_id["id"] if next_id else None, "note": note}


def related(query: str, limit: int = 10, min_books: int = 1) -> dict:
    """Concepts whose name or aliases contain the query (all words), most widely shared first."""
    words = query.lower().split()
    sec_titles: dict[str, dict[int, str]] = {}
    out = []
    for c in load_concepts():
        names = " | ".join([c["concept"], *c["aliases"]])
        if c["books"] < min_books or not all(w in names for w in words):
            continue
        where = []
        for book_id, secs in c["where"].items():
            if book_id not in sec_titles:
                sec_titles[book_id] = {s["n"]: " > ".join(s["path"]) for s in _manifest(book_id)["sections"]}
            where.append({"book": book_id, "title": _catalog()[book_id]["title"],
                          "chapters": [{"ref": f"{book_id}:{s}", "title": sec_titles[book_id].get(s, "")}
                                       for s in secs]})
        out.append({"concept": c["concept"], "aliases": c["aliases"], "books": c["books"],
                    "mentions": c["mentions"], "roles": c["roles"], "where": where})
    exact = [c for c in out if c["concept"] == query.lower()]
    rest = [c for c in out if c["concept"] != query.lower()]
    found = (exact + rest)[:limit]
    note = None if found else "No concept by that name. Try a shorter or more general term, or `books search`."
    return {"query": query, "concepts": found, "note": note}


def figure(ref: str) -> dict:
    """ref is BOOK:fig-NNN (as listed by chapter() and read())."""
    if ":" not in ref:
        raise NotFound("Give a figure as BOOK:fig-NNN, e.g. thinking-in-systems:fig-036")
    book_ref, fig = ref.rsplit(":", 1)
    book_id = resolve_book(book_ref)
    meta = _figures_meta(book_id).get(fig)
    if not meta:
        raise NotFound(f"{book_id} has no figure '{fig}'.")
    card = _figure_cards(book_id).get(fig)
    db = ix._connect(ix.INDEX_FILE, vectors=False)
    row = db.execute("SELECT id FROM chunks WHERE book = ? AND figures LIKE ?", (book_id, f'%"{fig}"%')).fetchone()
    db.close()
    s = _section(book_id, meta["section"])
    return {
        "ref": f"{book_id}:{fig}", "book_title": _catalog()[book_id]["title"], "chapter": " > ".join(s["path"]),
        "chapter_ref": f"{book_id}:{s['n']}", "in_passage": row["id"] if row else None,
        "caption": meta.get("caption"), "image": str(BOOKS_DIR / book_id / "figures" / meta["file"]),
        "description": card, "note": "The description is machine-written (Claude Haiku) from the image." if card
        else "Not described (decorative or not sent for description). Look at the image file.",
    }


TOOLS = {"books": books, "book": book, "chapter": chapter, "search": search,
         "read": read, "related": related, "figure": figure}
