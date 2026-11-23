"""Phase 2: chapter Markdown -> citable passages (chunks).

Rules, in order of importance:
  1. Never cross a chapter (section) boundary: a chunk always cites one chapter.
  2. Never split a figure from its caption, or an equation from the sentence
     that introduces it ("...given by:" $$...$$ "where x is...").
  3. Prefer to start chunks at headings; otherwise pack whole paragraphs up to
     TARGET words; only split inside a paragraph when it alone exceeds MAX.
  4. Each chunk carries `context_before`: the last sentence(s) of the previous
     chunk. It is used for search, but kept out of `text` so citations stay exact.

Output per book: data/books/<id>/chunks.jsonl (+ chunks.meta.json).
Front matter, indexes and bibliographies are not chunked (search noise).
"""

import hashlib
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .config import BOOKS_DIR
from .figures import CAPTION_RE

CHUNKER_VERSION = 2  # bump whenever chunking logic changes
TARGET_WORDS = 400  # ~530 tokens in English
MAX_WORDS = 600
MIN_WORDS = 120
OVERLAP_WORDS = 50

_SKIP_TITLES = re.compile(
    r"^(index|[ií]ndice remissivo|bibliograph|bibliografia|references|works cited|"
    r"photo(graph)? credits|credits|permissions)", re.I
)
_PAGE_RE = re.compile(r"^<!-- p\.(\d+) -->$")
_FIG_RE = re.compile(r"!\[(fig-\d+)")
_SENTENCE_RE = re.compile(r"(?<=[.!?…»”\"])\s+(?=[A-ZÀ-Ý0-9“\"(¿¡—–-])")
_FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.S)


@dataclass
class Unit:
    """Smallest piece that is never split: a paragraph, heading, table, or a
    figure with its caption, or an equation glued to its sentence."""

    text: str
    start: int
    end: int
    kind: str  # text | heading | figure | math | table
    pages: set[int] = field(default_factory=set)
    level: int = 0  # heading level

    @property
    def words(self) -> int:
        return len(self.text.split())


def _blocks(body: str):
    """Yield (start, end, text) for blank-line separated blocks."""
    pos = 0
    for m in re.finditer(r"\n[ \t]*\n+", body):
        if body[pos : m.start()].strip():
            yield pos, m.start(), body[pos : m.start()].strip()
        pos = m.end()
    if body[pos:].strip():
        yield pos, len(body), body[pos:].strip()


def load_figure_notes(book_id: str) -> dict[str, str]:
    """fig-id -> Markdown description from Phase 4 (enrich/figures.jsonl), if any."""
    f = BOOKS_DIR / book_id / "enrich" / "figures.jsonl"
    notes: dict[str, str] = {}
    if not f.exists():
        return notes
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        r = json.loads(line)
        c = r["card"]
        if c["type"] == "decorative":
            continue
        lines = [f"> **[{r['key']} · {c['type']}] {c['title']}**", f"> {c['summary']}"]
        if c["data"].strip():
            lines += [">", *[f"> {ln}" if ln.strip() else ">" for ln in c["data"].strip().splitlines()]]
        if c["text_in_image"].strip():
            lines += [">", f"> Labels: {' '.join(c['text_in_image'].split())}"]
        notes[r["key"]] = "\n".join(lines)
    return notes


def _units(body: str, figure_notes: dict[str, str] | None = None) -> list[Unit]:
    units: list[Unit] = []
    figure_notes = figure_notes or {}
    page = None
    caption_budget = 0  # blocks after a figure that may still be its caption
    for start, end, text in _blocks(body):
        pm = _PAGE_RE.match(text)
        if pm:
            page = int(pm.group(1))
            continue
        pages = {page} if page else set()
        prev = units[-1] if units else None

        if text.startswith("#"):
            level = len(text) - len(text.lstrip("#"))
            units.append(Unit(text, start, end, "heading", pages, level))
            caption_budget = 0
        elif text.startswith("![fig-"):
            # Phase 4 description travels with its figure (the original text is untouched).
            fid = _FIG_RE.match(text)
            note = figure_notes.get(fid.group(1)) if fid else None
            units.append(Unit(text + ("\n\n" + note if note else ""), start, end, "figure", pages))
            caption_budget = 3
        elif caption_budget and prev and prev.kind == "figure" and (
            CAPTION_RE.match(text) or len(text.split()) <= 40
        ):
            prev.text += "\n\n" + text
            prev.end, prev.pages = end, prev.pages | pages
            caption_budget -= 1
        elif text.startswith("$$") or (
            prev and prev.kind in ("text", "math") and prev.text.rstrip().endswith("$$")
            and text[:1].islower()
        ):
            # Display math joins the sentence before it; a lowercase sentence
            # right after an equation ("where x is ...") joins the equation.
            if prev and prev.kind in ("text", "math"):
                prev.text += "\n\n" + text
                prev.end, prev.pages, prev.kind = end, prev.pages | pages, "math"
            else:
                units.append(Unit(text, start, end, "math", pages))
            caption_budget = 0
        elif text.startswith("|"):
            units.append(Unit(text, start, end, "table", pages))
            caption_budget = 0
        else:
            units.append(Unit(text, start, end, "text", pages))
            caption_budget = 0
    return units


def _split_long(unit: Unit) -> list[Unit]:
    """Split a unit longer than MAX_WORDS: tables by rows (header repeated),
    text by sentences. Offsets of the pieces are approximate (the unit's)."""
    if unit.words <= MAX_WORDS:
        return [unit]
    if unit.kind == "table":
        lines = unit.text.splitlines()
        head, rows = lines[:2], lines[2:]
        pieces, cur = [], []
        for row in rows:
            if cur and len(" ".join(head + cur + [row]).split()) > TARGET_WORDS:
                pieces.append("\n".join(head + cur))
                cur = []
            cur.append(row)
        pieces.append("\n".join(head + cur))
    else:
        pieces = _pack(_SENTENCE_RE.split(unit.text), " ")
        if any(len(p.split()) > MAX_WORDS for p in pieces):
            # No sentence endings (lists, verse, OCR'd indexes): split by lines.
            pieces = _pack(unit.text.splitlines(), "\n")
    return [Unit(p, unit.start, unit.end, unit.kind, set(unit.pages)) for p in pieces if p.strip()]


def _pack(parts: list[str], sep: str) -> list[str]:
    pieces, cur = [], []
    for s in parts:
        if cur and len(sep.join(cur + [s]).split()) > TARGET_WORDS:
            pieces.append(sep.join(cur))
            cur = []
        cur.append(s)
    pieces.append(sep.join(cur))
    return pieces


_TITLE_NOISE = {"chapter", "capitulo", "capítulo", "part", "parte", "book", "livro"}


def _only_title(text: str, section_path: list[str]) -> bool:
    """True when a chunk holds nothing but its chapter title (chapter-opener
    pages, empty dividers): such chunks only add noise to search."""
    if _FIG_RE.search(text):
        return False
    words = set(re.findall(r"[^\W\d_]{2,}", text.lower())) - _TITLE_NOISE
    title_words = set(re.findall(r"[^\W\d_]{2,}", " ".join(section_path).lower()))
    roman = {w for w in words if re.fullmatch(r"[ivxlc]+", w)}
    return len(words - title_words - roman) < 3


def _tail(text: str, max_words: int = OVERLAP_WORDS) -> str:
    """Last whole sentences of a chunk, up to max_words."""
    plain = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text).strip()
    sentences = _SENTENCE_RE.split(plain)
    out: list[str] = []
    for s in reversed(sentences):
        if out and len(" ".join([s] + out).split()) > max_words:
            break
        out.insert(0, s)
    tail = " ".join(out)
    words = tail.split()
    return " ".join(words[-max_words:]) if len(words) > max_words else tail


def chunk_section(body: str, section_path: list[str], figure_notes: dict[str, str] | None = None) -> list[dict]:
    units = [p for u in _units(body, figure_notes) for p in _split_long(u)]
    groups: list[tuple[list[Unit], list[str]]] = []
    cur: list[Unit] = []
    heads: list[tuple[int, str]] = []  # in-section heading stack
    cur_heads: list[str] = []

    def words(us):
        return sum(u.words for u in us)

    def flush():
        nonlocal cur
        if cur:
            groups.append((cur, cur_heads))
        cur = []

    for u in units:
        if u.kind == "heading":
            title = " ".join(u.text.lstrip("#").split())
            if words(cur) >= MIN_WORDS:
                flush()
            if not (section_path and title.lower() == section_path[-1].lower()):
                heads = [h for h in heads if h[0] < u.level] + [(u.level, title)]
            if not cur:
                cur_heads = [h[1] for h in heads]
            cur.append(u)
            continue
        size = words(cur)
        if cur and ((size + u.words > TARGET_WORDS and size >= MIN_WORDS) or size + u.words > MAX_WORDS):
            flush()
        if not cur:
            cur_heads = [h[1] for h in heads]
        cur.append(u)
    flush()

    # A short last chunk joins the previous one when the result stays reasonable.
    if len(groups) > 1 and words(groups[-1][0]) < MIN_WORDS:
        last, prev = groups.pop(), groups[-1]
        if words(prev[0]) + words(last[0]) <= MAX_WORDS * 1.2:
            groups[-1] = (prev[0] + last[0], prev[1])
        else:
            groups.append(last)

    chunks = []
    previous_text = ""
    for us, heads_at_start in groups:
        text = "\n\n".join(u.text for u in us)
        if _only_title(text, section_path + heads_at_start):
            continue
        pages = sorted(set().union(*(u.pages for u in us)))
        chunks.append({
            "path": section_path + heads_at_start,
            "text": text,
            "context_before": _tail(previous_text) if previous_text else "",
            "start": us[0].start,
            "end": us[-1].end,
            "words": len(text.split()),
            "pages": [pages[0], pages[-1]] if pages else None,
            "figures": _FIG_RE.findall(text),
            "has_math": "$" in text and bool(re.search(r"\$[^$\s][^$]*\$", text)),
        })
        previous_text = text
    return chunks


# ---------------------------------------------------------------- per book


def _meta_path(book_id: str) -> Path:
    return BOOKS_DIR / book_id / "chunks.meta.json"


def load_chunks(book_id: str) -> list[dict]:
    f = BOOKS_DIR / book_id / "chunks.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line]


def chunk_book(book_id: str, force: bool = False) -> dict:
    out_dir = BOOKS_DIR / book_id
    manifest = json.loads((out_dir / "book.json").read_text(encoding="utf-8"))
    meta_file = _meta_path(book_id)
    notes = load_figure_notes(book_id)
    notes_hash = hashlib.sha1(json.dumps(notes, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    if not force and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if (meta["chunker_version"] == CHUNKER_VERSION and meta["extracted_at"] == manifest["extracted_at"]
                and meta.get("figure_notes") == notes_hash):
            return {**meta, "status": "skipped"}

    rows = []
    skipped_sections = []
    for s in manifest["sections"]:
        if s["kind"] == "front" or _SKIP_TITLES.match(s["title"].strip()):
            skipped_sections.append(s["n"])
            continue
        raw = (out_dir / s["file"]).read_text(encoding="utf-8")
        fm = _FRONT_MATTER_RE.match(raw)
        offset = fm.end() if fm else 0
        for seq, c in enumerate(chunk_section(raw[offset:], s["path"], notes), 1):
            rows.append({
                "id": f"{book_id}:{s['n']:03d}:{seq:03d}",
                "book": book_id,
                "section": s["n"],
                "seq": seq,
                "kind": s["kind"],
                "file": s["file"],
                **{k: c[k] for k in ("path", "words", "pages", "figures", "has_math")},
                "char_range": [c["start"] + offset, c["end"] + offset],
                "context_before": c["context_before"],
                "text": c["text"],
            })

    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sizes = [r["words"] for r in rows] or [0]
    meta = {
        "book": book_id,
        "chunker_version": CHUNKER_VERSION,
        "params": {"target": TARGET_WORDS, "max": MAX_WORDS, "min": MIN_WORDS, "overlap": OVERLAP_WORDS},
        "extracted_at": manifest["extracted_at"],
        "figure_notes": notes_hash,
        "described_figures": len(notes),
        "chunks": len(rows),
        "words": {"median": int(statistics.median(sizes)), "min": min(sizes), "max": max(sizes)},
        "sections_skipped": skipped_sections,
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {**meta, "status": "chunked"}


def find_chunk(chunk_id: str) -> dict:
    book_id = chunk_id.split(":")[0]
    for c in load_chunks(book_id):
        if c["id"] == chunk_id:
            return c
    raise KeyError(chunk_id)
