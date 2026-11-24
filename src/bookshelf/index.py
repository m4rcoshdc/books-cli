"""Phase 3: one SQLite file (data/index.sqlite) for search.

Tables
  books       id, title, authors, language
  chunks      every chunk with its citation data and original text
  chunks_fts  FTS5 keyword index (BM25): text, chapter path, context, book title
  chunk_vec   sqlite-vec table: one 1024-dim bge-m3 vector per chunk

Search modes
  keyword   BM25 over the words of the query (any word may match; rarer words weigh more)
  semantic  cosine similarity between the query's vector and each chunk's vector
  hybrid    Reciprocal Rank Fusion: score = sum over modes of 1 / (60 + rank)

The index is rebuilt from chunks.jsonl into a temp file and swapped in, so a
failed build never leaves a broken index behind.
"""

import json
import os
import re
import sqlite3
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from . import embed as emb
from .catalog import load_catalog
from .chunk import load_chunks
from .config import DATA

INDEX_FILE = DATA / "index.sqlite"
RRF_K = 60
POOL = 50  # candidates taken from each mode before fusing

_STOPWORDS = set("""
a an the and or of to in on at by for from with without about into over under is are was were be been being
do does did what which who whom whose why how when where this that these those it its as than then there their
they them he she his her we our you your i me my not no can could should would will shall may might must
o os as um uma uns umas de do da dos das em no na nos nas por para com sem sobre que qual quais quem como quando
onde e ou se ao aos à às é são foi eram ser estar este esta isto esse essa isso aquele aquela seu sua seus suas
""".split())


# ---------------------------------------------------------------- text prep

_IMG_RE = re.compile(r"!\[(?:fig-\d+:?\s*)?([^\]]*)\]\([^)]*\)")


def clean_text(text: str) -> str:
    """Chunk text as words: figure links become their caption."""
    text = _IMG_RE.sub(lambda m: f"[Figure: {m.group(1)}]" if m.group(1) else "[Figure]", text)
    return text


def fts_text(text: str) -> str:
    """For keyword search, LaTeX commands (\\left, \\frac) would collide with English words."""
    return re.sub(r"\\[A-Za-z]+", " ", clean_text(text))


def embed_text(chunk: dict, title: str) -> str:
    """What the model reads: book and chapter path as a header, then the text."""
    return f"{title}. {' > '.join(chunk['path'])}\n\n{clean_text(chunk['text'])}"


# ---------------------------------------------------------------- build


def _connect(path, vectors: bool = True) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    if vectors:
        import sqlite_vec

        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
    return db


_SCHEMA = f"""
CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT, authors TEXT, language TEXT);
CREATE TABLE chunks (
  rowid INTEGER PRIMARY KEY, id TEXT UNIQUE, book TEXT, section INT, seq INT, kind TEXT,
  path TEXT, pages TEXT, figures TEXT, has_math INT, words INT, file TEXT,
  char_start INT, char_end INT, text TEXT, context_before TEXT, embed_hash TEXT
);
CREATE INDEX chunks_book ON chunks(book);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  text, path, context, title, tokenize = 'unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE chunk_vec USING vec0(
  embedding float[{emb.DIMS}] distance_metric=cosine, book text, kind text
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""


def build(embeddings: bool = True, progress: Callable[[int, int, float], None] | None = None) -> dict:
    tmp = INDEX_FILE.with_suffix(".sqlite.tmp")
    if tmp.exists():
        tmp.unlink()
    db = _connect(tmp)
    db.executescript(_SCHEMA)

    titles = {}
    rows = []
    for b in load_catalog():
        chunks = load_chunks(b["id"])
        if not chunks:
            continue
        titles[b["id"]] = b["title"]
        db.execute("INSERT INTO books VALUES (?,?,?,?)",
                   (b["id"], b["title"], ", ".join(b["authors"]), (b["languages"] or [None])[0]))
        rows.extend(chunks)

    items = []
    for rowid, c in enumerate(rows, 1):
        etext = embed_text(c, titles[c["book"]])
        # text_hash includes the model's precision, which asks PyTorch: keyword-only builds skip it,
        # so they never load PyTorch at all.
        h = emb.text_hash(etext) if embeddings else None
        items.append((h, etext))
        db.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rowid, c["id"], c["book"], c["section"], c["seq"], c["kind"], json.dumps(c["path"], ensure_ascii=False),
             json.dumps(c["pages"]), json.dumps(c["figures"]), int(c["has_math"]), c["words"], c["file"],
             c["char_range"][0], c["char_range"][1], c["text"], c["context_before"], h),
        )
        db.execute("INSERT INTO chunks_fts(rowid, text, path, context, title) VALUES (?,?,?,?,?)",
                   (rowid, fts_text(c["text"]), " > ".join(c["path"]), c["context_before"], titles[c["book"]]))

    embedded = 0
    if embeddings:
        embedded = emb.embed_missing(items, progress=progress)
        vecs = emb.cached_vectors([h for h, _ in items])
        for rowid, ((h, _), c) in enumerate(zip(items, rows), 1):
            db.execute("INSERT INTO chunk_vec(rowid, embedding, book, kind) VALUES (?,?,?,?)",
                       (rowid, vecs[h], c["book"], c["kind"]))
    stats = {"books": len(titles), "chunks": len(rows), "embedded_now": embedded,
             "vectors": len(rows) if embeddings else 0,
             "model": f"{emb.MODEL}@{emb.REVISION[:7]} {emb.precision() if embeddings else ''}".strip()}
    db.executemany("INSERT INTO meta VALUES (?,?)", [(k, str(v)) for k, v in stats.items()])
    db.commit()
    db.close()
    os.replace(tmp, INDEX_FILE)
    return stats


# ---------------------------------------------------------------- search


@dataclass
class Hit:
    id: str
    book: str
    title: str
    path: list[str]
    pages: list[int] | None
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # rank in each mode that found it
    snippet: str = ""
    text: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def keyword_query(q: str) -> str | None:
    """Natural-language question -> FTS5 query. Quoted phrases stay phrases;
    other words are OR-ed (BM25 then rewards chunks matching more, and rarer, words)."""
    phrases = re.findall(r'"([^"]+)"', q)
    rest = re.sub(r'"[^"]*"', " ", q)
    words = [w for w in re.findall(r"\w+", rest.lower()) if w not in _STOPWORDS and len(w) > 1]
    terms = [f'"{p}"' for p in phrases] + [f'"{w}"' for w in dict.fromkeys(words)]
    return " OR ".join(terms) if terms else None


def _filters(book: str | None, kinds: list[str]) -> tuple[str, list]:
    sql, args = [], []
    if book:
        sql.append("c.book = ?")
        args.append(book)
    if kinds:
        sql.append(f"c.kind IN ({','.join('?' * len(kinds))})")
        args.extend(kinds)
    return (" AND " + " AND ".join(sql)) if sql else "", args


def _keyword(db, q: str, book, kinds, limit) -> list[tuple[int, float, str]]:
    fq = keyword_query(q)
    if not fq:
        return []
    where, args = _filters(book, kinds)
    # Column weights: text 1, chapter path 2 (headings are strong signals), context 0.3, title 0.5.
    sql = f"""SELECT c.rowid, bm25(chunks_fts, 1.0, 2.0, 0.3, 0.5) AS s,
                     snippet(chunks_fts, 0, '[', ']', '…', 18) AS snip
              FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid
              WHERE chunks_fts MATCH ? {where} ORDER BY s LIMIT ?"""
    return [(r["rowid"], -r["s"], r["snip"]) for r in db.execute(sql, [fq, *args, limit])]


def _semantic(db, q: str, book, kinds, limit) -> list[tuple[int, float, str]]:
    vec = emb.encode([q])[0]
    sql = "SELECT rowid, distance FROM chunk_vec WHERE embedding MATCH ? AND k = ?"
    args: list = [vec.astype(np.float32).tobytes(), limit]
    if book:
        sql += " AND book = ?"
        args.append(book)
    if len(kinds) == 1:
        sql += " AND kind = ?"
        args.append(kinds[0])
    rows = db.execute(sql + " ORDER BY distance", args).fetchall()
    return [(r["rowid"], 1 - r["distance"], "") for r in rows]


def search(q: str, mode: str = "hybrid", k: int = 10, book: str | None = None,
           kinds: tuple[str, ...] = ("body",)) -> list[Hit]:
    db = _connect(INDEX_FILE)
    kinds = list(kinds)
    results: dict[str, list[tuple[int, float, str]]] = {}
    if mode in ("keyword", "hybrid"):
        results["keyword"] = _keyword(db, q, book, kinds, POOL)
    if mode in ("semantic", "hybrid"):
        results["semantic"] = _semantic(db, q, book, kinds, POOL)

    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    snippets: dict[int, str] = {}
    for name, hits in results.items():
        for rank, (rowid, score, snip) in enumerate(hits, 1):
            ranks.setdefault(rowid, {})[name] = rank
            if snip:
                snippets[rowid] = snip
            if mode == "hybrid":
                scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (RRF_K + rank)
            else:
                scores[rowid] = score

    # Equal fused scores are common (rank 1 in either list gives 1/61). Break ties
    # toward the semantic rank: word overlap alone is the weaker signal.
    top = sorted(scores, key=lambda r: (-scores[r], ranks[r].get("semantic", POOL + 1)))[:k]
    out = []
    for rowid in top:
        c = db.execute("SELECT c.*, b.title FROM chunks c JOIN books b ON b.id = c.book WHERE c.rowid = ?",
                       (rowid,)).fetchone()
        text = c["text"]
        snip = snippets.get(rowid) or " ".join(clean_text(text).split()[:40]) + " …"
        out.append(Hit(c["id"], c["book"], c["title"], json.loads(c["path"]), json.loads(c["pages"]),
                       round(scores[rowid], 4), ranks[rowid], " ".join(snip.split()), text))
    db.close()
    return out


def index_stats() -> dict:
    if not INDEX_FILE.exists():
        return {}
    db = _connect(INDEX_FILE, vectors=False)
    stats = dict(db.execute("SELECT key, value FROM meta").fetchall())
    db.close()
    return stats
