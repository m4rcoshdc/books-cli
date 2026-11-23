"""Phase 4: LLM enrichment with Claude Haiku 5.5 through the Message Batches API.

The goal is NOT to think for the downstream agent (that will be a stronger
model). It is to sort, label and map the library so that agent finds things
fast. Three kinds of request, all returning JSON that follows a fixed schema:

  figure   image + caption + surrounding text -> type, summary, the DATA it shows
  chapter  a chapter's chunks (with ids) -> short summary, key points linked to
           chunk ids, concepts, entities, stated numbers/dates, questions answered
  book     the book's chapter cards -> summary, thesis, structure, top concepts

Every request has a content hash: re-running only pays for what changed.
Batches cost 50% of normal prices; results usually arrive within an hour.

Files
  data/books/<id>/enrich/{figures,chapters}.jsonl, book.json   results
  data/enrich/jobs.jsonl                                        submitted batches
  data/enrich/usage.jsonl                                       tokens + cost per request
"""

import base64
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .catalog import load_catalog
from .chunk import load_chunks
from .config import BOOKS_DIR, DATA, PROJECT_ROOT

MODEL = "claude-haiku-5-5"
# $ per million tokens, standard rate for prompts up to 100K tokens; batches pay half.
PRICE_IN, PRICE_OUT, BATCH_DISCOUNT = 0.10, 0.50, 0.5
VERSIONS = {"figure": 1, "chapter": 1, "book": 1}  # bump to redo one kind of request
ENRICH_DIR = DATA / "enrich"
JOBS_FILE = ENRICH_DIR / "jobs.jsonl"
USAGE_FILE = ENRICH_DIR / "usage.jsonl"

MIN_CHAPTER_WORDS = 250  # shorter sections are fully covered by their chunks
MAX_PART_WORDS = 20_000  # longer sections are carded in parts
MAX_IMAGE_SIDE = 1568    # larger images are downscaled before upload
FIGURE_KINDS = ("figure", "inline")


# ---------------------------------------------------------------- client


def client():
    """Anthropic client. Key from ANTHROPIC_API_KEY, else from a .env file in the
    project root (gitignored). The key is never printed or logged."""
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        env = PROJECT_ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition("=")
                if key.strip() == "ANTHROPIC_API_KEY" and value.strip():
                    os.environ["ANTHROPIC_API_KEY"] = value.strip().strip('"').strip("'")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("No API key: set ANTHROPIC_API_KEY or add it to .env in the project root")
    return anthropic.Anthropic()


# ---------------------------------------------------------------- schemas


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


_STR = {"type": "string"}
_STRS = {"type": "array", "items": _STR}

FIGURE_SCHEMA = _obj({
    "type": {"type": "string", "enum": ["chart", "diagram", "table", "map", "photo", "equation",
                                        "illustration", "decorative", "other"]},
    "title": _STR,
    "summary": _STR,
    "data": _STR,
    "text_in_image": _STR,
    "keywords": _STRS,
})

CHAPTER_SCHEMA = _obj({
    "summary": _STR,
    "content_type": {"type": "string", "enum": ["argument", "narrative", "data_analysis", "method_or_model",
                                                "history", "biography", "fiction", "poetry_or_fragments",
                                                "reference", "other"]},
    "key_points": {"type": "array", "items": _obj({"point": _STR, "chunks": _STRS})},
    "concepts": {"type": "array", "items": _obj({
        "name": _STR,
        "role": {"type": "string", "enum": ["defined", "applied", "criticized", "mentioned"]},
    })},
    "entities": {"type": "array", "items": _obj({
        "name": _STR,
        "kind": {"type": "string", "enum": ["person", "organization", "place", "work", "event", "other"]},
    })},
    "data_points": {"type": "array", "items": _obj({"fact": _STR, "value": _STR, "chunk": _STR})},
    "questions": _STRS,
})

BOOK_SCHEMA = _obj({
    "summary": _STR,
    "thesis": _STR,
    "structure": {"type": "array", "items": _obj({"section": {"type": "integer"}, "title": _STR, "one_line": _STR})},
    "top_concepts": _STRS,
    "use_for": _STRS,
})

# ---------------------------------------------------------------- prompts

SYSTEM = (
    "You index books for an AI research agent. The agent is a strong model that will read the "
    "original text itself; your job is to label, sort and extract so it can find the right "
    "passage fast. Be factual and specific, never evaluative. Do not invent anything that is "
    "not in the input. Write all output in English, even when the source is Portuguese; keep "
    "names, titles and short quotes in their original language."
)

FIGURE_TASK = """Index this figure from the book "{title}" by {authors}.
Chapter: {path}
Caption: {caption}
Text just before the figure: {before}
Text just after the figure: {after}
{labels}
Fill the fields:
- type: what kind of figure it is.
- title: short label; reuse the caption's figure number and title when there is one.
- summary: 1-2 sentences: what it shows and what point it supports in the text.
- data: the information in the figure, as data. Chart: axes with units, each series, notable values, peaks, trend and period. Table: a Markdown table transcription. Diagram or flowchart: a Mermaid flowchart of the boxes and arrows with their labels. Equation: LaTeX. Map: region, period, labelled places. Photo, illustration or decorative: empty string.
- text_in_image: labels or words printed in the image that are not in the caption, else empty string.
- keywords: 3-8 terms someone might search for to find this figure."""

CHAPTER_TASK = """Index this {part}from the book "{title}" by {authors} ({lang}).
Chapter path: {path}
The text is given as chunks; each chunk has an id. Use those ids exactly when a field asks for chunks.

Fill the fields:
- summary: at most 120 words: what this section covers and claims. Plain, factual.
- content_type: the main kind of writing.
- key_points: 3-7 central claims, findings or events, each with the chunk ids that support it.
- concepts: up to 12 ideas, theories, mechanisms or themes. Name each as a short lowercase English noun phrase in singular (e.g. "feedback loop", "global liquidity", "techno-economic paradigm"). role: defined = explained here; applied = used to analyse something; criticized = argued against; mentioned = in passing.
- entities: up to 15 named people, organizations, places, works or events that matter here.
- data_points: up to 10 numbers, dates, statistics or quantities stated in the text: fact (what is measured), value (with units), chunk (id). Empty list if none.
- questions: 3-5 questions this section answers, phrased as a researcher would ask them.

{chunks}"""

BOOK_TASK = """Below are index cards for every section of the book "{title}" by {authors}.
Write a book-level card:
- summary: at most 200 words: what the book covers, its approach and scope.
- thesis: the central argument in 1-2 sentences (for fiction: the central story and themes).
- structure: one entry per major section in reading order: section number, title, one line on its role.
- top_concepts: the 10-15 most important concepts, as lowercase English noun phrases (reuse names from the cards).
- use_for: 3-6 kinds of research questions or tasks this book is a good source for.

{cards}"""


# ---------------------------------------------------------------- requests


@dataclass
class Item:
    kind: str       # figure | chapter | book
    book: str
    key: str        # fig-012 | 007 | 007p2 | book
    custom_id: str  # hash: kind + book + key + version + content
    params: dict
    meta: dict
    est_in: int     # estimated input tokens
    est_out: int    # estimated output tokens


def _custom_id(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:40]


def _params(task: str, schema: dict, max_tokens: int, images: list[dict] | None = None) -> dict:
    content = (images or []) + [{"type": "text", "text": task}]
    return {
        "model": MODEL,
        "max_tokens": max_tokens,  # includes Haiku's thinking tokens
        "system": SYSTEM,
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        "messages": [{"role": "user", "content": content}],
    }


def _book_meta(book_id: str) -> dict:
    for b in load_catalog():
        if b["id"] == book_id:
            return b
    raise KeyError(book_id)


def _words(text: str, n: int, tail: bool = False) -> str:
    w = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text).split()
    return " ".join(w[-n:] if tail else w[:n])


def _image_block(path: Path) -> tuple[dict, int]:
    """Image as a base64 PNG/JPEG block, downscaled if large; SVG and GIF are
    rasterised (the API takes JPEG, PNG, GIF, WebP). Returns (block, est tokens)."""
    import pymupdf

    ext = path.suffix.lower().lstrip(".")
    data = path.read_bytes()
    if ext == "svg":
        doc = pymupdf.open(stream=data, filetype="svg")
        pix = doc[0].get_pixmap(dpi=150)
    else:
        pix = pymupdf.Pixmap(data)
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)  # drop transparency
    if pix.n not in (1, 3):
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)  # CMYK and others -> RGB
    scale = min(1.0, MAX_IMAGE_SIDE / max(pix.width, pix.height))
    if scale < 1.0:
        pix = pymupdf.Pixmap(pix, int(pix.width * scale), int(pix.height * scale), None)
    # Photos stay JPEG (PNG would multiply their size); drawings and equations use PNG.
    if ext in ("jpg", "jpeg"):
        raw, media = pix.tobytes("jpeg", jpg_quality=85), "image/jpeg"
    else:
        raw, media = pix.tobytes("png"), "image/png"
    block = {"type": "image", "source": {"type": "base64", "media_type": media,
                                         "data": base64.standard_b64encode(raw).decode("ascii")}}
    return block, min(1600, pix.width * pix.height // 750)  # rough image-token estimate


def figure_items(book_id: str) -> list[Item]:
    out_dir = BOOKS_DIR / book_id
    figs_file = out_dir / "figures.json"
    if not figs_file.exists():
        return []
    b = _book_meta(book_id)
    manifest = json.loads((out_dir / "book.json").read_text(encoding="utf-8"))
    sections = {s["n"]: s for s in manifest["sections"]}
    texts: dict[int, str] = {}
    items = []
    for f in json.loads(figs_file.read_text(encoding="utf-8")):
        if f["kind"] not in FIGURE_KINDS or f.get("section") not in sections:
            continue
        s = sections[f["section"]]
        if s["n"] not in texts:
            texts[s["n"]] = (out_dir / s["file"]).read_text(encoding="utf-8")
        text = texts[s["n"]]
        pos = text.find(f"](../figures/{f['file']})")
        if pos >= 0:
            pos = text.rfind("![", 0, pos)  # cut at the start of the image link itself
        before = _words(text[:pos], 120, tail=True) if pos >= 0 else ""
        after = _words(text[pos:].split(")", 1)[-1], 120) if pos >= 0 else ""
        labels = f"Labels extracted from the drawing: {f['embedded_text'][:600]}\n" if f.get("embedded_text") else ""
        task = FIGURE_TASK.format(title=b["title"], authors=", ".join(b["authors"]), path=" > ".join(s["path"]),
                                  caption=f.get("caption") or "(none)", before=before or "(none)",
                                  after=after or "(none)", labels=labels)
        image_file = out_dir / "figures" / f["file"]
        img, img_tokens = _image_block(image_file)
        content_hash = hashlib.sha1(image_file.read_bytes() + task.encode("utf-8")).hexdigest()
        items.append(Item(
            "figure", book_id, f["id"], _custom_id("figure", book_id, f["id"], VERSIONS["figure"], content_hash),
            _params(task, FIGURE_SCHEMA, 3000, [img]),
            {"section": s["n"], "file": f["file"]},
            est_in=img_tokens + len(task.split()) * 4 // 3 + 250, est_out=500,
        ))
    return items


def _chunk_text(c: dict) -> str:
    return f'<chunk id="{c["id"]}">\n{c["text"]}\n</chunk>'


def chapter_items(book_id: str) -> list[Item]:
    b = _book_meta(book_id)
    by_section: dict[int, list[dict]] = defaultdict(list)
    for c in load_chunks(book_id):
        if c["kind"] == "body":
            by_section[c["section"]].append(c)
    items = []
    for n, chunks in sorted(by_section.items()):
        total = sum(c["words"] for c in chunks)
        if total < MIN_CHAPTER_WORDS:
            continue
        parts, cur, cur_words = [], [], 0
        for c in chunks:
            if cur and cur_words + c["words"] > MAX_PART_WORDS:
                parts.append(cur)
                cur, cur_words = [], 0
            cur.append(c)
            cur_words += c["words"]
        parts.append(cur)
        for i, part in enumerate(parts, 1):
            key = f"{n:03d}" if len(parts) == 1 else f"{n:03d}p{i}"
            label = "section " if len(parts) == 1 else f"part {i} of {len(parts)} of a section "
            body = "\n\n".join(_chunk_text(c) for c in part)
            task = CHAPTER_TASK.format(part=label, title=b["title"], authors=", ".join(b["authors"]),
                                       lang={"por": "Portuguese"}.get((b["languages"] or [""])[0], "English"),
                                       path=" > ".join(part[0]["path"][:2]), chunks=body)
            words = sum(c["words"] for c in part)
            items.append(Item(
                "chapter", book_id, key,
                _custom_id("chapter", book_id, key, VERSIONS["chapter"], hashlib.sha1(task.encode("utf-8")).hexdigest()),
                _params(task, CHAPTER_SCHEMA, 6000),
                {"section": n, "part": i, "parts": len(parts), "chunks": [c["id"] for c in part],
                 "path": part[0]["path"][:2]},
                # Portuguese and dense text tokenise heavier than ~1.35 tokens/word; stay conservative.
                est_in=int(words * 1.6) + 600, est_out=1500,
            ))
    return items


def book_items(book_id: str) -> list[Item]:
    cards = load_results(book_id, "chapter")
    if not cards:
        return []
    b = _book_meta(book_id)
    compact = []
    for key in sorted(cards):
        c = cards[key]
        compact.append(json.dumps({
            "section": c["meta"]["section"], "title": " > ".join(c["meta"]["path"]),
            "summary": c["card"]["summary"], "concepts": [x["name"] for x in c["card"]["concepts"]],
        }, ensure_ascii=False))
    task = BOOK_TASK.format(title=b["title"], authors=", ".join(b["authors"]), cards="\n".join(compact))
    return [Item("book", book_id, "book",
                 _custom_id("book", book_id, VERSIONS["book"], hashlib.sha1(task.encode("utf-8")).hexdigest()),
                 # 200-section books (Engines) need room for a long structure list.
                 _params(task, BOOK_SCHEMA, 16000), {}, est_in=len(task.split()) * 3 // 2 + 300, est_out=1500)]


BUILDERS = {"figure": figure_items, "chapter": chapter_items, "book": book_items}


# ---------------------------------------------------------------- results on disk


def _results_file(book_id: str, kind: str) -> Path:
    return BOOKS_DIR / book_id / "enrich" / {"figure": "figures.jsonl", "chapter": "chapters.jsonl",
                                            "book": "book.jsonl"}[kind]


def load_results(book_id: str, kind: str) -> dict[str, dict]:
    """key -> {"custom_id", "meta", "card"}; the latest result per key wins.

    Chapter cards are cleaned on read: chunk ids that are not in the section the
    card was made from (Haiku occasionally invents one, ~0.2%) are dropped, so an
    agent is never handed a citation that does not exist. The file keeps the raw output.
    """
    f = _results_file(book_id, kind)
    out = {}
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                out[r["key"]] = r
    if kind == "chapter":
        for r in out.values():
            ok = set(r["meta"]["chunks"])
            card = r["card"]
            for kp in card["key_points"]:
                kp["chunks"] = [c for c in kp["chunks"] if c in ok]
            card["data_points"] = [d for d in card["data_points"] if d["chunk"] in ok]
    return out


def pending(kind: str, books: list[str]) -> list[Item]:
    """Requests whose exact content has no stored result and is not already in flight."""
    in_flight = {cid for job in load_jobs() if job["status"] != "collected" for cid in job["custom_ids"]}
    todo = []
    for book_id in books:
        done = {r["custom_id"] for r in load_results(book_id, kind).values()}
        todo += [it for it in BUILDERS[kind](book_id) if it.custom_id not in done and it.custom_id not in in_flight]
    return todo


def estimate(items: list[Item]) -> dict:
    tin, tout = sum(i.est_in for i in items), sum(i.est_out for i in items)
    cost = (tin * PRICE_IN + tout * PRICE_OUT) / 1e6 * BATCH_DISCOUNT
    return {"requests": len(items), "input_tokens": tin, "output_tokens": tout, "usd_batch": round(cost, 3)}


# ---------------------------------------------------------------- batches


def load_jobs() -> list[dict]:
    if not JOBS_FILE.exists():
        return []
    return [json.loads(l) for l in JOBS_FILE.read_text(encoding="utf-8").splitlines() if l]


def _save_jobs(jobs: list[dict]) -> None:
    ENRICH_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_FILE.write_text("".join(json.dumps(j) + "\n" for j in jobs), encoding="utf-8")


def submit(items: list[Item], max_mb: int = 150) -> list[str]:
    """Send items as one or more batches (each under the 256 MB request limit)."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    api = client()
    jobs = load_jobs()
    ids, group, size = [], [], 0
    groups = []
    for it in items:
        n = len(json.dumps(it.params))
        if group and size + n > max_mb * 1_000_000:
            groups.append(group)
            group, size = [], 0
        group.append(it)
        size += n
    if group:
        groups.append(group)
    for g in groups:
        batch = api.messages.batches.create(requests=[
            Request(custom_id=it.custom_id, params=MessageCreateParamsNonStreaming(**it.params)) for it in g
        ])
        jobs.append({
            "batch_id": batch.id, "status": "submitted", "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "custom_ids": [it.custom_id for it in g],
            "items": {it.custom_id: {"kind": it.kind, "book": it.book, "key": it.key, "meta": it.meta} for it in g},
        })
        ids.append(batch.id)
        _save_jobs(jobs)
    return ids


def collect(wait: bool = False, poll_seconds: int = 30) -> list[dict]:
    """Download results of finished batches into the books' enrich/ files."""
    api = client()
    reports = []
    while True:
        jobs = load_jobs()
        open_jobs = [j for j in jobs if j["status"] != "collected"]
        for job in open_jobs:
            batch = api.messages.batches.retrieve(job["batch_id"])
            if batch.processing_status != "ended":
                reports.append({"batch": job["batch_id"], "status": batch.processing_status,
                                "counts": batch.request_counts.model_dump()})
                continue
            reports.append(_store(api, job))
            job["status"] = "collected"
            _save_jobs(jobs)
        if not wait or all(j["status"] == "collected" for j in load_jobs()):
            return reports
        reports = []
        time.sleep(poll_seconds)


def _cost(tin: int, tout: int, mode: str) -> float:
    return (tin * PRICE_IN + tout * PRICE_OUT) / 1e6 * (BATCH_DISCOUNT if mode == "batch" else 1.0)


def _save(info: dict, custom_id: str, msg, mode: str, source: str, usage_log) -> tuple[str | None, int, int]:
    """Log usage and store one result. Returns (error or None, input tokens, output tokens)."""
    u = msg.usage
    tin = u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
    usage_log.write(json.dumps({"source": source, "mode": mode, "kind": info["kind"], "book": info["book"],
                                "key": info["key"], "input": tin, "output": u.output_tokens,
                                "stop": msg.stop_reason}) + "\n")
    label = f"{info['book']}:{info['key']}"
    if msg.stop_reason != "end_turn":
        return f"{label} stop={msg.stop_reason}", tin, u.output_tokens
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        card = json.loads(text)
    except json.JSONDecodeError:
        return f"{label} bad json", tin, u.output_tokens
    out = _results_file(info["book"], info["kind"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": info["key"], "custom_id": custom_id, "meta": info["meta"], "card": card},
                           ensure_ascii=False) + "\n")
    return None, tin, u.output_tokens


def _store(api, job: dict) -> dict:
    ENRICH_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed, tin, tout = 0, [], 0, 0
    with USAGE_FILE.open("a", encoding="utf-8") as usage_log:
        for res in api.messages.batches.results(job["batch_id"]):
            info = job["items"][res.custom_id]
            if res.result.type != "succeeded":
                failed.append(f"{info['book']}:{info['key']} {res.result.type}")
                continue
            err, i, o = _save(info, res.custom_id, res.result.message, "batch", job["batch_id"], usage_log)
            tin, tout = tin + i, tout + o
            if err:
                failed.append(err)
            else:
                ok += 1
    return {"batch": job["batch_id"], "status": "collected", "succeeded": ok, "failed": failed,
            "input_tokens": tin, "output_tokens": tout, "usd": round(_cost(tin, tout, "batch"), 4)}


def run_now(items: list[Item], workers: int = 8, progress=None) -> dict:
    """Direct (non-batch) calls, `workers` at a time: minutes instead of up to an
    hour, at full price. Results and usage are stored exactly like batch results."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    api = client()
    ENRICH_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed, tin, tout = 0, [], 0, 0
    with ThreadPoolExecutor(workers) as pool, USAGE_FILE.open("a", encoding="utf-8") as usage_log:
        futures = {pool.submit(api.messages.create, **it.params): it for it in items}
        for done, fut in enumerate(as_completed(futures), 1):
            it = futures[fut]
            info = {"kind": it.kind, "book": it.book, "key": it.key, "meta": it.meta}
            try:
                msg = fut.result()
            except Exception as exc:  # the SDK already retried 429/5xx
                failed.append(f"{it.book}:{it.key} {type(exc).__name__}: {str(exc)[:120]}")
                continue
            err, i, o = _save(info, it.custom_id, msg, "now", "direct", usage_log)
            usage_log.flush()
            tin, tout = tin + i, tout + o
            if err:
                failed.append(err)
            else:
                ok += 1
            if progress:
                progress(done, len(items), _cost(tin, tout, "now"))
    return {"succeeded": ok, "failed": failed, "input_tokens": tin, "output_tokens": tout,
            "usd": round(_cost(tin, tout, "now"), 4)}


def cancel_open_batches() -> list[str]:
    api = client()
    ids = [j["batch_id"] for j in load_jobs() if j["status"] != "collected"]
    for batch_id in ids:
        api.messages.batches.cancel(batch_id)
    return ids


def spent() -> dict:
    """Total tokens and cost of everything collected so far (batch rows at half price)."""
    tin = tout = n = 0
    usd = 0.0
    if USAGE_FILE.exists():
        for line in USAGE_FILE.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            tin, tout, n = tin + r["input"], tout + r["output"], n + 1
            usd += _cost(r["input"], r["output"], r.get("mode", "batch"))
    return {"requests": n, "input_tokens": tin, "output_tokens": tout, "usd": round(usd, 4)}
