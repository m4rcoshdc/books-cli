"""Phase 1: book file -> clean Markdown, one file per TOC section.

Pipeline per book:
  1. pick the best source format (EPUB > AZW3 > MOBI); convert non-EPUB with
     Calibre's `ebook-convert` (cached by source hash)
  2. read spine (reading order) + TOC
  3. drop a marker into the HTML at every TOC target, convert the whole spine
     to Markdown, then split the text on the markers. This handles both
     "one chapter per file" and "many chapters per file" books.
  4. write data/books/<id>/book.json + sections/NNN-slug.md

Re-running skips books whose source hash and extractor version are unchanged.
"""

import hashlib
import json
import re
import shutil
import subprocess
import warnings as _warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import posixpath
from collections import Counter
from urllib.parse import unquote

from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning
from markdownify import MarkdownConverter

from .catalog import slugify
from .config import BOOKS_DIR, CACHE_DIR, LIBRARY
from .epub import Epub, TocEntry, _norm
from .figures import FigureSink, clean_alt, find_caption, image_size, is_decorative
from .mathml import mathml_to_latex
from .pdf import read_pdf

_warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

EXTRACTOR_VERSION = 8  # bump whenever extraction logic changes
FORMAT_PREFERENCE = ["EPUB", "AZW3", "MOBI", "PDF"]
_MIN_TOC_ENTRIES = 3  # fewer than this on a long book -> split on headings instead

_MARK = "\ue000{}\ue001"
_MARK_RE = re.compile("\ue000(\\d+)\ue001")

_FRONT_RE = re.compile(
    r"^(cover|capa|title( page)?|half[- ]title|copyright|ficha|contents|table of contents|"
    r"sum[aá]rio|dedication|epigraph|also by|praise|frontispiece|about the publisher|"
    r"other books|maps?$|list of (illustrations|figures|maps|tables))",
    re.I,
)
_BACK_RE = re.compile(
    r"^(index|[ií]ndice remissivo|notes|notas|endnotes|bibliography|bibliografia|references|"
    r"acknowledg|agradecimentos|about the authors?|sobre o autor|further reading|sources|"
    r"glossary|credits|permissions|photo(graph)? credits|copyright)",
    re.I,
)


class _Converter(MarkdownConverter):
    """Markdown without links/images, with less aggressive escaping."""

    def __init__(self, **options):
        options.setdefault("heading_style", "ATX")
        options.setdefault("bullets", "-")
        options.setdefault("strip", ["a", "img"])
        options.setdefault("escape_underscores", False)
        options.setdefault("escape_asterisks", False)
        super().__init__(**options)


_CONVERTER = _Converter()


@dataclass
class Section:
    title: str
    depth: int
    path: list[str]
    markdown: str = ""
    kind: str = "body"
    page: int | None = None  # first page, for PDFs

    @property
    def words(self) -> int:
        return len(self.markdown.split())


@dataclass
class Result:
    book_id: str
    status: str  # extracted | skipped | failed
    source_format: str | None = None
    sections: int = 0
    words: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------- source files


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_EBOOK_CONVERT_PATHS = (  # Calibre's default install locations, for when it isn't on PATH
    r"C:\Program Files\Calibre2\ebook-convert.exe",
    "/Applications/calibre.app/Contents/MacOS/ebook-convert",
    "/opt/calibre/ebook-convert",
)


def _ebook_convert() -> str:
    exe = shutil.which("ebook-convert") or next((p for p in _EBOOK_CONVERT_PATHS if Path(p).exists()), None)
    if not exe:
        raise RuntimeError("Calibre's ebook-convert not found (needed for MOBI/AZW3): install Calibre or add it to PATH")
    return exe


def to_epub(src: Path, sha: str) -> Path:
    """Convert MOBI/AZW3 to EPUB once; cache by content hash."""
    dest = CACHE_DIR / "epub" / f"{sha[:16]}.epub"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + ".tmp.epub")
    proc = subprocess.run(
        [_ebook_convert(), str(src), str(tmp)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"ebook-convert failed: {proc.stderr.strip()[-500:]}")
    tmp.replace(dest)
    return dest


def pick_source(book: dict) -> tuple[str, Path] | None:
    for fmt in FORMAT_PREFERENCE:
        if fmt in book["formats"]:
            return fmt, LIBRARY / book["formats"][fmt]
    return None


# ---------------------------------------------------------------- text


def _clean(md: str) -> str:
    md = md.replace("\xa0", " ").replace("\xad", "").replace("\u200b", "")
    md = "\n".join(line.rstrip() for line in md.splitlines())
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


_IMG_REF_RE = re.compile(
    rb"""(?:src|xlink:href|href)\s*=\s*["']([^"']+\.(?:png|jpe?g|gif|svg|webp|bmp))["']""", re.I
)


def _count_image_uses(epub: Epub) -> Counter:
    """How often each image file is referenced across the book (ornament filter)."""
    uses: Counter = Counter()
    for doc in epub.spine:
        try:
            raw = epub.read(doc)
        except KeyError:
            continue
        base = posixpath.dirname(doc)
        for ref in _IMG_REF_RE.findall(raw):
            uses[_norm(base, unquote(ref.decode("utf-8", "replace")))[0].lower()] += 1
    return uses


def _kids(el) -> list:
    return [c for c in el.children if getattr(c, "name", None) or str(c).strip()]


def _next_element(el):
    sib = el.next_sibling
    while sib is not None and not getattr(sib, "name", None) and not str(sib).strip():
        sib = sib.next_sibling
    return sib if getattr(sib, "name", None) else None


def _convert_math(soup, sink: FigureSink) -> None:
    """MathML -> LaTeX text; drop the image fallback publishers put next to it."""
    for m in soup.find_all("math"):
        # The fallback may sit next to <math> or next to a wrapper that holds
        # only the <math> (e.g. <span class="epub"><math/></span><span class="mobi"><img/></span>).
        el = m
        for _ in range(3):
            fallback = _next_element(el)
            if fallback is not None and (fallback.name == "img" or (
                    fallback.find("img") is not None and not fallback.get_text(strip=True))):
                fallback.decompose()
                break
            parent = el.parent
            if parent is None or parent.name in ("p", "div", "body", "td", "li") or len(_kids(parent)) > 1:
                break
            el = parent
        latex = mathml_to_latex(m)
        block = m.get("display") == "block"
        sink.equations += 1
        m.replace_with(sink.token(f"\n\n$${latex}$$\n\n" if block else f"${latex}$"))


def _convert_images(soup, doc: str, epub: Epub, sink: FigureSink, uses: Counter) -> None:
    base = posixpath.dirname(doc)
    # Inline <svg> wrapping an <image> (common for covers) -> treat as that image.
    for svg in soup.find_all("svg"):
        inner = svg.find("image")
        if inner is not None:
            svg.replace_with(inner)
        else:
            data = str(svg).encode("utf-8")
            w, h = image_size(data, "svg")
            if is_decorative(w, h, 1):
                svg.decompose()
                continue
            fig = sink.add(data, "svg", source=f"{doc} (inline svg)", width=w, height=h,
                           caption=find_caption(svg))
            svg.replace_with(sink.token(sink.markdown(fig)))
    for img in soup.find_all(["img", "image"]):
        ref = img.get("src") or img.get("xlink:href") or img.get("href")
        if not ref:
            img.decompose()
            continue
        path = _norm(base, unquote(ref))[0]
        try:
            data = epub.read(path)
        except KeyError:
            img.decompose()
            continue
        ext = path.rsplit(".", 1)[-1].lower().replace("jpeg", "jpg")
        w, h = image_size(data, ext)
        if is_decorative(w, h, uses[path.lower()]):
            img.decompose()
            continue
        block = img.find_parent(["p", "div", "li", "td", "figure"])
        context = block.get_text(" ", strip=True) if block is not None else ""
        inline = len(context) > 40 and (h is None or h < 80)
        hint = " ".join(block.get("class", [])) if block is not None else None
        fig = sink.add(
            data, ext, kind="inline" if inline else "figure", source=path, width=w, height=h,
            caption=None if inline else find_caption(img), alt=clean_alt(img.get("alt")),
            hint=hint or None,
        )
        img.replace_with(sink.token(sink.markdown(fig)))


def _doc_to_markdown(
    raw: bytes,
    targets: list[tuple[int, str | None]],
    doc: str,
    epub: Epub,
    sink: FigureSink,
    uses: Counter,
) -> tuple[str, str | None]:
    """Convert one spine document; insert split markers at TOC targets,
    figure links where images were, and LaTeX where MathML was.

    Returns (markdown, first heading text).
    """
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    body = soup.body or soup
    _convert_math(soup, sink)
    _convert_images(soup, doc, epub, sink, uses)
    heading = body.find(re.compile(r"^h[1-3]$"))
    heading_text = " ".join(heading.get_text(" ").split()) if heading else None

    at_start = []
    for idx, frag in targets:
        el = None
        if frag:
            el = soup.find(id=frag) or soup.find(attrs={"name": frag})
        if el is None or el is body or el.name in ("html", "body"):
            at_start.append(_MARK.format(idx))
        else:
            el.insert_before(NavigableString(_MARK.format(idx) + "\n\n"))
    if at_start:
        body.insert(0, NavigableString("".join(at_start) + "\n\n"))
    return sink.resolve(_CONVERTER.convert_soup(body)), heading_text


def _kind(title: str) -> str:
    t = title.strip().strip(".:").strip()
    if _FRONT_RE.match(t):
        return "front"
    if _BACK_RE.match(t):
        return "back"
    return "body"


_HEADING_RE = re.compile(r"^(#{1,2}) +(.+)$", re.M)
_IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def split_on_headings(stream: str) -> list[Section]:
    """Fallback when the TOC is missing or useless: split on # / ## headings.

    A heading with no text under it (e.g. "# IV" followed by "## Feedback and
    Oscillation") is folded into the next section's title.
    """
    stream = _MARK_RE.sub("", stream)
    matches = list(_HEADING_RE.finditer(stream))
    sections: list[Section] = []
    front = _clean(stream[: matches[0].start()] if matches else stream)
    if front:
        sections.append(Section("Front matter", 1, ["Front matter"], front, "front"))
    pending: list[str] = []
    carried: list[str] = []  # images under a heading-only section move along with its title
    for k, m in enumerate(matches):
        end = matches[k + 1].start() if k + 1 < len(matches) else len(stream)
        title = " ".join(_IMAGE_LINK_RE.sub("", m.group(2)).replace("#", "").split())
        text = _clean(stream[m.end() : end])
        if not _IMAGE_LINK_RE.sub("", text).strip():
            if title:
                pending.append(title)
            if text:
                carried.append(text)
            continue
        full = ". ".join(pending + [title])
        body = "\n\n".join(carried + [text])
        pending, carried = [], []
        sections.append(Section(full, 1, [full], f"{m.group(0).strip()}\n\n{body}"))
    return sections


_BARE_NUMBER_RE = re.compile(r"^[\divxlc]+\.?$", re.I)


def _classify(sections: list[Section]) -> None:
    """Inherit front/back kind from parents, then classify by own title.

    Bare-number titles ("1", "2", ... under Notes) inherit the previous kind.
    """
    prev = "front"
    for s in sections:
        if s.kind == "body":
            if _BARE_NUMBER_RE.match(s.title.strip()):
                s.kind = prev
            else:
                kinds = [_kind(t) for t in s.path]
                s.kind = next((k for k in kinds if k != "body"), "body")
        prev = s.kind


def split_sections(epub: Epub) -> tuple[list[Section], list[str], FigureSink]:
    warnings: list[str] = []
    sink = FigureSink()
    uses = _count_image_uses(epub)
    spine_set = {p.lower() for p in epub.spine}
    toc: list[TocEntry] = []
    for e in epub.toc:
        if e.file.lower() in spine_set:
            toc.append(e)
    dropped = len(epub.toc) - len(toc)
    if dropped:
        warnings.append(f"{dropped} TOC entries point outside the reading order")

    by_file: dict[str, list[tuple[int, str | None]]] = {}
    for i, e in enumerate(toc):
        by_file.setdefault(e.file.lower(), []).append((i, e.fragment))

    parts, headings = [], []
    for doc in epub.spine:
        try:
            raw = epub.read(doc)
        except KeyError:
            warnings.append(f"missing spine file {doc}")
            continue
        md, heading = _doc_to_markdown(raw, by_file.get(doc.lower(), []), doc, epub, sink, uses)
        parts.append(md)
        headings.append((doc, heading, len(md)))
    stream = "\n\n".join(parts)

    if len(toc) < _MIN_TOC_ENTRIES:
        sections = split_on_headings(stream)
        if len(sections) >= _MIN_TOC_ENTRIES:
            warnings.append(f"TOC has {len(toc)} entries; split on headings instead")
        else:
            warnings.append("no usable TOC or headings; one section per spine file")
            sections = []
            for (doc, heading, _), md in zip(headings, parts):
                title = heading or Path(doc).stem
                sections.append(Section(title, 1, [title], _clean(_MARK_RE.sub("", md))))
            sections = [s for s in sections if s.words]
        _classify(sections)
        return sections, warnings, sink

    # Split on markers: pieces = [before, idx, text, idx, text, ...]
    pieces = _MARK_RE.split(stream)
    sections: list[Section] = []
    front = _clean(pieces[0])
    if front:
        sections.append(Section("Front matter", 1, ["Front matter"], front, "front"))

    seen = set(int(i) for i in pieces[1::2])
    missing = len(toc) - len(seen)
    if missing:
        warnings.append(f"{missing} TOC targets not found in the text")

    stack: list[str] = []
    for idx_str, text in zip(pieces[1::2], pieces[2::2]):
        entry = toc[int(idx_str)]
        stack = stack[: entry.depth - 1] + [entry.title]
        sections.append(Section(entry.title, entry.depth, list(stack), _clean(text)))

    _classify(sections)

    # Drop empty sections (duplicate TOC targets, part dividers with no text)
    # unless they only contain their own heading.
    kept = []
    for s in sections:
        body_text = _IMAGE_LINK_RE.sub("", re.sub(r"^#+ .*$", "", s.markdown, flags=re.M)).strip()
        if body_text or s.depth == 1:
            kept.append(s)
    return kept, warnings, sink


# ---------------------------------------------------------------- output


def _front_matter(meta: dict) -> str:
    lines = ["---"] + [f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items()]
    return "\n".join(lines + ["---", ""])


_FIG_LINK_RE = re.compile(r"\]\(\.\./figures/(fig-\d+)\.")


def _write_figures(out: Path, sections: list[Section], sink: FigureSink) -> dict:
    """Save figures that ended up in a kept section; return stats."""
    fig_dir = out / "figures"
    if fig_dir.exists():
        shutil.rmtree(fig_dir)
    by_id = {f.id: f for f in sink.figures}
    kept = []
    for n, s in enumerate(sections, 1):
        for fid in _FIG_LINK_RE.findall(s.markdown):
            fig = by_id.get(fid)
            if fig is None or fig.section is not None:
                continue
            fig.section = n
            if s.kind == "front" and fig.kind == "figure":
                fig.kind = "front"  # covers, title-page art
            kept.append(fig)
    if kept:
        fig_dir.mkdir(parents=True)
        for fig in kept:
            (fig_dir / fig.file).write_bytes(sink.blobs[fig.id])
    sink.figures = kept
    (out / "figures.json").write_text(
        json.dumps(sink.as_json(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    kinds = Counter(f.kind for f in kept)
    return {
        "figures": kinds.get("figure", 0),
        "inline_images": kinds.get("inline", 0),
        "front_images": kinds.get("front", 0),
        "equations": sink.equations,
    }


def write_book(
    book: dict, sections: list[Section], source: dict, warnings: list[str], sink: FigureSink
) -> Path:
    out = BOOKS_DIR / book["id"]
    sec_dir = out / "sections"
    if sec_dir.exists():
        shutil.rmtree(sec_dir)
    sec_dir.mkdir(parents=True)
    fig_stats = _write_figures(out, sections, sink)
    fig_count = Counter(f.section for f in sink.figures if f.kind != "inline")

    index = []
    for n, s in enumerate(sections, 1):
        name = f"{n:03d}-{slugify(s.title, 8)[:60]}.md"
        meta = {
            "book": book["id"],
            "section": n,
            "title": s.title,
            "path": s.path,
            "kind": s.kind,
        }
        (sec_dir / name).write_text(_front_matter(meta) + s.markdown + "\n", encoding="utf-8")
        entry = {
            "n": n,
            "title": s.title,
            "depth": s.depth,
            "path": s.path,
            "kind": s.kind,
            "words": s.words,
            "file": f"sections/{name}",
        }
        if s.page is not None:
            entry["page"] = s.page
        if fig_count.get(n):
            entry["figures"] = fig_count[n]
        index.append(entry)

    manifest = {
        "id": book["id"],
        "title": book["title"],
        "authors": book["authors"],
        "languages": book["languages"],
        "source": source,
        "extractor_version": EXTRACTOR_VERSION,
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stats": {
            "sections": len(sections),
            "words": sum(s.words for s in sections),
            "body_words": sum(s.words for s in sections if s.kind == "body"),
            **fig_stats,
        },
        "warnings": warnings,
        "sections": index,
    }
    (out / "book.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def pdf_sections(path: Path) -> tuple[list[Section], list[str], FigureSink]:
    sink = FigureSink()
    raw, warnings = read_pdf(path, sink)
    sections = [
        Section(s.title, s.depth, [s.title], _clean(s.markdown),
                "front" if s.title == "Front matter" else "body", s.first_page)
        for s in raw
    ]
    _classify(sections)
    return [s for s in sections if s.words], warnings, sink


def load_manifest(book_id: str) -> dict | None:
    f = BOOKS_DIR / book_id / "book.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def structure_score(sections: list[Section]) -> float:
    """0..1: how plausible a section split looks. Penalises text stranded in
    front matter, one section swallowing the book, and many tiny sections."""
    total = sum(s.words for s in sections)
    body = [s for s in sections if s.kind == "body"]
    body_words = sum(s.words for s in body)
    if total < 5000 or not body_words:
        return 0.0
    front = sum(s.words for s in sections if s.kind == "front") / total
    largest = max(s.words for s in body) / body_words
    tiny = sum(1 for s in body if s.words < 50) / len(body)
    score = 1.0 - 2 * max(0.0, front - 0.15) - max(0.0, largest - 0.35) - 0.5 * tiny
    return round(max(score, 0.0), 3)


# PDF text carries page-layout noise (running heads, page numbers); prefer
# reflowable formats unless the PDF's structure is clearly better.
_FORMAT_FACTOR = {"PDF": 0.95}


def _extract_format(fmt: str, path: Path, sha: str) -> tuple[list[Section], list[str], FigureSink]:
    if fmt == "PDF":
        return pdf_sections(path)
    epub_path = path if fmt == "EPUB" else to_epub(path, sha)
    return split_sections(Epub(epub_path))  # -> sections, warnings, figures


def extract_book(book: dict, force: bool = False) -> Result:
    """Try formats in preference order, score each split, keep the best."""
    sources = [(f, LIBRARY / book["formats"][f]) for f in FORMAT_PREFERENCE if f in book["formats"]]
    if not sources:
        return Result(book["id"], "failed", error=f"no supported format in {list(book['formats'])}")
    shas = {f: _sha256(p) for f, p in sources}

    prev = load_manifest(book["id"])
    if (
        not force
        and prev
        and prev["extractor_version"] == EXTRACTOR_VERSION
        and prev.get("source_hashes") == shas
    ):
        # Text is unchanged, but Calibre metadata may have been fixed since.
        fresh = {k: book[k] for k in ("title", "authors", "languages")}
        if any(prev.get(k) != v for k, v in fresh.items()):
            prev.update(fresh)
            (BOOKS_DIR / book["id"] / "book.json").write_text(
                json.dumps(prev, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return Result(
            book["id"], "skipped", prev["source"]["format"], prev["stats"]["sections"],
            prev["stats"]["words"], prev["warnings"],
        )

    candidates, errors = [], []
    for fmt, path in sources:
        try:
            sections, warns, sink = _extract_format(fmt, path, shas[fmt])
        except Exception as exc:  # try the next format
            errors.append(f"{fmt}: {type(exc).__name__}: {exc}")
            continue
        score = structure_score(sections) * _FORMAT_FACTOR.get(fmt, 1.0)
        candidates.append((score, fmt, path, sections, warns, sink))
    if not candidates:
        return Result(book["id"], "failed", error="; ".join(errors))

    # Penalise candidates that lost a lot of text relative to the richest one.
    most = max(sum(s.words for s in c[3]) for c in candidates)
    ranked = [
        (round(score * min(1.0, sum(s.words for s in secs) / most / 0.9), 3), fmt, path, secs, w, sk)
        for score, fmt, path, secs, w, sk in candidates
    ]
    best = max(ranked, key=lambda c: c[0])  # ties keep format preference (max is stable)
    score, fmt, path, sections, warnings, sink = best
    warnings = list(warnings) + errors
    if score < 0.6:
        warnings.append(f"low structure score {score}; check the outline")
    if len(ranked) > 1:
        warnings.append(
            "chose " + fmt + " over " + ", ".join(f"{c[1]} ({c[0]})" for c in ranked if c is not best)
        )

    words = sum(s.words for s in sections)
    source = {
        "format": fmt,
        "file": book["formats"][fmt],
        "sha256": shas[fmt],
        "converted": fmt in ("MOBI", "AZW3"),
        "structure_score": score,
    }
    out = write_book(book, sections, source, warnings, sink)
    manifest = json.loads((out / "book.json").read_text(encoding="utf-8"))
    manifest["source_hashes"] = shas
    manifest["candidates"] = [
        {"format": c[1], "score": c[0], "sections": len(c[3]), "words": sum(s.words for s in c[3])}
        for c in ranked
    ]
    (out / "book.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return Result(book["id"], "extracted", fmt, len(sections), words, warnings)
