"""PDF -> sections. Uses the PDF outline when present; otherwise detects
chapter-opening pages ("CHAPTER" / "CAPÍTULO", tolerant of OCR noise); otherwise
falls back to fixed page ranges. Every page is marked `<!-- p.N -->` so later
layers can cite page numbers.
"""

import re
from dataclasses import dataclass

import pymupdf
from collections import Counter

from .figures import CAPTION_RE, FigureSink

_CHAPTER_RE = re.compile(r"^\W*(C\w{0,3}A?\w{0,3}P\w{0,4}[IÍ]\w{0,3}TULO|CHAPTER)\b", re.I)
_PAGE_REF_RE = re.compile(r",\s*\d{2,4}\b")  # "Something, 151" lines in a contents page
_ROMAN_RE = re.compile(r"^[IVXLCJ1lv]{1,5}$", re.I)  # OCR turns I into J/1/l
_FIXED_PAGES = 25


@dataclass
class PdfSection:
    title: str
    depth: int
    first_page: int
    markdown: str


def _is_scanned(page) -> bool:
    area = abs(page.rect)
    return any(abs(pymupdf.Rect(i["bbox"])) > 0.8 * area for i in page.get_image_info())


def _merge(rects: list) -> list:
    rects = [pymupdf.Rect(r) for r in rects]
    merged = True
    while merged:
        merged = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if rects[i].intersects(rects[j] + (-6, -6, 6, 6)):
                    rects[i] |= rects.pop(j)
                    merged = True
                    break
            if merged:
                break
    return rects


def _figure_regions(page, blocks, repeated: set[int]) -> list:
    """Areas of the page that are pictures: raster images, plus clusters of
    vector drawings that look like a chart/diagram (enough paths, or a
    "Figure/Table N" caption right next to them)."""
    page_area = abs(page.rect)
    regions = []
    for info in page.get_image_info(xrefs=True):
        r = pymupdf.Rect(info["bbox"]) & page.rect
        if r.width >= 60 and r.height >= 40 and info.get("xref") not in repeated:
            regions.append(r)
    drawings = page.get_drawings()
    if drawings:
        for cl in page.cluster_drawings(drawings=drawings):
            if cl.width < 100 or cl.height < 60 or abs(cl) > 0.6 * page_area:
                continue  # rules, underlines, page frames
            paths = sum(1 for d in drawings if d["rect"].intersects(cl))
            captioned = any(CAPTION_RE.match(b[4]) and _caption_near(b, cl) for b in blocks)
            if paths >= 4 or captioned:
                regions.append(cl)
    return [r + (-3, -3, 3, 3) for r in _merge(regions)]


def _caption_near(block, region) -> bool:
    b = pymupdf.Rect(block[:4])
    horizontal = min(b.x1, region.x1) - max(b.x0, region.x0) > 0
    vertical_gap = max(region.y0 - b.y1, b.y0 - region.y1)
    return horizontal and vertical_gap < 50


def _inside(block, region) -> bool:
    b = pymupdf.Rect(block[:4])
    return abs(b) > 0 and abs(b & region) / abs(b) >= 0.6


def _page_markdown(page, number: int, sink: FigureSink | None = None, repeated: set[int] = frozenset()) -> str:
    blocks = [b for b in page.get_text("blocks") if b[6] == 0]
    regions = [] if sink is None or _is_scanned(page) else _figure_regions(page, blocks, repeated)

    figures = []  # (y0, markdown)
    for region in regions:
        inside = [b for b in blocks if _inside(b, region)]
        caption = next((b[4] for b in inside if CAPTION_RE.match(b[4])), None) or next(
            (b[4] for b in blocks if CAPTION_RE.match(b[4]) and _caption_near(b, region)), None)
        label_text = " ".join(" ".join(b[4].split()) for b in inside) or None
        pix = page.get_pixmap(clip=region & page.rect, dpi=150)
        fig = sink.add(
            pix.tobytes("png"), "png", source=f"page {number}", page=number,
            width=pix.width, height=pix.height,
            caption=" ".join(caption.split())[:400] if caption else None,
            embedded_text=label_text,
        )
        figures.append((region.y0, sink.markdown(fig).strip()))
        blocks = [b for b in blocks if b not in inside]  # labels leave the text flow
    figures.sort()

    paras = []
    for block in blocks:
        while figures and figures[0][0] <= block[1] + 2:
            paras.append(figures.pop(0)[1])
        text = block[4]
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # de-hyphenate
        text = " ".join(text.split())
        if not text:
            continue
        if paras and paras[-1].startswith("!["):
            paras.append(text)
            continue
        # OCR often emits one block per line: re-join hyphenated words and
        # sentences that clearly continue (next block starts lowercase/digit).
        if paras and paras[-1].endswith("-") and text[0].isalpha():
            paras[-1] = paras[-1][:-1] + text
        elif paras and not re.search(r"[.!?:;\"”»)]$", paras[-1]) and re.match(r"[a-zà-ÿ0-9]", text):
            paras[-1] += " " + text
        else:
            paras.append(text)
    paras.extend(md for _, md in figures)
    return f"<!-- p.{number} -->\n\n" + "\n\n".join(paras)


def _chapter_title(page) -> str | None:
    lines = [ln.strip() for ln in page.get_text().splitlines() if ln.strip()][:15]
    head = " ".join(lines)
    if not _CHAPTER_RE.match(head) or len(_PAGE_REF_RE.findall(head)) >= 2:
        return None
    # Skip the "CAPÍTULO" line and roman numeral, then take short title lines.
    i = next(k for k, ln in enumerate(lines) if _CHAPTER_RE.match(ln) or k == len(lines) - 1) + 1
    title = []
    for ln in lines[i:]:
        if not title and (_ROMAN_RE.match(ln.replace(" ", "")) or not re.search(r"\w{3}", ln)):
            continue
        if len(ln) > 45 or len(title) >= 6:
            break
        title.append(ln)
    words = [w for w in " ".join(title).split() if re.search(r"\w{2}", w) or w.isalpha()]
    return " ".join(words[:10]).title() or "Untitled chapter"


def read_pdf(path, sink: FigureSink | None = None) -> tuple[list[PdfSection], list[str]]:
    doc = pymupdf.open(path)
    warnings = []
    # Images placed on 3+ pages are logos/ornaments, not figures.
    xref_pages = Counter(x[0] for p in doc for x in {(i[0],) for i in p.get_images()})
    repeated = {x for x, n in xref_pages.items() if n >= 3}
    pages = [_page_markdown(p, i + 1, sink, repeated) for i, p in enumerate(doc)]

    starts: list[tuple[int, str, int]] = []  # (page index, title, depth)
    outline = doc.get_toc()
    if outline:
        starts = [(max(pg - 1, 0), title, lvl) for lvl, title, pg in outline if pg > 0]
    else:
        for i, page in enumerate(doc):
            title = _chapter_title(page)
            if title:
                starts.append((i, title, 1))
        if starts:
            warnings.append("no PDF outline; sections detected from chapter headings (OCR)")
    if not starts:
        warnings.append(f"no PDF outline or chapter headings; fixed {_FIXED_PAGES}-page sections")
        starts = [(i, f"Pages {i + 1}-{min(i + _FIXED_PAGES, len(pages))}", 1)
                  for i in range(0, len(pages), _FIXED_PAGES)]

    empty = sum(1 for p in doc if len(p.get_text().strip()) < 50)
    if empty > len(pages) * 0.2:
        warnings.append(f"{empty}/{len(pages)} pages have no text layer (scanned without OCR?)")

    sections = []
    if starts[0][0] > 0:
        sections.append(PdfSection("Front matter", 1, 1, "\n\n".join(pages[: starts[0][0]])))
    for k, (first, title, depth) in enumerate(starts):
        last = starts[k + 1][0] if k + 1 < len(starts) else len(pages)
        if last <= first and k + 1 < len(starts):
            last = first  # several outline entries on the same page
        sections.append(PdfSection(title, depth, first + 1, "\n\n".join(pages[first:last])))
    return sections, warnings
