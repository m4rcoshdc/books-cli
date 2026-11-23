"""Figures: keep images instead of throwing them away.

Extraction replaces every meaningful image with a Markdown image link
(`![fig-012: Figure 5.3 ...](../figures/fig-012.png)`) at the exact spot it
appeared, saves the image file, and records metadata in figures.json. A later
LLM step reads the images (vision) and writes descriptions back.

Dropped as decorative: tiny images (spacers, bullets), images used 3+ times
(ornaments, logos), and image fallbacks of MathML equations.
"""

import re
from dataclasses import asdict, dataclass, field

import pymupdf

CAPTION_RE = re.compile(
    r"^\s*(fig(ure)?\.?|table|chart|exhibit|map|plate|diagram|graph|illustration|"
    r"quadro|tabela|gr[aá]fico|figura)\s*[\dIVX]",
    re.I,
)
_TOKEN = "{}"
TOKEN_RE = re.compile("(\\d+)")
_MIN_SIDE = (24, 16)  # px: smaller is decoration
_REPEAT_LIMIT = 3


@dataclass
class Figure:
    id: str
    file: str
    kind: str  # figure | inline | front (decided after sectioning)
    caption: str | None = None
    alt: str | None = None
    width: int | None = None
    height: int | None = None
    source: str | None = None  # path inside the EPUB, or "page N" for PDFs
    page: int | None = None
    hint: str | None = None  # CSS class of the container, e.g. "cl-eq-c-v1"
    embedded_text: str | None = None  # PDF vector figures: their label text
    section: int | None = None


@dataclass
class FigureSink:
    """Collects figures (and inserts for Markdown) while a book is converted."""

    figures: list[Figure] = field(default_factory=list)
    blobs: dict[str, bytes] = field(default_factory=dict)
    inserts: list[str] = field(default_factory=list)
    equations: int = 0

    def token(self, markdown: str) -> str:
        """Placeholder that survives HTML->Markdown conversion untouched."""
        self.inserts.append(markdown)
        return _TOKEN.format(len(self.inserts) - 1)

    def resolve(self, text: str) -> str:
        return TOKEN_RE.sub(lambda m: self.inserts[int(m.group(1))], text)

    def add(self, data: bytes, ext: str, **meta) -> Figure:
        fid = f"fig-{len(self.figures) + 1:03d}"
        fig = Figure(id=fid, file=f"{fid}.{ext}", kind=meta.pop("kind", "figure"), **meta)
        self.figures.append(fig)
        self.blobs[fid] = data
        return fig

    def markdown(self, fig: Figure) -> str:
        label = fig.caption or fig.alt
        label = f"{fig.id}: {_short(label)}" if label else fig.id
        link = f"![{label}](../figures/{fig.file})"
        return link if fig.kind == "inline" else f"\n\n{link}\n\n"

    def as_json(self) -> list[dict]:
        return [{k: v for k, v in asdict(f).items() if v is not None} for f in self.figures]


def _short(text: str, n: int = 140) -> str:
    text = " ".join(text.replace("[", "(").replace("]", ")").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def image_size(data: bytes, ext: str) -> tuple[int | None, int | None]:
    if ext == "svg":
        head = data[:2000].decode("utf-8", "ignore")
        vb = re.search(r'viewBox="[\d.\-]+[ ,]+[\d.\-]+[ ,]+([\d.]+)[ ,]+([\d.]+)"', head)
        if vb:
            return round(float(vb.group(1))), round(float(vb.group(2)))
        w, h = re.search(r'width="([\d.]+)', head), re.search(r'height="([\d.]+)', head)
        return (round(float(w.group(1))) if w else None, round(float(h.group(1))) if h else None)
    try:
        pix = pymupdf.Pixmap(data)
        return pix.width, pix.height
    except Exception:
        return None, None


def is_decorative(width: int | None, height: int | None, uses: int) -> bool:
    if uses >= _REPEAT_LIMIT:
        return True
    return bool(width and height and (width < _MIN_SIDE[0] or height < _MIN_SIDE[1]))


def _text(el) -> str:
    return " ".join(el.get_text(" ").split()) if el is not None else ""


def find_caption(img) -> str | None:
    """Caption from <figcaption>, a caption-classed element, or a neighbour
    paragraph that starts like "Figure 2.1" / "Table 3"."""
    fig = img.find_parent("figure")
    if fig is not None:
        cap = fig.find("figcaption")
        if cap is not None and _text(cap):
            return _text(cap)[:400]
    block = img.find_parent(["figure", "div", "p", "td", "li", "section"]) or img.parent
    for anc in (block, block.parent if block is not None else None):
        if anc is None or anc.name in ("body", "html"):
            continue
        cap = anc.find(class_=re.compile("caption", re.I))
        if cap is not None and _text(cap):
            return _text(cap)[:400]
    if block is None:
        return None
    neighbours = block.find_next_siblings(limit=2) + block.find_previous_siblings(limit=1)
    for sib in neighbours:
        t = _text(sib)
        if t and CAPTION_RE.match(t):
            return t[:400]
    return None


def clean_alt(alt: str | None) -> str | None:
    if not alt:
        return None
    alt = " ".join(alt.split())
    if re.search(r"[/\\]|\.(png|jpe?g|gif|svg)$", alt, re.I) or alt.lower() in ("image", "img", "figure", "graphic", ""):
        return None
    return alt
