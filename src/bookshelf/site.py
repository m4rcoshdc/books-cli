"""The shelf: a static web page of the library, in the spirit of stripe.press.

`books site` writes data/site/:

  index.html        the page (src/bookshelf/site_template.html with the library inlined)
  covers/<id>.jpg   each Calibre cover, shrunk to at most COVER_HEIGHT pixels tall

A book can use a different cover than Calibre's: set "cover_url" for it in
metadata_overrides.json (then `books catalog`). Only the URL is tracked in git;
the image is downloaded once into data/cache/covers/, named by a hash of the URL.

Everything the page shows comes from library.books()/book() and the concept map,
so it says the same thing the CLI does. The data is inlined rather than fetched
so the page also works straight from disk (file://), with no server.

Reading status (unread / reading / read) is saved in reading.json. `books site --serve`
serves the page on 127.0.0.1 and saves each click there; opened from disk instead,
the page keeps changes in that browser only (localStorage) and says so.

Spine colours are picked here, from the cover's pixels: a browser can't read the
pixels of a file:// image (the canvas is "tainted"), and it's deterministic anyway.
"""

import colorsys
import hashlib
import json
import shutil
import urllib.request
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from collections import Counter
from pathlib import Path

import pymupdf

from . import library as lib
from . import reading
from .config import CACHE_DIR, DATA, LIBRARY

SITE_DIR = DATA / "site"
TEMPLATE = Path(__file__).with_name("site_template.html")
COVER_HEIGHT = 900


def _cover(src: Path, dst: Path) -> dict:
    """Shrink the cover into dst and return its size and two colours picked from it."""
    pix = pymupdf.Pixmap(str(src))
    if pix.n != 3 or pix.alpha:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
    w, h = pix.width, pix.height
    shrink = 0
    while (h >> (shrink + 1)) >= COVER_HEIGHT:
        shrink += 1
    if shrink:
        pix.shrink(shrink)
    pix.save(str(dst), jpg_quality=85)

    # Colours from a ~40 px thumbnail: the most common coarse colour, and the most
    # common colourful one (for the spine's title and emblem).
    small = pymupdf.Pixmap(pix, 0) if pix.alpha else pymupdf.Pixmap(pix)
    k = 0
    while (small.width >> (k + 1)) >= 40:
        k += 1
    if k:
        small.shrink(k)
    s, n = small.samples, small.n
    buckets = Counter()
    for i in range(0, len(s), n):
        r, g, b = s[i], s[i + 1], s[i + 2]
        buckets[(r >> 5, g >> 5, b >> 5)] += 1

    def rgb(key):
        return tuple(min(255, (c << 5) + 16) for c in key)

    base = rgb(buckets.most_common(1)[0][0])
    vivid = [(cnt, key) for key, cnt in buckets.items()
             if colorsys.rgb_to_hsv(*(c / 255 for c in rgb(key)))[1] > 0.45 and rgb(key) != base]
    accent = rgb(max(vivid)[1]) if vivid else None
    return {"w": w, "h": h, "base": "#%02x%02x%02x" % base,
            "accent": "#%02x%02x%02x" % accent if accent else None}


def _cover_source(meta: dict) -> Path | None:
    """The file to make the cover from: the cover_url override (downloaded once), else Calibre's."""
    url = meta.get("cover_url")
    if url:
        cached = CACHE_DIR / "covers" / f"{meta['id']}-{hashlib.sha1(url.encode()).hexdigest()[:10]}"
        if not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(url, headers={"User-Agent": "books-cli"})
            with urllib.request.urlopen(req, timeout=60) as r:
                tmp = cached.with_suffix(".part")
                tmp.write_bytes(r.read())
            tmp.replace(cached)
        return cached
    if meta.get("cover") and (LIBRARY / meta["cover"]).exists():
        return LIBRARY / meta["cover"]
    return None


def _related(concepts: list[dict]) -> dict[str, list[dict]]:
    """For each book, the other books it shares the most concepts with (from the concept map)."""
    shared: dict[str, Counter] = {}
    names: dict[tuple[str, str], list[str]] = {}
    for c in concepts:
        where = list(c.get("where", {}))
        for a in where:
            for b in where:
                if a != b:
                    shared.setdefault(a, Counter())[b] += 1
                    names.setdefault((a, b), []).append(c["concept"])
    return {a: [{"id": b, "shared": n, "concepts": names[(a, b)][:4]} for b, n in cnt.most_common(4)]
            for a, cnt in shared.items()}


def build() -> Path:
    from .concepts import load as load_concepts

    covers = SITE_DIR / "covers"
    if covers.exists():
        shutil.rmtree(covers)
    covers.mkdir(parents=True)

    related = _related(load_concepts())
    cat = lib._catalog()
    shelf = []
    for entry in lib.books()["books"]:
        bid = entry["id"]
        full = lib.book(bid)
        meta = cat[bid]
        cover = None
        if src := _cover_source(meta):
            cover = _cover(src, covers / f"{bid}.jpg")
            cover["src"] = f"covers/{bid}.jpg"
        # Top-level chapters, unless the outline is flat or has only a few parts.
        body = [c for c in full["chapters"] if c["kind"] == "body"]
        top = [c for c in body if c["depth"] == 1]
        chapters = len(top) if len(top) >= 3 else len(body)
        shelf.append({
            "id": bid, "title": meta["title"], "authors": meta["authors"],
            "language": entry["language"], "words": entry["words"], "publisher": meta.get("publisher"),
            "pubdate": (meta.get("pubdate") or "")[:4] or None,
            "thesis": full["thesis"] or full["summary"], "use_for": full["use_for"],
            "concepts": full["top_concepts"], "chapters": chapters, "cover": cover,
            "related": [r for r in related.get(bid, []) if r["id"] in cat],
        })

    html = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps({"books": shelf, "note": lib.CARD_NOTE, "reading": reading.load()}, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")  # never close the <script> early
    out = SITE_DIR / "index.html"
    out.write_text(html.replace("/*__LIBRARY__*/null", payload), encoding="utf-8")
    return out


# ---------------------------------------------------------------- local server (--serve)

SERVE_HOST, SERVE_PORT = "127.0.0.1", 8790


class _SiteHandler(SimpleHTTPRequestHandler):
    """Static files from data/site, plus GET/POST /api/reading.

    Writes are accepted only from the page itself: same Origin, JSON body. Another site
    open in the browser can't send a JSON POST here without a CORS preflight, which
    this server never answers.
    """

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")  # always the latest build
        super().end_headers()

    def do_GET(self):
        if self.path == "/api/reading":
            return self._json(200, reading.load())
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/reading":
            return self._json(404, {"error": "unknown path"})
        host = self.headers.get("Host", "")
        if self.headers.get("Origin") != f"http://{host}" or                 not self.headers.get("Content-Type", "").startswith("application/json"):
            return self._json(403, {"error": "only the shelf page can change reading status"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            if body.get("id") not in lib._catalog():
                raise ValueError(f"unknown book {body.get('id')!r}")
            self._json(200, reading.mark(body["id"], body.get("status", "")))
        except (ValueError, AttributeError) as e:
            self._json(400, {"error": str(e)})

    def log_message(self, *args):
        pass


def serve(port: int = SERVE_PORT, log=print) -> None:
    handler = partial(_SiteHandler, directory=str(SITE_DIR))
    httpd = HTTPServer((SERVE_HOST, port), handler)
    log(f"Shelf on http://{SERVE_HOST}:{port} (local only). Reading status saves to {reading.READING_FILE.name}. "
        "Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
