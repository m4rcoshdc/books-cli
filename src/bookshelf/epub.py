"""Minimal EPUB reader: reading order (spine) and table of contents (NCX or nav).

An EPUB is a zip. META-INF/container.xml points to the OPF package file, which
lists every file (manifest), the reading order (spine), and the TOC location.
"""

import posixpath
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from lxml import etree

_PARSER = etree.XMLParser(recover=True, resolve_entities=False, huge_tree=True)
_OPS_NS = "{http://www.idpf.org/2007/ops}"
_HTML_TYPES = {"application/xhtml+xml", "text/html", "application/xml"}


@dataclass
class TocEntry:
    title: str
    file: str  # zip path, normalised
    fragment: str | None
    depth: int  # 1 = top level


def _xp(node, expr: str):
    """XPath with `{name}` shorthand for namespace-agnostic element names."""
    return node.xpath(expr.replace("{", '*[local-name()="').replace("}", '"]'))


def _norm(base: str, href: str) -> tuple[str, str | None]:
    path, _, frag = href.partition("#")
    full = posixpath.normpath(posixpath.join(base, unquote(path))) if path else base
    return full, (unquote(frag) or None)


class Epub:
    def __init__(self, path: Path):
        self.zf = zipfile.ZipFile(path)
        self._names = {n.lower(): n for n in self.zf.namelist()}
        container = etree.fromstring(self.zf.read("META-INF/container.xml"), _PARSER)
        self.opf_path = _xp(container, "//{rootfile}/@full-path")[0]
        opf = etree.fromstring(self.read(self.opf_path), _PARSER)
        base = posixpath.dirname(self.opf_path)

        manifest = {}
        for item in _xp(opf, "//{manifest}/{item}"):
            href = item.get("href")
            if href:
                manifest[item.get("id")] = {
                    "path": _norm(base, href)[0],
                    "type": item.get("media-type", ""),
                    "props": item.get("properties", ""),
                }

        self.spine: list[str] = []
        for ref in _xp(opf, "//{spine}/{itemref}"):
            item = manifest.get(ref.get("idref"))
            if item and item["type"] in _HTML_TYPES and ref.get("linear") != "no":
                self.spine.append(item["path"])

        self.toc: list[TocEntry] = []
        nav = next((m for m in manifest.values() if "nav" in m["props"].split()), None)
        if nav:
            self.toc = self._parse_nav(nav["path"])
        if not self.toc:
            spine_el = _xp(opf, "//{spine}")
            ncx_id = spine_el[0].get("toc") if spine_el else None
            ncx = manifest.get(ncx_id) or next(
                (m for m in manifest.values() if m["type"] == "application/x-dtbncx+xml"), None
            )
            if ncx:
                self.toc = self._parse_ncx(ncx["path"])

    def resolve(self, path: str) -> str | None:
        return self._names.get(path.lower())

    def read(self, path: str) -> bytes:
        name = self.resolve(path)
        if name is None:
            raise KeyError(path)
        return self.zf.read(name)

    def _parse_ncx(self, path: str) -> list[TocEntry]:
        root = etree.fromstring(self.read(path), _PARSER)
        base = posixpath.dirname(path)
        out: list[TocEntry] = []

        def walk(node, depth):
            for point in _xp(node, "./{navPoint}"):
                label = " ".join(_xp(point, "string(./{navLabel}/{text})").split())
                src = _xp(point, "string(./{content}/@src)")
                if src:
                    out.append(TocEntry(label, *_norm(base, src), depth))
                walk(point, depth + 1)

        navmap = _xp(root, "//{navMap}")
        if navmap:
            walk(navmap[0], 1)
        return out

    def _parse_nav(self, path: str) -> list[TocEntry]:
        root = etree.fromstring(self.read(path), _PARSER)
        base = posixpath.dirname(path)
        navs = _xp(root, "//{nav}")
        toc_navs = [n for n in navs if "toc" in (n.get(f"{_OPS_NS}type") or "").split()]
        nav = (toc_navs or navs or [None])[0]
        if nav is None:
            return []
        out: list[TocEntry] = []

        def walk(ol, depth):
            for li in _xp(ol, "./{li}"):
                link = _xp(li, "./{a}") or _xp(li, "./{span}")
                if link:
                    label = " ".join("".join(link[0].itertext()).split())
                    href = link[0].get("href")
                    if href:
                        out.append(TocEntry(label, *_norm(base, href), depth))
                for child in _xp(li, "./{ol}"):
                    walk(child, depth + 1)

        for ol in _xp(nav, "./{ol}") or _xp(nav, ".//{ol}")[:1]:
            walk(ol, 1)
        return out
