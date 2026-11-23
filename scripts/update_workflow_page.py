"""Refresh the per-book numbers embedded in docs/workflow.html from data/.

Run after any extraction change, then republish the page:
    uv run python scripts/update_workflow_page.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "workflow.html"
DATA = ROOT / "data"


def book_rows() -> list[dict]:
    catalog = {b["id"]: b for b in json.loads((DATA / "catalog.json").read_text(encoding="utf-8"))["books"]}
    rows = []
    for manifest_file in sorted((DATA / "books").glob("*/book.json")):
        m = json.loads(manifest_file.read_text(encoding="utf-8"))
        b = catalog[m["id"]]
        figs_file = manifest_file.parent / "figures.json"
        figs = json.loads(figs_file.read_text(encoding="utf-8")) if figs_file.exists() else []
        body_figs = [f for f in figs if f["kind"] == "figure"]
        words = {k: sum(s["words"] for s in m["sections"] if s["kind"] == k) for k in ("front", "body", "back")}
        rows.append({
            "id": m["id"],
            "t": re.split(r":| - ", b["title"])[0].strip(),
            "a": ", ".join(b["authors"]),
            "lang": (b["languages"] or ["?"])[0],
            "fmts": list(b["formats"]),
            "src": m["source"]["format"],
            "score": m["source"]["structure_score"],
            "cands": {c["format"]: c["score"] for c in m["candidates"]},
            "secs": m["stats"]["sections"],
            **words,
            "figs": len(body_figs),
            "caps": sum(1 for f in body_figs if f.get("caption")),
            "inl": m["stats"].get("inline_images", 0),
            "eqs": m["stats"].get("equations", 0),
            "cards": len(read_jsonl(manifest_file.parent / "enrich" / "chapters.jsonl", key="key")),
            "described": len(read_jsonl(manifest_file.parent / "enrich" / "figures.jsonl", key="key")),
            **book_card(manifest_file.parent),
            "chunks": chunk_meta(m["id"]).get("chunks", 0),
            "chunkMedian": chunk_meta(m["id"]).get("words", {}).get("median", 0),
            "fixed": b.get("overridden", []),
        })
    return rows


def read_jsonl(path: Path, key: str | None = None) -> list | dict:
    if not path.exists():
        return {} if key else []
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
    return {r[key]: r for r in rows} if key else rows


def book_card(book_dir: Path) -> dict:
    cards = read_jsonl(book_dir / "enrich" / "book.jsonl", key="key")
    if "book" not in cards:
        return {}
    c = cards["book"]["card"]
    return {"thesis": c["thesis"], "useFor": c["use_for"][:4]}


def enrich_summary() -> dict:
    """Totals for the stage 4 card: counts, figure types, cost by step, top concepts."""
    usage = read_jsonl(DATA / "enrich" / "usage.jsonl")
    if not usage:
        return {}

    def usd(rows):
        return sum((r["input"] * 0.10 + r["output"] * 0.50) / 1e6 * (0.5 if r.get("mode", "batch") == "batch" else 1)
                   for r in rows)

    by_kind = {}
    for kind in ("figure", "chapter", "book"):
        rows = [r for r in usage if r["kind"] == kind]
        by_kind[kind] = {"requests": len(rows), "usd": round(usd(rows), 3)}
    types, figures, cards, books = {}, 0, 0, 0
    for d in (DATA / "books").iterdir():
        for r in read_jsonl(d / "enrich" / "figures.jsonl", key="key").values():
            types[r["card"]["type"]] = types.get(r["card"]["type"], 0) + 1
            figures += 1
        cards += len(read_jsonl(d / "enrich" / "chapters.jsonl", key="key"))
        books += "book" in read_jsonl(d / "enrich" / "book.jsonl", key="key")
    concepts_file = DATA / "concepts.json"
    concepts = json.loads(concepts_file.read_text(encoding="utf-8"))["concepts"] if concepts_file.exists() else []
    return {
        "spent": {"usd": round(usd(usage), 2), "input_tokens": sum(r["input"] for r in usage),
                  "output_tokens": sum(r["output"] for r in usage)},
        "byKind": by_kind, "figureTypes": types, "figures": figures, "cards": cards, "books": books,
        "crossBook": sum(1 for c in concepts if c["books"] >= 2),
        "topConcepts": [{"concept": c["concept"], "books": list(c["where"])} for c in concepts[:10]],
    }


def chunk_meta(book_id: str) -> dict:
    f = DATA / "books" / book_id / "chunks.meta.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def chunk_histogram() -> list[dict]:
    """Words-per-chunk histogram in 100-word bins; 200-499 is the target band."""
    counts = [0] * 7
    for f in (DATA / "books").glob("*/chunks.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            counts[min(json.loads(line)["words"] // 100, 6)] += 1
    labels = ["0", "100", "200", "300", "400", "500", "600+"]
    return [{"label": l, "n": n, "target": 2 <= i <= 4} for i, (l, n) in enumerate(zip(labels, counts))]


def replace_block(html: str, name: str, js: str) -> str:
    start, end = f"/*{name}:START*/", f"/*{name}:END*/"
    if start not in html:
        sys.exit(f"{name} data block not found in docs/workflow.html")
    return re.sub(re.escape(start) + ".*?" + re.escape(end), lambda _: start + js + end, html, flags=re.S)


def main() -> None:
    html = PAGE.read_text(encoding="utf-8")
    dump = lambda v: json.dumps(v, ensure_ascii=False, separators=(",", ":"))  # noqa: E731
    html = replace_block(html, "BOOKS", f"const BOOKS = {dump(book_rows())};")
    html = replace_block(html, "HIST", f"const CHUNK_HIST = {dump(chunk_histogram())};")
    compare_file = DATA / "evals" / "search_compare.json"
    compare = json.loads(compare_file.read_text(encoding="utf-8")) if compare_file.exists() else {}
    for r in compare.get("results", []):  # the page needs ranks only, not chunk ids
        for m in r["modes"].values():
            m.pop("top", None)
    html = replace_block(html, "COMPARE", f"const COMPARE = {dump(compare)};")
    html = replace_block(html, "ENRICH", f"const ENRICH = {dump(enrich_summary())};")
    PAGE.write_text(html, encoding="utf-8")
    print(f"updated {PAGE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
