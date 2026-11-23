"""`books` command-line interface."""

import json
import sys

import typer

from . import catalog as cat
from . import chunk as ch
from . import extract as ex

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)

# Windows consoles default to cp1252; book titles are not.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


@app.command()
def catalog():
    """Phase 0: read Calibre's metadata.db into data/catalog.json."""
    books = cat.build_catalog()
    for b in books:
        fmts = ",".join(b["formats"])
        typer.echo(f"{b['id']:<45} {fmts:<14} {', '.join(b['authors'])[:40]}")
    typer.echo(f"\n{len(books)} books -> {cat.CATALOG_FILE}")


@app.command()
def extract(
    books: list[str] = typer.Argument(None, help="Book ids (default: all)"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-extract even if unchanged"),
):
    """Phase 1: extract books into Markdown sections under data/books/."""
    catalog = cat.load_catalog() or cat.build_catalog()
    if books:
        catalog = [cat.find_book(b) for b in books]
    failed = 0
    for book in catalog:
        typer.echo(f"{book['id']:<45} ", nl=False)
        r = ex.extract_book(book, force=force)
        if r.status == "failed":
            failed += 1
            typer.secho(f"FAILED  {r.error}", fg="red")
            continue
        typer.echo(f"{r.status:<9} {r.source_format:<5} {r.sections:>4} sections {r.words:>8,} words")
        for w in r.warnings:
            typer.secho(f"{'':<46}! {w}", fg="yellow")
    raise typer.Exit(1 if failed else 0)


@app.command()
def chunk(
    books: list[str] = typer.Argument(None, help="Book ids (default: all extracted)"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-chunk even if unchanged"),
):
    """Phase 2: split chapters into citable passages (data/books/<id>/chunks.jsonl)."""
    ids = [cat.find_book(b)["id"] for b in books] if books else [
        b["id"] for b in cat.load_catalog() if ex.load_manifest(b["id"])
    ]
    total = 0
    for book_id in ids:
        r = ch.chunk_book(book_id, force=force)
        total += r["chunks"]
        w = r["words"]
        typer.echo(f"{book_id:<45} {r['status']:<8} {r['chunks']:>5} chunks  "
                   f"words median {w['median']:>3}  min {w['min']:>3}  max {w['max']:>4}")
    typer.echo(f"\n{total:,} chunks")


@app.command("show-chunk")
def show_chunk(chunk_id: str, context: bool = typer.Option(False, "--context", help="Also print context_before")):
    """Print one chunk by id, e.g. thinking-in-systems:007:003."""
    c = ch.find_chunk(chunk_id)
    pages = f"  pp. {c['pages'][0]}-{c['pages'][1]}" if c["pages"] else ""
    typer.echo(f"{c['id']}  ({c['words']} words){pages}\n{' > '.join(c['path'])}\n{c['file']} chars {c['char_range']}\n")
    if context and c["context_before"]:
        typer.secho(f"[before] {c['context_before']}\n", dim=True)
    typer.echo(c["text"])


@app.command()
def index(no_embed: bool = typer.Option(False, "--no-embed", help="Keyword index only (no model needed)")):
    """Phase 3: build data/index.sqlite (keyword index + bge-m3 vectors)."""
    from . import index as ix

    def progress(done, total, secs):
        rate = done / secs if secs else 0
        eta = (total - done) / rate / 60 if rate else 0
        typer.echo(f"\r  embedded {done:,}/{total:,}  {rate:.1f}/s  ~{eta:.0f} min left   ", nl=False)

    stats = ix.build(embeddings=not no_embed, progress=progress)
    typer.echo(f"\n{stats}")


# ---------------------------------------------------------------- Phase 5: agent tools
# Each prints compact text by default (cheap for an agent to read) or JSON with --json.

_JSON = typer.Option(False, "--json", help="Raw JSON instead of compact text")


def _emit(tool: str, fn, as_json: bool, **render):
    from . import library as lib
    from . import present

    try:
        result = fn()
    except (lib.NotFound, ValueError) as e:
        if as_json:
            typer.echo(json.dumps({"error": str(e)}, ensure_ascii=False))
        else:
            typer.secho(f"Error: {e}", fg="red", err=True)
        raise typer.Exit(2)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=1) if as_json else getattr(present, tool)(result, **render))


@app.command("library")
def library_cmd(full: bool = typer.Option(False, "--full", help="Also list what each book is a good source for"),
                as_json: bool = _JSON):
    """The shelf: every book with its thesis (and, with --full, what it is good for)."""
    from . import library as lib

    _emit("books", lib.books, as_json, full=full)


@app.command("book")
def book_cmd(book: str = typer.Argument(..., help="Book id, or an unambiguous part of id/title"), as_json: bool = _JSON):
    """One book: summary, thesis, and its chapters with one-line summaries."""
    from . import library as lib

    _emit("book", lambda: lib.book(book), as_json)


@app.command("chapter")
def chapter_cmd(ref: str = typer.Argument(..., help="BOOK:SECTION, e.g. thinking-in-systems:11"), as_json: bool = _JSON):
    """One chapter: summary, key points with passage ids, concepts, data, figures."""
    from . import library as lib

    _emit("chapter", lambda: lib.chapter(ref), as_json)


@app.command()
def search(
    query: str,
    mode: str = typer.Option("hybrid", "--mode", "-m", help="hybrid | keyword | semantic | all (compare)"),
    k: int = typer.Option(8, "--k", "-k", help="Number of results"),
    book: str = typer.Option(None, "--book", "-b", help="Limit to one book (id or part of title)"),
    notes: bool = typer.Option(False, "--notes", help="Also search notes/appendix (back matter)"),
    per_book: int = typer.Option(3, "--per-book", help="Max results per book when searching all books (0 = no cap)"),
    as_json: bool = _JSON,
):
    """Find passages that answer a question. Each result is a citable passage id.

    Hybrid search uses the warm server (`books serve`) when it is running, and
    falls back to keyword search (with a note) when it is not."""
    from . import library as lib

    for m in (["keyword", "semantic", "hybrid"] if mode == "all" else [mode]):
        if mode == "all":
            typer.secho(f"\n== {m} ==", bold=True)
        _emit("search", lambda m=m: lib.search(query, k, book, m, notes, per_book), as_json)


@app.command("read")
def read_cmd(
    chunk_id: str = typer.Argument(..., help="Passage id, e.g. thinking-in-systems:011:028"),
    before: int = typer.Option(0, "--before", "-B", help="Also show N passages before (max 5)"),
    after: int = typer.Option(0, "--after", "-A", help="Also show N passages after (max 5)"),
    around: int = typer.Option(0, "--around", "-C", help="Shorthand for -B N -A N"),
    as_json: bool = _JSON,
):
    """The exact text of a passage, with a ready-made citation."""
    from . import library as lib

    _emit("read", lambda: lib.read(chunk_id, max(before, around), max(after, around)), as_json)


@app.command("related")
def related_cmd(
    concept: str,
    min_books: int = typer.Option(1, "--min-books", help="Only concepts found in at least N books"),
    limit: int = typer.Option(10, "--limit", "-n"),
    as_json: bool = _JSON,
):
    """Where an idea appears across books (concept map from the chapter cards)."""
    from . import library as lib

    _emit("related", lambda: lib.related(concept, limit, min_books), as_json)


@app.command("figure")
def figure_cmd(ref: str = typer.Argument(..., help="BOOK:fig-NNN, e.g. thinking-in-systems:fig-036"),
               as_json: bool = _JSON):
    """What a chart, table, diagram or equation shows, as data."""
    from . import library as lib

    _emit("figure", lambda: lib.figure(ref), as_json)


@app.command()
def serve(
    status: bool = typer.Option(False, "--status", help="Only report whether the server is running"),
    stop: bool = typer.Option(False, "--stop", help="Stop a running server"),
):
    """Keep bge-m3 loaded so meaning search takes ~0.1 s instead of ~50 s (local only)."""
    from . import server

    if status or stop:
        typer.echo(server.stop() if stop else server.status())
        return
    server.run(log=typer.echo)


enrich_app = typer.Typer(no_args_is_help=True, help="Phase 4: index figures, chapters and books with Claude Haiku (batch).")
app.add_typer(enrich_app, name="enrich")
_KINDS = ["figure", "chapter", "book"]


def _enrich_books(books: list[str] | None) -> list[str]:
    from . import enrich as en  # noqa: F401

    if books:
        return [cat.find_book(b)["id"] for b in books]
    return [b["id"] for b in cat.load_catalog() if ex.load_manifest(b["id"])]


@enrich_app.command("estimate")
def enrich_estimate(
    kind: str = typer.Option("all", "--kind", help="figure | chapter | book | all"),
    book: list[str] = typer.Option(None, "--book", "-b", help="Book id(s); default all"),
):
    """Count pending requests and estimate their cost (nothing is sent)."""
    from . import enrich as en

    ids = _enrich_books(book)
    for k in (_KINDS if kind == "all" else [kind]):
        e = en.estimate(en.pending(k, ids))
        typer.echo(f"{k:<8} {e['requests']:>5} requests  ~{e['input_tokens']:>10,} in  ~{e['output_tokens']:>9,} out"
                   f"  ~${e['usd_batch']:.2f} (batch)")


@enrich_app.command("submit")
def enrich_submit(
    kind: str = typer.Option(..., "--kind", help="figure | chapter | book"),
    book: list[str] = typer.Option(None, "--book", "-b", help="Book id(s); default all"),
    yes: bool = typer.Option(False, "--yes", help="Actually send (without it: estimate only)"),
    now: bool = typer.Option(False, "--now", help="Direct calls, 8 at a time: minutes, full price (2x batch)"),
):
    """Send pending requests: as a Message Batch (50% price), or directly with --now."""
    from . import enrich as en

    items = en.pending(kind, _enrich_books(book))
    e = en.estimate(items)
    usd = e["usd_batch"] * (2 if now else 1)
    typer.echo(f"{e['requests']} {kind} requests, estimated ${usd:.2f} ({'direct' if now else 'batch'})")
    if not items or not yes:
        if items:
            typer.echo("Add --yes to send.")
        return
    if now:
        def progress(done, total, cost):
            typer.echo(f"\r  {done}/{total} done  ${cost:.3f} so far   ", nl=False)

        r = en.run_now(items, progress=progress)
        typer.echo(f"\n{r['succeeded']} stored, {len(r['failed'])} failed, ${r['usd']:.4f}")
        for f in r["failed"][:20]:
            typer.secho(f"  ! {f}", fg="yellow")
        return
    for batch_id in en.submit(items):
        typer.echo(f"submitted {batch_id}")


@enrich_app.command("collect")
def enrich_collect(wait: bool = typer.Option(False, "--wait", help="Poll until every batch has ended")):
    """Download finished batch results into data/books/<id>/enrich/."""
    from . import enrich as en

    for r in en.collect(wait=wait):
        typer.echo(json.dumps(r))
    typer.echo(f"total spent: {en.spent()}")


@app.command()
def concepts(
    build: bool = typer.Option(False, "--build", help="Rebuild data/concepts.json from the chapter cards"),
    min_books: int = typer.Option(2, "--min-books", help="Only concepts found in at least this many books"),
    find: str = typer.Option(None, "--find", "-f", help="Filter concepts containing this text"),
    limit: int = typer.Option(30, "--limit", "-n"),
):
    """Concept map: ideas that link books, with where each appears."""
    from . import concepts as cm

    if build:
        typer.echo(cm.build())
    shown = 0
    for c in cm.load():
        if c["books"] < min_books:
            continue
        if find and find.lower() not in (c["concept"] + " " + " ".join(c["aliases"])).lower():
            continue
        where = ", ".join(f"{b}:{','.join(str(s) for s in secs)}" for b, secs in c["where"].items())
        typer.secho(f"{c['concept']}  [{c['books']} books, {c['mentions']} mentions]", fg="cyan")
        if c["aliases"]:
            typer.echo(f"    aka: {', '.join(c['aliases'][:8])}")
        typer.echo(f"    {where[:300]}")
        shown += 1
        if shown >= limit:
            break


@enrich_app.command("cancel")
def enrich_cancel():
    """Cancel batches still running (finished results are still collected)."""
    from . import enrich as en

    typer.echo(f"canceled: {en.cancel_open_batches()}")


@enrich_app.command("status")
def enrich_status():
    """Submitted batches and money spent so far."""
    from . import enrich as en

    for j in en.load_jobs():
        typer.echo(f"{j['batch_id']}  {j['status']:<10} {len(j['custom_ids']):>5} requests  {j['submitted_at']}")
    typer.echo(f"spent: {en.spent()}")


@app.command("list")
def list_books():
    """List books and their extraction status."""
    for b in cat.load_catalog():
        m = ex.load_manifest(b["id"])
        status = f"{m['stats']['sections']:>4} sec {m['stats']['words']:>8,} w" if m else "not extracted"
        typer.echo(f"{b['id']:<45} {status}  {b['title'][:50]}")


@app.command()
def outline(book: str):
    """Show a book's section tree."""
    m = ex.load_manifest(cat.find_book(book)["id"])
    if not m:
        raise typer.BadParameter(f"{book} is not extracted yet")
    typer.echo(f"{m['title']} - {', '.join(m['authors'])}\n")
    for s in m["sections"]:
        tag = "" if s["kind"] == "body" else f" [{s['kind']}]"
        typer.echo(f"{s['n']:>4}  {'  ' * (s['depth'] - 1)}{s['title']}{tag}  ({s['words']:,} w)")


if __name__ == "__main__":
    app()
