# Books CLI

Repo: github.com/m4rcoshdc/books-cli (public; renamed from Books-Analyser on 2026-11-23).
The local folder is still `Books-Analyser`, because the skill's path and Claude's
session memory point at it. Rename it only together with those.

Turns the user's Calibre library (`books/`, mirrors their Kindle) into an
agent-friendly knowledge layer. It's also a learning project: keep each
pipeline stage explicit and inspectable, and prefer plain deterministic code
before reaching for an LLM.

## Run

`README.md` is the public guide for people forking this. Keep it in sync when commands change.

```
uv run books build [ID...] [--yes] [--no-embed]   # whole pipeline; free without --yes; re-runs skip unchanged
uv run books skill [--global] [--print]           # generate the Claude Code skill for this library

uv run books catalog            # Phase 0: books/metadata.db -> data/catalog.json
uv run books extract [ID...]    # Phase 1: -> data/books/<id>/{book.json,sections/*.md}
uv run books chunk [ID...]      # Phase 2: -> data/books/<id>/chunks.jsonl
uv run books show-chunk BOOK:SECTION:SEQ
uv run books index [--no-embed]  # Phase 3: -> data/index.sqlite (FTS5 + bge-m3 vectors)
uv run books search "question" [-m keyword|semantic|hybrid|all] [-b BOOK] [--json]
uv run python scripts/compare_search.py   # same 12 questions through all three modes
uv run books enrich estimate                         # Phase 4: pending requests + cost, sends nothing
uv run books enrich submit --kind figure|chapter|book [-b ID] [--now] --yes
uv run books enrich collect --wait | status | cancel
uv run books concepts [--build] [--find TEXT]       # concept map across books
uv run books list | outline ID

# Phase 5: agent tools (compact text by default, --json for programs)
uv run books serve [--status|--stop]   # keep bge-m3 loaded: hybrid search ~0.1 s instead of ~50 s
uv run books library [--full]          # shelf: thesis (+ what each book is good for)
uv run books book BOOK                 # card + chapters with one-liners (BOOK = id, title or author part)
uv run books chapter BOOK:N            # chapter card: key points with passage ids, concepts, data, figures
uv run books search "question" [-b BOOK] [-k N] [--per-book N] [--notes]
uv run books read CHUNK_ID [-C N]      # exact text + "Cite as" line
uv run books related "concept"         # where an idea appears across books
uv run books figure BOOK:fig-NNN       # chart values / table / Mermaid / LaTeX
uv run books site [--open] [--serve]   # bookshelf web page -> data/site/index.html; --serve on :8790 saves reading status

# Phase 6: evals (re-run `eval run` after ANY change to search or ranking)
uv run books eval run                  # score all modes on the frozen set -> data/evals/report.json (free)
uv run books eval judge [--yes]        # Haiku grades top-5 pairs not judged yet (cents)
uv run books eval questions [--yes]    # only if the sample changes; bump QUESTION_VERSION to rebuild
```

## Layout

- `src/bookshelf/catalog.py`: reads Calibre's SQLite **read-only**. Book ids are slugs, kept stable by Calibre uuid.
- `src/bookshelf/epub.py`: spine + TOC (NCX or nav) reader.
- `src/bookshelf/pdf.py`: PDF outline, or OCR chapter detection; `<!-- p.N -->` page markers.
- `src/bookshelf/figures.py`: keeps images as `figures/fig-NNN.*` with a Markdown image link where they appeared, finds captions, and skips ornaments (tiny, or used 3+ times). Metadata goes in `figures.json`.
- `src/bookshelf/mathml.py`: MathML to LaTeX (`$...$` / `$$...$$`). Publishers' PNG copies of the same equations are dropped.
- `src/bookshelf/extract.py`: tries every format (MOBI/AZW3 converted via Calibre's `ebook-convert`, cached in `data/cache/epub`), scores each split with `structure_score`, keeps the best. Candidates and scores are recorded in `book.json`.

- `src/bookshelf/chunk.py`: ~400-word passages that never cross a chapter. Figures stay with their captions and equations with their sentences. Chunk id `book:SSS:NNN` is the citation key. `context_before` holds the tail of the previous chunk, for search only. Bump `CHUNKER_VERSION` when chunking logic changes.

- `src/bookshelf/index.py`: one SQLite file with `chunks`, `chunks_fts` (FTS5 BM25, accent-insensitive, column weights text 1 / path 2 / context 0.3 / title 0.5) and `chunk_vec` (sqlite-vec, cosine). Hybrid = Reciprocal Rank Fusion (k=60) over the top 50 of each mode. It's built into a temp file and swapped in.
- `src/bookshelf/embed.py`: BAAI/bge-m3, pinned revision. Runs on the GPU (RTX 3050 Ti, 4 GB) in fp16 via CUDA PyTorch (`pyproject.toml` points `torch` at the cu132 index), falling back to CPU fp32. Precision is part of the cache key. Cold start is ~47 s per process (imports plus weight loading); a query itself takes ~0.06 s, so agents use the long-running `books serve` (Phase 5). Download it with `scripts/download_model.py` (~2.3 GB into the Hugging Face cache). Vectors are cached by text hash in `data/cache/embeddings.sqlite`, so rebuilds only embed changed chunks. Disk on C: is tight; check free space before large downloads.

- `src/bookshelf/enrich.py`: Claude Haiku 5.5 (the user's choice: cheap labeling; the downstream agent will be the strong model). Output is structured JSON (`output_config.format`), effort low. Three request kinds: figure (image to type and data: chart values, Markdown tables, Mermaid diagrams, LaTeX), chapter card (summary, key points with chunk ids, concepts, entities, data points, questions), and book card. Batch by default (50% off); `--now` makes direct calls, 8 at a time, at full price. Content-hashed custom ids, so re-runs only pay for changes. Results go in `data/books/<id>/enrich/*.jsonl`, usage and cost in `data/enrich/usage.jsonl`. Bump `VERSIONS[kind]` to redo a kind.
- Figure descriptions reach search through the chunker (`load_figure_notes`); stage 1 files are never modified. Order: figures, then `books chunk`, then `books index`, then chapter cards (cards read chunk text).
- `src/bookshelf/concepts.py`: groups chapter-card concept names with bge-m3 (cosine >= 0.90; 0.86 wrongly merged "leverage" with "leverage point") into `data/concepts.json`. No LLM.
- `src/bookshelf/library.py`: Phase 5 core. Every agent-facing tool is a function returning plain dicts (`TOOLS`). The CLI, the server and any future MCP layer must stay thin wrappers over it. Search has no per-book cap by default (Phase 6 evals showed a cap of 3 only cost answers); `per_book=2` is for comparative questions. Hybrid falls back to keyword (with a note) when the model isn't loaded here or in the server.
- Book cards' `structure` section numbers are unreliable: Haiku misnumbered 127 of 549 entries. Chapter one-liners come from chapter cards first. A book-card line is used only when its title matches the real section. Never key anything on an LLM-produced section number without checking it against `book.json`.
- `src/bookshelf/present.py`: compact text rendering per tool. Text is about half the tokens of JSON; keep full ids so they can be pasted into the next command.
- `src/bookshelf/server.py`: `books serve`, a stdlib HTTP server on 127.0.0.1:8765 that keeps bge-m3 warm. It refuses browser requests (Origin / Sec-Fetch-Site headers) and is single-threaded (one GPU model). Restart it after changing `library.py`.
- `src/bookshelf/site.py`: `books site`, a static Stripe-Press-style shelf. Data comes from `library.books()/book()` plus the concept map (related books), inlined into `site_template.html` (edit the template, not `data/site/`) so it works from `file://`. Covers are shrunk with pymupdf, and spine colours are picked from cover pixels in Python (a browser can't read `file://` image pixels). CSS 3D, no libraries. A `cover_url` in `metadata_overrides.json` swaps a book's cover (downloaded once into `data/cache/covers/`; the image itself never goes in git). Reading status (unread/reading/read) is page-only: `reading.py` keeps it in `reading.json` (project root, gitignored, personal state, not derived), written through `--serve`'s same-origin JSON `POST /api/reading`; opened from `file://` the page falls back to localStorage. The agent tools deliberately ignore it. Look: bottle-green reading room, walnut ledge, a lamp glow that follows the selected book, brass = read, red ribbon = reading. `.claude/launch.json` runs `books site --serve` for the browser pane. Speed rule (measured; it was the lag): per-frame code must never set a CSS custom property on a slot or on a container of the shelf (every face inherits it and is restyled each frame); set the property itself (`perspectiveOrigin`, the lamp's `left`) and read all positions before writing.
- `.claude/skills/books/SKILL.md`: teaches agents the zoom-in workflow and citation rules. It's **generated** by `books skill` (`skill.py`) from `src/bookshelf/skill_template.md`, filling in this folder's path, the book list and non-English books. It's gitignored because it contains a machine path and a personal library. To change the skill, edit the template, then re-run `books skill`. `--global` also writes `~/.claude/skills/books/`.
- Portability: unknown ids in `metadata_overrides.json` only warn (forks carry the author's file). `ebook-convert` is looked up on PATH, then in Calibre's default install folders on Windows, macOS and Linux. PyTorch uses the cu132 index everywhere except macOS. The `.env` reader accepts UTF-8, UTF-8 with BOM and UTF-16 (PowerShell's `>`). Only tested on Windows.
- `src/bookshelf/evals.py`: Phase 6. 6 passages per book (one seed per book), Haiku writes a question per passage (lookup / paraphrase / crosslang), `lint()` drops generic questions and ones that point at the source (applied to all questions, never only to failures). Dev/test split by chunk-id hash. Metrics: hit@1/5/10, MRR, near@5, book@5, answered@5 (Haiku judge, blind to the target). `CONFIGS["hybrid"]` must match the library's default search. Eval spend is logged separately in `data/evals/usage.jsonl`. Baseline (2026-11-23, 123 questions): hybrid hit@5 0.88, hit@10 0.92, MRR 0.76. Decide changes on dev, look at test once.
- API key: `ANTHROPIC_API_KEY` env var or the project `.env` (gitignored). Never print it.

## Adding a book

The user adds it in Calibre and copies the library over `books/`. Then `books build NEW_ID --yes` (it stops a running server), then `books eval run` (check for regressions), `books skill` (update the book list) and `books serve`. Inside `build`, the order is catalog → extract → figure notes → chunk → index → chapter cards → book cards → concept map. The order matters: figure notes go into chunks, and cards read chunks. Leave the eval set frozen, unless a new baseline is wanted.

## Rules

- `books/` is a static copy of the Calibre library (the user won't round-trip through Calibre). Never write inside it. Fix metadata in `metadata_overrides.json` (keyed by book id) instead.
- Everything derived goes in `data/` and can be rebuilt.
- `docs/workflow.html` is the published workflow tracker artifact (https://claude.ai/artifact/Maw4PRmmgSst44yGa394BD). After data changes, run `uv run python scripts/update_workflow_page.py`, edit the stage text by hand, and republish to the same URL.
- `data/` is gitignored: it holds the full text of purchased books. Keep it local.
- Bump `EXTRACTOR_VERSION` in `extract.py` whenever extraction logic changes, otherwise unchanged books are skipped.
- Section kinds: `front` / `body` / `back`. Later phases (chunking, search, LLM enrichment) should mostly use `body`.

## Roadmap

0 catalog → 1 extract → 2 chunk → 3 index → 4 enrich ($1.96 on Haiku) → 5 access (CLI tools, warm server, skill) → 6 evals ($0.18; per-book cap turned off by default). All phases done. Open ideas, to be judged by `books eval run`: on paraphrase questions semantic (80%) beats hybrid (77%); human-written questions would make the set more realistic; cross-language has only 5 questions. MCP was deliberately skipped: Claude Code uses the CLI and skill. Add it as a thin layer over `library.TOOLS` only if the library is wanted in apps without a shell (e.g. a regular Claude Desktop chat).

Kindle highlights were dropped on 2026-11-23 (the user has few highlights). If they come back, they'd attach to chunk ids like the enrichment cards do.
