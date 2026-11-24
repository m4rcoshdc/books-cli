# Books CLI

Turn your Calibre library into a knowledge base that AI agents can **search, read
and cite**, down to the exact passage.

Ask Claude Code *"What does Munger say about Coca-Cola in his talks?"* and it finds
the right talk, reads it, and answers with citations like
`Munger, Poor Charlie's Almanack, “Practical Thought About Practical Thought?” [poor-charlies-almanack:019:004]`.
Every id points to a real passage you can open with `books read`.

It's a retrieval-augmented generation (RAG) system you run on your own machine:

- **Search by keyword and by meaning**, across languages: an English question finds a Portuguese passage.
- **Citable passages.** Every book is split into ~400-word passages with stable ids. Page numbers are included for PDFs.
- **Charts and tables as data** (optional). A vision model writes down the values a chart shows, so search can find them.
- **Chapter and book summaries and a concept map** (optional). They help an agent decide where to look before it reads.
- **A Claude Code skill included.** It teaches the agent how to research your library: zoom from shelf to book to
  chapter to passage, quote only the real text, and cite every claim. See [Use it from Claude Code](#use-it-from-claude-code).
- **A measured search.** A built-in eval tells you how often the right passage comes up, so changes are compared, not guessed.
- **Local first.** Book text, the index and the embedding model stay on your machine. Only the optional summary steps send text to the Anthropic API.

It was built step by step as a learning project. [`docs/workflow.html`](docs/workflow.html) is the build log, with
each stage, the decisions behind it and what broke. Download and open it in a browser.

## What you need

| | |
|---|---|
| **Python 3.12+ and [uv](https://docs.astral.sh/uv/)** | uv installs everything else |
| **A Calibre library** | The folder with `metadata.db` and your book files. EPUB, PDF, MOBI and AZW3 are supported. **Files must be DRM-free.** |
| **Calibre itself** | Only for MOBI/AZW3: its `ebook-convert` turns them into EPUB |
| **~6 GB of disk** | PyTorch (~3 GB) and the bge-m3 embedding model (~2.3 GB) |
| **An NVIDIA GPU** (optional) | Embedding ~10,000 passages takes ~8 minutes on a 4 GB laptop GPU, or ~2–3 hours on a CPU. It's a one-time cost; later runs only embed new passages |
| **An Anthropic API key** (optional) | Only for the summary steps (`--yes`) and for building your own eval set |

> Developed and tested on Windows 11. macOS and Linux should work, but haven't been tested yet:
> please open an issue if something breaks.

## Quick start

```bash
# 1. Get the code
git clone https://github.com/<you>/books-cli.git      # or fork first, then clone your fork
cd books-cli
uv sync

# 2. Point it at your Calibre library (read-only: nothing is ever written there)
#    Either copy the library folder to ./books/, or set the path:
export BOOKS_LIBRARY="/path/to/Calibre Library"        # PowerShell: $env:BOOKS_LIBRARY = "C:\...\Calibre Library"

# 3. metadata_overrides.json holds the author's fixes for their own books: replace its contents with {}
#    (fixes for ids not in your library are ignored with a warning anyway)

# 4. Download the embedding model (~2.3 GB, once)
uv run python scripts/download_model.py

# 5. Build: free, no API key needed. Gives keyword and meaning search.
uv run books build

# 6. Try it
uv run books library
uv run books search "how do feedback loops cause oscillation"
uv run books read <a passage id from the results>
```

### Add the summaries (optional, paid)

Put your key in a file named `.env` in the project folder (it's gitignored and never printed), with one line:
`ANTHROPIC_API_KEY=sk-ant-...`. Or set it as an environment variable. Then:

```bash
uv run books build --yes
```

This runs Claude Haiku over your books. It describes figures, writes a summary card per chapter and per book, and
then builds a concept map that links ideas across books. Without `--yes`, `books build` prints what this would cost
for your library. **For reference:** 24 books (~3 million words, 857 figures) cost **$1.96**. Re-runs only pay for
what changed.

### Use it from Claude Code

The repo ships a **Claude Code skill**: instructions that Claude
loads by itself whenever you ask about your books. You can read it before installing:
**[`src/bookshelf/skill_template.md`](src/bookshelf/skill_template.md)**. It covers:

- **The workflow:** shelf → search → chapter card → exact passage, and when to use each command.
- **The rules:** quote only text returned by `books read`, cite every claim with its passage id, treat machine-written
  summaries as navigation (never as the author's words), and say so when the library doesn't cover something.
- **When to start the server** for meaning and cross-language search.

It's a template because the finished skill holds things that are yours: this folder's path, your book list (so Claude
knows when to use it) and your non-English books. One command fills them in:

```bash
uv run books skill            # writes .claude/skills/books/SKILL.md for your library
uv run books skill --global   # ...and installs it for every Claude Code session
uv run books serve            # keep the model loaded: meaning search in ~0.1 s
```

Then just ask Claude Code about your books, or type `/books`. Re-run `books skill` after adding books, so its book list
stays current. Without `books serve`, search falls back to keyword-only and says so. Starting the server takes about a minute, because loading PyTorch and the
model is the slow part.

### Browse it as a bookshelf

```bash
uv run books site --open           # writes data/site/index.html and opens it
uv run books site --serve --open   # same, served on 127.0.0.1:8790 so reading status is saved
```

A static page in the spirit of [Stripe Press](https://stripe.press): your books as 3D volumes on a shelf, with spine
colours picked from each cover and thickness from the word count. Hover (or use ← →) to pull a book out; rest the mouse near either edge and the shelf glides that way. Click to open
it: the cover swings open onto a one-page overview with the thesis, what the book is good for, its main topics and
the books it shares ideas with. Everything is inlined, so it works straight from disk.
Mark each book unread, reading or read: a ribbon bookmark sticks out of the book you're reading, and read books
turn brass. With `--serve` your choices are saved to `reading.json` (gitignored); opened from disk, they stay in that
browser only. The page lives under `data/` (gitignored), like the rest of your library's derived data. Re-run after
`books build`.
Don't like a cover? Give the book a `"cover_url"` in `metadata_overrides.json`, run `books catalog`, then `books site`.

### Add a book later

Add it in Calibre (and copy the library again if you use `./books/`), then:

```bash
uv run books build NEW_BOOK_ID --yes   # `uv run books catalog` then `uv run books list` shows the new id
```

## Commands

| Command | What it does |
|---|---|
| `books build [ID...] [--yes] [--no-embed]` | The whole pipeline. Re-runs skip anything unchanged |
| `books library [--full]` | The shelf: each book's thesis (and what it's a good source for) |
| `books book BOOK` | Book summary and chapter list. `BOOK` can be any unambiguous part of the id, title or author |
| `books chapter BOOK:N` | Chapter card: summary, key points with passage ids, concepts, figures |
| `books search "question" [-b BOOK] [-k N] [--per-book N]` | Hybrid search. `--per-book 2` for comparative questions |
| `books read PASSAGE_ID [-C N]` | The exact text, nearby passages, and a ready-made citation |
| `books related "concept"` | Where an idea appears across books |
| `books figure BOOK:fig-NNN` | What a chart, table or diagram shows, as data |
| `books serve [--status\|--stop]` | Local server that keeps the model warm (127.0.0.1 only) |
| `books skill [--global] [--print]` | Write the Claude Code skill for this library (`--print` to preview it) |
| `books site [--open] [--serve]` | Build the bookshelf web page into `data/site/` (`--serve` saves reading status) |
| `books eval questions\|judge\|run` | Measure search quality (see below) |

Every agent-facing command prints compact text by default (cheaper for an agent to read) or JSON with `--json`.
The step-by-step commands (`catalog`, `extract`, `chunk`, `index`, `enrich`, `concepts`) still exist if you want to
run or inspect one stage at a time. See `uv run books --help`.

## How it works

| Stage | What happens | Output (in `data/`) |
|---|---|---|
| 0. Catalog | Reads Calibre's `metadata.db` (read-only) | `catalog.json` |
| 1. Extract | Tries every format of each book, scores how well each splits into chapters, keeps the best. Saves figures; turns MathML into LaTeX | `books/<id>/sections/*.md`, `figures/` |
| 2. Chunk | ~400-word passages that never cross a chapter. Figures stay with their captions | `books/<id>/chunks.jsonl` |
| 3. Index | SQLite FTS5 (keyword) + [bge-m3](https://huggingface.co/BAAI/bge-m3) vectors (meaning), merged with Reciprocal Rank Fusion | `index.sqlite` |
| 4. Enrich *(paid)* | Claude Haiku: figure data, chapter and book cards (structured JSON) | `books/<id>/enrich/` |
| 5. Access | `library.py` holds the tools; the CLI, the server and the skill are thin layers over it | |
| 6. Evaluate | A frozen test set, hit@k / MRR, an LLM judge | `evals/` |

The design choices (why one SQLite file, why chunks never cross chapters, why a small model writes the summaries)
are explained in the build log, [`docs/workflow.html`](docs/workflow.html).

### Use it from other agents or code

- **Any agent with a shell:** the CLI works the same way (`--json` for programs).
- **Python:** `from bookshelf import library; library.search("...")`. Every tool returns plain dicts.
- **HTTP** (while `books serve` runs): `POST http://127.0.0.1:8765/tool/<name>` with JSON arguments. Requests from web
  browsers are refused.

## Evaluate search on your library

```bash
uv run books eval questions --yes   # Haiku writes 6 test questions per book (a few cents)
uv run books eval judge --yes       # Haiku grades the top results, blind to the expected answer (cents)
uv run books eval run               # free: scores keyword / semantic / hybrid -> data/evals/report.json
```

Re-run `books eval run` after any change to search, and compare. On the author's 24 books, hybrid search puts the
right passage in the top 5 **88%** of the time and in the top 10 **92%** of the time. Synthetic questions are cleaner
than real ones, so add your own questions for a realistic picture.

## Configuration

| Variable | Default | |
|---|---|---|
| `BOOKS_LIBRARY` | `./books` | Your Calibre library folder (read-only) |
| `BOOKS_DATA` | `./data` | Where everything derived is written. Delete it to start over |
| `ANTHROPIC_API_KEY` | read from `.env` | Only for the paid steps and evals |

Fix wrong Calibre metadata (authors, titles, languages) in `metadata_overrides.json`, keyed by book id. The language
matters: it decides which books get cross-language test questions.

## Privacy and copyright

- `data/` holds the **full text of your books**. It's gitignored: never commit it or publish it. The same goes for
  your Calibre library.
- Search and embeddings run locally. Text leaves your machine only in the optional Haiku steps (`build --yes`,
  `eval questions`, `eval judge`), which send passages to the Anthropic API.
- The tool only reads DRM-free files. It doesn't remove DRM.

## Limitations

- Scanned PDFs need a text layer (OCR); there is no OCR engine built in.
- The summaries are written by a small model and can be wrong (for example, it misnumbered some chapters). The
  tools use them for navigation only, and the skill tells agents to quote from the text.
- Some editions put sidebars and pull-quotes inside the main text, and they get extracted along with it.
- Single user, local. The server is a convenience, not a service. There's one per machine (port 8765): if you keep
  two copies, a copy only uses the server it started itself, and otherwise falls back to keyword search with a note.
- No MCP server yet: Claude Code uses the CLI through the skill. A regular Claude chat without a shell can't use it.

## License

[MIT](LICENSE) for the code. The license covers this software only: it gives no rights to any book you process
with it.
