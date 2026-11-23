---
name: books
description: Search, read and cite the user's personal book library ({summary}). Use when the user asks what their books say about a topic, wants quotes or citations from their books, asks to compare authors' ideas, or mentions a book or author they own.
---

# The user's book library

A local, read-only index of the user's books: every book is split into chapters
and ~400-word passages. Each passage has a stable id (`book:section:seq`, e.g.
`{example_id}`) that is its citation key.

Run every command as (any working directory):

```
uv run --project "{project}" books <command>
```

Below, `books` is short for that. Output is compact text, made for you to read.
Add `--json` only when a program needs to parse it.

## Workflow: zoom in, then read

1. **Shelf**: `books library` lists each book's thesis (`--full` adds what each one is a good source for). Skip this if you already know which book.
2. **Find**: `books search "question"`. Add `-b BOOK` to stay in one book, `-k 15` for more results, `--notes` to include endnotes and appendices. Each result shows its chapter's one-line summary, so you know where it landed.
3. **Orient**: `books book BOOK` gives the summary, thesis and chapter list. `books chapter BOOK:N` gives the chapter summary, key points (each with passage ids), concepts, numbers and dates, the questions it answers, and its figures.
4. **Read**: `books read ID` returns the exact text and a `Cite as:` line. Use `-C 1` (or `-B`/`-A`) for surrounding passages, and follow `Prev`/`Next` to keep reading.
5. **Connect**: `books related "concept"` shows where an idea appears across books, with chapter refs to open.
6. **Figures**: `books figure BOOK:fig-NNN` gives a chart's values, a table, a Mermaid diagram or LaTeX.

Usually: search → read the 2–4 best hits → open a chapter only if you need its argument.
For "what does book X argue", start with `books book X`. For cross-book questions, start with `books related` or a search without `-b`.
For "what does the author conclude", a chapter's "Conclusions" section (see `books book X`) is often the best entry point.

## Rules

- **Quote only from `books read`.** Search snippets are fragments. Book, chapter and figure summaries were machine-written by a small model: use them to navigate, never as the author's words.
- **Cite every claim** with the `Cite as:` line, or at least the passage id in brackets, e.g. *Author, Title, “Chapter” [{example_id}]*. Page numbers exist only for PDF books.
- Blockquotes starting `[fig-NNN · type]` inside a passage are machine-written figure descriptions, not the author's words. Some editions also print editors' sidebars and pull-quotes inside the text: check a line is the author's before quoting it.
- If the library doesn't cover something, say so. Don't fill gaps from general knowledge without saying which part is not from the books.
{languages_rule}- Book arguments accept any unambiguous part of the id, title or author (`-b {example_book_ref}`). If a name is ambiguous, the error lists the matches.
- For comparative or cross-book questions ("how do different authors see X?"), add `--per-book 2` so one book can't fill all the results. Leave it off for specific lookups: measured, it pushes the right passage out.

## Meaning search needs the server

If search output starts with `NOTE: Keyword search only`, the warm server isn't
running. Keyword search still works, but it misses paraphrases and cross-language matches.

- Check: `books serve --status`
- Start it in the background (takes about a minute, then every search takes ~0.1 s): `books serve`
- Stop it when done (it holds ~1.2 GB of GPU memory): `books serve --stop`

Start it when a task will involve more than a couple of searches, or any cross-language or "ideas like X" question.
