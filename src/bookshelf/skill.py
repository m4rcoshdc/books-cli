"""`books skill`: write the Claude Code skill for THIS library and THIS folder.

The skill teaches an agent the commands and the citation rules. Its text comes
from skill_template.md; only the library-specific parts are filled in here (the
project path, how many books and which, non-English books, example ids), so a
fork gets a skill about its own books.

  .claude/skills/books/SKILL.md      used by Claude Code sessions in this folder
  ~/.claude/skills/books/SKILL.md    with --global: every Claude Code session
"""

from collections import Counter
from pathlib import Path

from . import library as lib
from .chunk import load_chunks
from .config import PROJECT_ROOT

TEMPLATE = Path(__file__).with_name("skill_template.md")
PROJECT_SKILL = PROJECT_ROOT / ".claude" / "skills" / "books" / "SKILL.md"
GLOBAL_SKILL = Path.home() / ".claude" / "skills" / "books" / "SKILL.md"
LANGUAGES = {"por": "Portuguese", "spa": "Spanish", "fra": "French", "deu": "German", "ita": "Italian",
             "jpn": "Japanese", "zho": "Chinese", "rus": "Russian", "nld": "Dutch"}
ENGLISH = ("eng", "en", "?")


def _short(title: str) -> str:
    return title.split(":")[0].split(" - ")[0].strip()


def render() -> str:
    books = lib.books()["books"]
    if not books:
        raise lib.NotFound("No extracted books yet: run `books build` first.")
    titles = [_short(b["title"]) for b in books]
    listed = "; ".join(titles[:12]) + ("; and more" if len(titles) > 12 else "")
    langs = Counter(LANGUAGES.get(b["language"], "English" if b["language"] in ENGLISH else b["language"])
                    for b in books)
    lang_text = " and ".join(sorted(langs, key=lambda k: -langs[k]))
    foreign = [b for b in books if b["language"] not in ENGLISH]
    languages_rule = ""
    if foreign:
        names = "; ".join(f"*{_short(b['title'])}* ({LANGUAGES.get(b['language'], b['language'])})" for b in foreign[:8])
        languages_rule = (f"- Some books are not in English: {names}. With the server on, English questions find "
                          "their passages. Quote the original and translate it if helpful.\n")
    first = books[0]
    example_id = next((c["id"] for c in load_chunks(first["id"]) if c["kind"] == "body"), f"{first['id']}:001:001")
    surname = (first["authors"] or [""])[0].split()[-1:] or [""]
    ref = surname[0].lower()
    try:
        ref = ref if ref and lib.resolve_book(ref) == first["id"] else first["id"]
    except lib.NotFound:
        ref = first["id"]
    return TEMPLATE.read_text(encoding="utf-8").format(
        summary=f"{len(books)} books in {lang_text}, including {listed}",
        project=PROJECT_ROOT.as_posix(), example_id=example_id, example_book_ref=ref,
        languages_rule=languages_rule)


def install(global_: bool = False) -> list[Path]:
    text = render()
    targets = [PROJECT_SKILL] + ([GLOBAL_SKILL] if global_ else [])
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return targets
