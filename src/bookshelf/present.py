"""Compact text for each library tool: what an agent reads by default.

Plain text costs far fewer tokens than the same data as indented JSON, and
models read it just as well. Every id is printed in full, so it can be copied
straight into the next command. `--json` gives the raw dicts instead.
"""


def books(r: dict, full: bool = False) -> str:
    out = [f"{len(r['books'])} books. {r['note']}", ""]
    for b in r["books"]:
        out.append(f"{b['id']} | {b['title']} | {', '.join(b['authors'])} | {b['language']} | "
                   f"{b['words']:,} words, {b['chapters']} chapters")
        if b["thesis"]:
            out.append(f"  Thesis: {b['thesis']}")
        if full and b["use_for"]:
            out.append(f"  Use for: {'; '.join(b['use_for'])}")
    more = "" if full else " Add --full for what each book is a good source for."
    out.append(f"\nNext: `books book ID` for chapters, or `books search \"question\"`.{more}")
    return "\n".join(out)


def book(r: dict) -> str:
    out = [f"{r['title']} - {', '.join(r['authors'])} ({r['id']}, {r['language']}, {r['words']:,} words)"]
    for label, key in (("Thesis", "thesis"), ("Summary", "summary")):
        if r[key]:
            out.append(f"{label}: {r[key]}")
    if r["top_concepts"]:
        out.append(f"Top concepts: {', '.join(r['top_concepts'])}")
    if r["use_for"]:
        out.append(f"Use for: {'; '.join(r['use_for'])}")
    out += ["", f"Chapters (open with `books chapter {r['id']}:N`):"]
    for c in r["chapters"]:
        tag = "" if c["kind"] == "body" else f" [{c['kind']}]"
        title = c["title"].split(" > ")[-1]
        gist = f" - {c['one_line']}" if c["one_line"] else ""
        out.append(f"{c['section']:>4}  {'  ' * (c['depth'] - 1)}{title}{tag} ({c['words']:,} w){gist}")
    out.append(f"\n{r['note']}")
    return "\n".join(out)


def chapter(r: dict) -> str:
    pages = f", pp. {r['pages'][0]}-{r['pages'][1]}" if r["pages"] else ""
    out = [f"{r['book']}:{r['section']} | {r['title']} | {r['book_title']} ({r['words']:,} words{pages})"]
    card = r["card"]
    if card:
        out += [f"Summary ({card['content_type']}): {card['summary']}", "", "Key points:"]
        out += [f"  - {kp['point']} [{', '.join(kp['chunks'])}]" for kp in card["key_points"]]
        if card["concepts"]:
            out.append("Concepts: " + ", ".join(f"{c['name']} ({c['role']})" for c in card["concepts"]))
        if card["entities"]:
            out.append("Entities: " + ", ".join(f"{e['name']} ({e['kind']})" for e in card["entities"]))
        if card["data_points"]:
            out.append("Data:")
            out += [f"  - {d['fact']}: {d['value']} [{d['chunk']}]" for d in card["data_points"]]
        if card["questions"]:
            out.append("Questions it answers:")
            out += [f"  - {q}" for q in card["questions"]]
    elif r.get("hint"):
        out.append(r["hint"])
    else:
        out.append("No card for this section.")
    if r["figures"]:
        out.append("Figures (`books figure REF`):")
        out += [f"  - {f['ref']} {f['type']}: {f['title']}" for f in r["figures"]]
    ch = r["chunks"]
    if ch:
        out.append(f"Passages ({len(ch)}): {ch[0]} .. {ch[-1]}  (`books read ID`)")
    out.append(f"Prev: {r['prev'] or '-'}   Next: {r['next'] or '-'}")
    if card:
        out.append(r["note"])
    return "\n".join(out)


def search(r: dict) -> str:
    out = []
    if r.get("note"):
        out += [f"NOTE: {r['note']}", ""]
    if not r["results"]:
        out.append("No results. Try other words, a broader question, or drop --book.")
    for i, h in enumerate(r["results"], 1):
        pages = f" | p. {h['pages'][0]}" if h["pages"] else ""
        out.append(f"{i}. {h['id']} | {h['title']} | {h['chapter']}{pages}")
        if h["chapter_gist"]:
            out.append(f"   Chapter {h['chapter_ref']}: {h['chapter_gist']}")
        out.append(f"   {h['snippet']}")
    if r["results"]:
        out.append("\nSnippets are fragments. Open a passage with `books read ID` before quoting it.")
    return "\n".join(out)


def read(r: dict) -> str:
    out = []
    for p in r["passages"]:
        pages = f" | p. {p['pages'][0]}" + (f"-{p['pages'][1]}" if p["pages"][1] != p["pages"][0] else "") \
            if p["pages"] else ""
        out += [f"=== {p['id']} | {p['chapter']}{pages}", f"Cite as: {p['citation']}", "", p["text"], ""]
    out.append(f"Prev: {r['prev'] or '-'}   Next: {r['next'] or '-'}")
    if r.get("note"):
        out.append(f"NOTE: {r['note']}")
    return "\n".join(out)


def related(r: dict) -> str:
    if not r["concepts"]:
        return r["note"]
    out = []
    for c in r["concepts"]:
        roles = ", ".join(f"{k} {v}" for k, v in sorted(c["roles"].items(), key=lambda x: -x[1]))
        out.append(f"{c['concept']} ({c['books']} books, {c['mentions']} mentions: {roles})")
        if c["aliases"]:
            out.append(f"  aka: {', '.join(c['aliases'][:10])}")
        for w in c["where"]:
            chs = "; ".join(f"{x['ref']} {x['title'].split(' > ')[-1]}" for x in w["chapters"][:6])
            more = f" (+{len(w['chapters']) - 6} more)" if len(w["chapters"]) > 6 else ""
            out.append(f"  {w['title']}: {chs}{more}")
    out.append("\nOpen a chapter with `books chapter BOOK:N` to see how the idea is used there.")
    return "\n".join(out)


def figure(r: dict) -> str:
    out = [f"{r['ref']} | {r['book_title']} | {r['chapter']} ({r['chapter_ref']})"]
    if r["caption"]:
        out.append(f"Caption (from the book): {r['caption']}")
    d = r["description"]
    if d:
        out += [f"Type: {d['type']}. {d['title']}", f"Shows: {d['summary']}", "Data:", d["data"]]
        if d.get("text_in_image"):
            out.append(f"Text in image: {d['text_in_image']}")
    out += [f"In passage: {r['in_passage'] or '-'}", f"Image file: {r['image']}", f"NOTE: {r['note']}"]
    return "\n".join(out)
