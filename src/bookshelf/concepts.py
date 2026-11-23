"""Phase 4d: concept map across books, from the chapter cards (no LLM calls).

1. Collect every concept name from every chapter card, with book, section, role.
2. Embed the distinct names with the local bge-m3 model.
3. Group names that mean the same thing (cosine similarity >= THRESHOLD),
   greedily, most frequent name first; the most frequent name labels the group.
4. Write data/concepts.json: each concept with its aliases and where it occurs,
   sorted so concepts shared by the most books come first. These cross-book
   concepts are the links a research agent can follow between books.
"""

import json
from collections import Counter, defaultdict

import numpy as np

from . import embed as emb
from .catalog import load_catalog
from .config import DATA
from .enrich import load_results

# bge-m3 cosine. At 0.86 "leverage" merged with "leverage point" (0.867), with the
# same score as the good merge "feedback loop" ~ "reinforcing feedback loop" (0.865).
# A false link misleads an agent more than a missed one, so stay strict: 0.90 keeps
# spelling variants (0.94-0.96) together and keeps distinct ideas apart.
THRESHOLD = 0.90
CONCEPTS_FILE = DATA / "concepts.json"


def _normalise(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").split()).strip(" .,;:")


def build(threshold: float = THRESHOLD) -> dict:
    mentions = []  # (name, book, section, role)
    for b in load_catalog():
        for card in load_results(b["id"], "chapter").values():
            for c in card["card"]["concepts"]:
                name = _normalise(c["name"])
                if name:
                    mentions.append((name, b["id"], card["meta"]["section"], c["role"]))
    if not mentions:
        return {"concepts": 0}

    freq = Counter(m[0] for m in mentions)
    names = [n for n, _ in freq.most_common()]
    vecs = emb.encode(names, batch_size=64)  # normalised, so dot product = cosine

    canon_of: dict[str, str] = {}
    centers: list[np.ndarray] = []
    labels: list[str] = []
    for i, name in enumerate(names):
        if centers:
            sims = np.stack(centers) @ vecs[i]
            j = int(sims.argmax())
            if sims[j] >= threshold:
                canon_of[name] = labels[j]
                continue
        centers.append(vecs[i])
        labels.append(name)
        canon_of[name] = name

    groups: dict[str, dict] = {}
    for name, book, section, role in mentions:
        g = groups.setdefault(canon_of[name], {"aliases": set(), "where": defaultdict(set), "roles": Counter()})
        g["aliases"].add(name)
        g["where"][book].add(section)
        g["roles"][role] += 1

    concepts = []
    for label, g in groups.items():
        concepts.append({
            "concept": label,
            "aliases": sorted(g["aliases"] - {label}),
            "books": len(g["where"]),
            "mentions": sum(g["roles"].values()),
            "roles": dict(g["roles"]),
            "where": {book: sorted(secs) for book, secs in sorted(g["where"].items())},
        })
    concepts.sort(key=lambda c: (-c["books"], -c["mentions"], c["concept"]))
    payload = {"threshold": threshold, "model": f"{emb.MODEL} {emb.precision()}",
               "names": len(names), "concepts": concepts}
    CONCEPTS_FILE.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    shared = [c for c in concepts if c["books"] >= 2]
    return {"distinct_names": len(names), "concepts": len(concepts), "cross_book": len(shared)}


def load() -> list[dict]:
    if not CONCEPTS_FILE.exists():
        return []
    return json.loads(CONCEPTS_FILE.read_text(encoding="utf-8"))["concepts"]
