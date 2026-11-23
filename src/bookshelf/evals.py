"""Phase 6: measure search quality instead of guessing.

An eval has three parts:
  1. A test set: questions whose correct answer location we know.
  2. Metrics: numbers that say how often, and how high, search puts that answer.
  3. Baselines and comparisons: every change is scored against the same set.

Test set (built once, then frozen)
  We sample passages (6 per book, fixed seed) and ask Claude Haiku to write one
  question that each passage answers. The passage is then the known answer
  ("synthetic" labels: cheap, but only as good as the questions, so we read them).
  Question styles:
    lookup      may use the passage's own names and terms (a half-remembered fact)
    paraphrase  the same idea in different everyday words (tests meaning search)
    crosslang   English question about a Portuguese passage
  Each question is assigned to dev (~60%, used to try ideas) or test (~40%,
  looked at only to confirm). Tuning on the same questions you report would
  flatter the result.

Metrics (per question, then averaged)
  hit@k     the target passage is in the top k results
  MRR@10    1/rank of the target (1, 0.5, 0.33 ... or 0 beyond 10): rewards being near the top
  near@5    the target or a neighbouring passage of the same chapter is in the top 5
  book@5    a passage from the right book is in the top 5

Judge (catches "false misses")
  Another passage may answer the question as well as the target. Haiku grades
  the top 5 passages of each mode (blind: it doesn't know which one is the
  target). answered@5 = some top-5 passage fully answers. Its verdict on the
  target itself checks the judge: it should say yes or partly almost always.

Files: data/evals/{questions,judgments,usage}.jsonl, report.json
"""

import hashlib
import json
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from . import embed as emb
from . import enrich as en
from . import index as ix
from .catalog import load_catalog
from .chunk import load_chunks
from .config import DATA

EVAL_DIR = DATA / "evals"
QUESTIONS_FILE = EVAL_DIR / "questions.jsonl"
JUDGE_FILE = EVAL_DIR / "judgments.jsonl"
USAGE_FILE = EVAL_DIR / "usage.jsonl"
REPORT_FILE = EVAL_DIR / "report.json"

N_PER_BOOK = 6
SEED = 6
MIN_WORDS = 150   # passages shorter than this rarely answer a real question
K = 10
QUESTION_VERSION = 2  # v1: shared seed, dropped all blockquotes, allowed "this scene"

# ---------------------------------------------------------------- 1. test set

SYSTEM = ("You write and grade test questions for evaluating a search engine over a personal library of books, "
          "some possibly not in English. Follow the instructions exactly and answer only with the JSON asked for.")

STYLE_RULES = {
    "lookup": "Ask about a specific fact, claim, event or example in the passage. You MAY use its names, "
              "terms and numbers, the way a reader who half-remembers it would.",
    "paraphrase": "Ask about the passage's main idea in your own everyday words. Do NOT reuse its distinctive "
                  "words, names, terms or phrases; describe them instead. The question must still have this "
                  "passage as a good answer.",
    "crosslang": "The passage is not in English. Write the question in ENGLISH, about its main idea, in your own "
                 "words (don't just translate a sentence).",
}

QUESTION_TASK = """Book: "{title}" by {authors}
Chapter: {chapter}

PASSAGE:
{text}

Write ONE question that a reader of this library (about {n_books} books) might type into a search box, and that this passage answers well.
Style: {style}. {rule}
Rules:
- It must make sense on its own, to someone who has not seen the passage: never say "the passage", "this text",
  "this scene", "the author" or "the narrator". The library has many books, so don't write "the company", "the
  protagonist" or "the country" without saying which: name the person, company, place or topic instead.
- Do not name the book or its author.
- One sentence, under 25 words.
- "language": the question's ISO 639-1 code ("en", "pt", ...). Use the passage's language only if the style allows it.
Also return:
- "answer": the answer, in one short sentence, as the passage gives it.
- "specificity": "specific" if this passage would be among the few best answers in the whole library;
  "generic" if many passages in many books would answer it equally well.
- "usable": false if the passage has no answerable content (a list of names, a table of contents, a bare transition)."""

QUESTION_SCHEMA = en._obj({
    "question": en._STR,
    "language": en._STR,
    "answer": en._STR,
    "specificity": {"type": "string", "enum": ["specific", "generic"]},
    "usable": {"type": "boolean"},
})


_FIG_NOTE = re.compile(r"^>\s*\*\*\[fig-\d+ · ")


def author_text(text: str) -> str:
    """The author's words only: drop image links and Haiku's figure-description blocks.

    Only blocks that start with `> **[fig-NNN · type]` are dropped (they run until the
    next blank line). Plain blockquotes stay: some EPUBs (Cybernetic Revolutionaries)
    wrap the whole body text in <blockquote>."""
    out, skipping = [], False
    for line in text.splitlines():
        if _FIG_NOTE.match(line.lstrip()):
            skipping = True
        elif skipping and not line.strip():
            skipping = False
        if not skipping:
            out.append(line)
    return re.sub(r"!\[[^\]]*\]\([^)]*\)", "", "\n".join(out)).strip()


def _split(chunk_id: str) -> str:
    return "dev" if int(hashlib.sha1(chunk_id.encode()).hexdigest()[:8], 16) % 10 < 6 else "test"


def sample_passages() -> list[dict]:
    """N_PER_BOOK random body passages per book. Each book has its own seed, so adding or
    fixing one book never changes which passages were picked for the others."""
    out = []
    for b in load_catalog():
        rng = random.Random(f"{SEED}:{b['id']}")
        chunks = sorted((c for c in load_chunks(b["id"])
                         if c["kind"] == "body" and len(author_text(c["text"]).split()) >= MIN_WORDS),
                        key=lambda c: c["id"])
        if not chunks:
            continue
        lang = (b["languages"] or ["eng"])[0]
        styles = ["paraphrase", "lookup"] if lang in ("eng", "en") else ["crosslang", "lookup"]
        for i, c in enumerate(rng.sample(chunks, min(N_PER_BOOK, len(chunks)))):
            out.append({"chunk": c["id"], "book": b["id"], "title": b["title"], "authors": b["authors"],
                        "chapter": " > ".join(c["path"]), "lang": lang, "style": styles[i % 2],
                        "split": _split(c["id"]), "text": author_text(c["text"])})
    return out


def _ask(requests: list[dict], kind: str, workers: int = 8, progress=None) -> dict[str, dict]:
    """requests: {key, task, schema, max_tokens, book}. Direct Haiku calls, usage logged to data/evals/usage.jsonl."""
    api = en.client()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out, cost = {}, 0.0
    with ThreadPoolExecutor(workers) as pool, USAGE_FILE.open("a", encoding="utf-8") as log:
        futs = {pool.submit(api.messages.create, model=en.MODEL, max_tokens=r["max_tokens"], system=SYSTEM,
                            output_config={"effort": "low", "format": {"type": "json_schema", "schema": r["schema"]}},
                            messages=[{"role": "user", "content": r["task"]}]): r for r in requests}
        for n, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            try:
                msg = fut.result()
            except Exception as exc:  # the SDK already retried 429/5xx
                out[r["key"]] = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
                continue
            u = msg.usage
            tin = u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
            cost += en._cost(tin, u.output_tokens, "now")
            log.write(json.dumps({"kind": kind, "key": r["key"], "input": tin, "output": u.output_tokens,
                                  "stop": msg.stop_reason}) + "\n")
            text = next((b.text for b in msg.content if b.type == "text"), "")
            try:
                out[r["key"]] = json.loads(text) if msg.stop_reason == "end_turn" else {"error": msg.stop_reason}
            except json.JSONDecodeError:
                out[r["key"]] = {"error": "bad json"}
            if progress:
                progress(n, len(requests), cost)
    return out


def load_questions() -> list[dict]:
    if not QUESTIONS_FILE.exists():
        return []
    return [json.loads(l) for l in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines() if l]


def build_questions(send: bool = False, progress=None) -> dict:
    """Write one question per sampled passage. Passages that already have one are skipped."""
    have = {q["chunk"] for q in load_questions() if q["version"] == QUESTION_VERSION}
    todo = [p for p in sample_passages() if p["chunk"] not in have]
    words = sum(len(p["text"].split()) for p in todo)
    est = en._cost(int(words * 1.6) + 450 * len(todo), 250 * len(todo), "now")
    if not send or not todo:
        return {"pending": len(todo), "estimated_usd": round(est, 3)}
    n_books = round(len({p["book"] for p in sample_passages()}), -1) or 10
    reqs = [{"key": p["chunk"], "max_tokens": 1500, "schema": QUESTION_SCHEMA,
             "task": QUESTION_TASK.format(n_books=n_books, title=p["title"], authors=", ".join(p["authors"]),
                                          chapter=p["chapter"],
                                          text=p["text"], style=p["style"], rule=STYLE_RULES[p["style"]])}
            for p in todo]
    res = _ask(reqs, "question", progress=progress)
    failed = []
    with QUESTIONS_FILE.open("a", encoding="utf-8") as f:
        for p in todo:
            r = res.get(p["chunk"], {"error": "missing"})
            if "error" in r:
                failed.append(f"{p['chunk']}: {r['error']}")
                continue
            q = {k: p[k] for k in ("chunk", "book", "chapter", "lang", "style", "split")}
            f.write(json.dumps({"id": p["chunk"], **q, **r, "version": QUESTION_VERSION}, ensure_ascii=False) + "\n")
    return {"written": len(todo) - len(failed), "failed": failed}


# Questions that point at the source ("this novel", "the narrator") can't be answered from
# the question alone. The rule runs over EVERY question, never only over the ones search
# failed: dropping just the failures you happen to read inflates the score.
_POINTS_AT_SOURCE = re.compile(r"\b(this|the) (passage|text|scene|novel|book|story|excerpt)\b|\bnarrator\b"
                               r"|\bin a (story|novel)\b", re.I)


def lint(q: dict) -> str | None:
    if not q.get("usable"):
        return "unusable passage"
    if q.get("specificity") != "specific":
        return "generic question"
    if _POINTS_AT_SOURCE.search(q["question"]):
        return "points at the source text"
    return None


def eval_set(questions: list[dict] | None = None) -> list[dict]:
    """The questions that count: those that pass lint()."""
    qs = questions if questions is not None else load_questions()
    return [q for q in qs if lint(q) is None]


# ---------------------------------------------------------------- 2. retrieve + rank


def candidates(questions: list[dict]) -> tuple[dict, dict]:
    """For every question, the top-50 keyword and top-50 semantic rowids (computed once, then fused many ways)."""
    db = ix._connect(ix.INDEX_FILE)
    rows = {r["rowid"]: (r["id"], r["book"], r["section"], r["seq"])
            for r in db.execute("SELECT rowid, id, book, section, seq FROM chunks")}
    vecs = emb.encode([q["question"] for q in questions], batch_size=16)
    out = {}
    for q, v in zip(questions, vecs):
        kw = [r for r, _, _ in ix._keyword(db, q["question"], None, ["body"], ix.POOL)]
        sem = [r["rowid"] for r in db.execute(
            "SELECT rowid FROM chunk_vec WHERE embedding MATCH ? AND k = ? AND kind = 'body' ORDER BY distance",
            (v.astype(np.float32).tobytes(), ix.POOL))]
        out[q["id"]] = {"keyword": kw, "semantic": sem}
    db.close()
    return out, rows


def rank(cands: dict, rows: dict, mode: str, w_kw: float = 1.0, w_sem: float = 1.0, cap: int = 0) -> list[str]:
    """Top-K chunk ids for one question. hybrid = weighted Reciprocal Rank Fusion, ties toward semantic
    (same as index.search when weights are 1:1); cap = max results per book (0 = none)."""
    if mode in ("keyword", "semantic"):
        order = cands[mode]
    else:
        score, sem_rank = defaultdict(float), {}
        for name, w in (("keyword", w_kw), ("semantic", w_sem)):
            for r, rowid in enumerate(cands[name], 1):
                score[rowid] += w / (ix.RRF_K + r)
                if name == "semantic":
                    sem_rank[rowid] = r
        order = sorted(score, key=lambda x: (-score[x], sem_rank.get(x, ix.POOL + 1)))
    out, per_book = [], defaultdict(int)
    for rowid in order:
        cid, book = rows[rowid][0], rows[rowid][1]
        if cap and per_book[book] >= cap:
            continue
        per_book[book] += 1
        out.append(cid)
        if len(out) == K:
            break
    return out


def score(q: dict, ranked: list[str], verdicts: dict | None = None) -> dict:
    target = q["chunk"]
    book, sec, seq = target.rsplit(":", 2)
    r = ranked.index(target) + 1 if target in ranked else None
    near = any(c.rsplit(":", 2)[:2] == [book, sec] and abs(int(c.rsplit(":", 2)[2]) - int(seq)) <= 1
               for c in ranked[:5])
    s = {"rank": r, "hit@1": r == 1, "hit@5": bool(r and r <= 5), "hit@10": bool(r),
         "mrr": 1 / r if r else 0.0, "near@5": near, "book@5": any(c.startswith(book + ":") for c in ranked[:5])}
    if verdicts is not None:
        s["answered@5"] = bool(r and r <= 5) or any(verdicts.get(c) == "yes" for c in ranked[:5])
    return s


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k != "rank"]
    return {"n": len(rows), **{k: round(sum(float(x[k]) for x in rows) / len(rows), 3) for k in keys}}


# ---------------------------------------------------------------- 3. judge

JUDGE_TASK = """Question: {question}

Below are passages from books, labelled P1, P2, ... For EACH passage, say whether a reader could answer the question from that passage:
- "yes": it answers the question (fully, on its own)
- "partly": relevant and gives part of the answer, or the answer needs more context
- "no": does not help answer it
Judge each passage independently, by its content only.

{passages}"""

JUDGE_SCHEMA = en._obj({"verdicts": {"type": "array", "items": en._obj({
    "label": en._STR, "verdict": {"type": "string", "enum": ["yes", "partly", "no"]}})}})


def load_judgments() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = defaultdict(dict)
    if JUDGE_FILE.exists():
        for l in JUDGE_FILE.read_text(encoding="utf-8").splitlines():
            if l:
                j = json.loads(l)
                out[j["question"]].update(j["verdicts"])
    return out


def judge(pools: dict[str, list[str]], questions: dict[str, dict], send: bool = False, progress=None) -> dict:
    """pools: question id -> chunk ids to grade (each question's target is always added).
    Only pairs not judged before are sent; passages are shuffled so position can't hint at the target."""
    done = load_judgments()
    db = ix._connect(ix.INDEX_FILE, vectors=False)
    reqs, labels = [], {}
    for qid, pool in pools.items():
        todo = [c for c in dict.fromkeys(pool + [qid]) if c not in done.get(qid, {})]
        if not todo:
            continue
        random.Random(qid).shuffle(todo)
        lab = {f"P{i}": c for i, c in enumerate(todo, 1)}
        texts = []
        for l, c in lab.items():
            t = db.execute("SELECT text FROM chunks WHERE id = ?", (c,)).fetchone()["text"]
            texts.append(f"[{l}]\n{author_text(t)}")
        labels[qid] = lab
        reqs.append({"key": qid, "max_tokens": 2000, "schema": JUDGE_SCHEMA,
                     "task": JUDGE_TASK.format(question=questions[qid]["question"], passages="\n\n".join(texts))})
    db.close()
    words = sum(len(r["task"].split()) for r in reqs)
    est = en._cost(int(words * 1.6) + 150 * len(reqs), 40 * sum(len(l) for l in labels.values()), "now")
    if not send or not reqs:
        return {"requests": len(reqs), "pairs": sum(len(l) for l in labels.values()), "estimated_usd": round(est, 3)}
    res = _ask(reqs, "judge", progress=progress)
    failed = []
    with JUDGE_FILE.open("a", encoding="utf-8") as f:
        for qid, lab in labels.items():
            r = res.get(qid, {"error": "missing"})
            if "error" in r:
                failed.append(f"{qid}: {r['error']}")
                continue
            v = {lab[x["label"]]: x["verdict"] for x in r["verdicts"] if x["label"] in lab}
            f.write(json.dumps({"question": qid, "verdicts": v}, ensure_ascii=False) + "\n")
    return {"judged_questions": len(labels) - len(failed), "failed": failed}


# ---------------------------------------------------------------- 4. report (re-run after every search change)

# What `books eval run` scores. "hybrid" must match the library's default search.
CONFIGS = {
    "keyword": dict(mode="keyword"),
    "semantic": dict(mode="semantic"),
    "hybrid": dict(mode="hybrid"),
    "hybrid, 3 per book": dict(mode="hybrid", cap=3),
}


def judge_pools(qs, cands, rows) -> dict[str, list[str]]:
    return {q["id"]: [c for cfg in CONFIGS.values() for c in rank(cands[q["id"]], rows, **cfg)[:5]] for q in qs}


def report(save: bool = True) -> dict:
    import datetime

    qs = eval_set()
    cands, rows = candidates(qs)
    J = load_judgments()
    unjudged = sum(1 for qid, pool in judge_pools(qs, cands, rows).items() for c in set(pool) if c not in J.get(qid, {}))
    out = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "questions": len(qs), "split": {s: sum(q["split"] == s for q in qs) for s in ("dev", "test")},
           "styles": {s: sum(q["style"] == s for q in qs) for s in sorted({q["style"] for q in qs})},
           "unjudged_pairs": unjudged, "spent": spent(), "modes": {}, "by_style": {}}
    for name, cfg in CONFIGS.items():
        ranked = {q["id"]: rank(cands[q["id"]], rows, **cfg) for q in qs}
        out["modes"][name] = {split: summarise([score(q, ranked[q["id"]], J.get(q["id"], {})) for q in qs
                                                if split == "all" or q["split"] == split])
                              for split in ("dev", "test", "all")}
        out["by_style"][name] = {st: summarise([score(q, ranked[q["id"]]) for q in qs if q["style"] == st])
                                 for st in out["styles"]}
    if save:
        REPORT_FILE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def load_report() -> dict:
    return json.loads(REPORT_FILE.read_text(encoding="utf-8")) if REPORT_FILE.exists() else {}


def spent() -> dict:
    if not USAGE_FILE.exists():
        return {"usd": 0.0}
    rows = [json.loads(l) for l in USAGE_FILE.read_text(encoding="utf-8").splitlines() if l]
    by = defaultdict(float)
    for r in rows:
        by[r["kind"]] += en._cost(r["input"], r["output"], "now")
    return {"usd": round(sum(by.values()), 3), **{k: round(v, 3) for k, v in by.items()}}
