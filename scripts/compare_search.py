"""Run the same questions through keyword, semantic and hybrid search and
report whether the expected book shows up in the top results.

    uv run python scripts/compare_search.py

Writes data/evals/search_compare.json (chunk ids and ranks only, no book text).
A first, small taste of Phase 7 (evaluation).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bookshelf import index as ix  # noqa: E402

TOP = 5
# (question, expected book, what it tests)
QUESTIONS = [
    ("Medallion fund fees", "the-man-who-solved-the-market", "exact names"),
    ("Cybersyn operations room design", "cybernetic-revolutionaries", "exact names"),
    ('"invert, always invert"', "poor-charlies-almanack", "exact quote"),
    ("why do systems overshoot and oscillate", "thinking-in-systems", "shared words"),
    ("how delays in a feedback loop make a system unstable", "thinking-in-systems", "paraphrase"),
    ("the pressure to constantly perform and optimise yourself", "the-disappearance-of-rituals", "paraphrase"),
    ("financial bubbles during the installation of a new technology", "technological-revolutions-and-financial-capital", "paraphrase"),
    ("central banks as the source of global liquidity", "capital-wars", "concept"),
    ("status games between actors on stage", "impro", "concept"),
    ("how a trading company turned into a military power", "the-anarchy", "paraphrase"),
    ("loneliness and the feeling of not belonging anywhere", "o-livro-do-desassossego", "English query, Portuguese book"),
    ("como compradores e vendedores se encontram em mercados de balcão", "the-economics-of-over-the-counter", "Portuguese query, English book"),
]


def main() -> None:
    results = []
    totals = {"keyword": 0, "semantic": 0, "hybrid": 0}
    for q, expected, tests in QUESTIONS:
        row = {"question": q, "expected": expected, "tests": tests, "modes": {}}
        line = f"{q[:55]:<56}"
        for mode in totals:
            hits = ix.search(q, mode, TOP)
            books = [h.book for h in hits]
            rank = books.index(expected) + 1 if expected in books else None
            totals[mode] += rank is not None
            row["modes"][mode] = {"rank": rank, "top": [h.id for h in hits]}
            line += f"  {mode[:3]}:{('#' + str(rank)) if rank else ' - ':>3}"
        print(line)
        results.append(row)
    n = len(QUESTIONS)
    print("\nexpected book in top", TOP, {m: f"{v}/{n}" for m, v in totals.items()})
    out = ROOT / "data" / "evals" / "search_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"top": TOP, "totals": totals, "results": results}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
