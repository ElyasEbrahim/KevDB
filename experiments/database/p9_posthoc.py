"""Phase 9 post-hoc analysis: everything here was added after seeing the results and is not in the criteria fixed beforehand (experiments/database/protocol/p9.txt).

    python experiments/database/p9_posthoc.py collect   # two extra sets of questions (Jev, about $0.23)
    python experiments/database/p9_posthoc.py report > out/database/p9/posthoc.txt   # calls no model

The two extra sets:
- Q, k=10, shuffled order, both questions asked again unchanged: how often does the same request asked twice give the same judgement? The reordering test has to be compared with this.
- Q, k=1 (one partner per request), opposite only: separates "the second review moved into the question" from "many questions in one request".

A  Agreement when the same request is asked twice: pairwise (the three cold runs of phase 1's day, today vs phase 1), packed (today's two passes), compared with agreement after reordering.
B  Packing with k=1 compared with pairwise and with packing at k=10.
C  Which way packing leans: share judged yes, mean probability on pairs whose truth is yes / no.
D  How well each form agrees with the answer derived from "each review's own sentiment" (phase 1's rewrite).
"""

from __future__ import annotations

import json
import sqlite3
import sys

import numpy as np

import p9_join as J
from kevdb.cache import cache_key
from paths import ROOT

AGAIN = [{"pred": p, "form": "Q", "order": "random-again", "k": 10} for p in J.PREDICATES]  # arrange treats an unknown order name as random, so the requests are identical to the random pass
SINGLE = {"pred": "opposite", "form": "Q", "order": "natural", "k": 1}
PAIR = lambda pred: {"pred": pred, "form": "pair", "order": "natural", "k": 1}
Q = lambda pred, order="random", k=10: {"pred": pred, "form": "Q", "order": order, "k": k}


def collect() -> None:
    backend = J.backend_for("jev")
    for step in (*AGAIN, SINGLE):
        if J.spent_so_far() > J.BUDGET_USD:
            raise SystemExit("over the spending cap fixed beforehand, stopping")
        J.collect(backend, "jev", step)
    backend.close()


def stage1(run: str) -> np.ndarray:
    """Probabilities of phase 1's run (r1, r2, r3) asking opposite pairwise."""
    text = {r[0]: r[1] for r in J.reviews()}
    db = sqlite3.connect(f"file:{ROOT / '.jevcache.sqlite'}?mode=ro", uri=True)
    get = lambda a, b: db.execute("SELECT response FROM answers WHERE key = ?", (cache_key(f"p1.jev.{run}.Q7:jev-1.13.0", *J.build("opposite", "pair", text[a], [text[b]])),)).fetchone()
    return np.array([json.loads(row[0])["answers"]["q"]["noul"] if (row := get(a, b)) else np.nan for a, b in J.pairs()])


def same(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(J.decide(a) == J.decide(b)))


def report() -> None:
    p = lambda step: J.read("jev", step)[0]
    print("Phase 9 post-hoc analysis (added after seeing the results, not in the criteria fixed beforehand)\n")

    print("A  The same request asked twice: share of identical judgements (Jev)")
    runs = {r: stage1(r) for r in ("r1", "r2", "r3")}
    print("   pairwise, opposite, the three cold runs on phase 1's day, two at a time: " + ", ".join(f"{same(runs[a], runs[b]):.2%}" for a, b in (("r1", "r2"), ("r1", "r3"), ("r2", "r3"))))
    for pred in J.PREDICATES:
        again, orders = same(p(Q(pred)), p(Q(pred, "random-again"))), [same(p(Q(pred, a)), p(Q(pred, b))) for a, b in (("random", "natural"), ("random", "reversed"), ("natural", "reversed"))]
        print(f"   {pred:<9}pairwise, today vs phase 1: {same(p(PAIR(pred)), J.earlier('jev', pred)):.2%}; Q k=10 same order asked twice: {again:.2%}; "
              f"Q k=10 reordered: {min(orders):.2%} to {max(orders):.2%}")
        near = float(np.mean(np.abs(p(PAIR(pred)) - 0.5) < 0.1)), float(np.mean(np.abs(p(Q(pred)) - 0.5) < 0.1))
        print(f"   {'':<9}pairs with probability between 0.4 and 0.6: pairwise {near[0]:.1%}, Q {near[1]:.1%}")

    print("\nB  Packing with one partner per request (k=1), opposite: precision / recall / F1")
    y = J.truth("opposite")
    single, pair, ten = p(SINGLE), p(PAIR("opposite")), p(Q("opposite"))
    for label, x in (("pairwise", pair), ("Q, k=1", single), ("Q, k=10", ten)):
        print(f"   {label:<9}" + " / ".join(f"{v:.1%}" for v in J.prf(J.decide(x), y)))
    d, lo, hi = J.f1_difference(J.decide(single), J.decide(pair), y)
    print(f"   Q k=1 vs pairwise {d:+.1f} ({lo:+.1f} to {hi:+.1f}); agreement: Q k=1 vs pairwise {same(single, pair):.2%}, Q k=1 vs Q k=10 {same(single, ten):.2%}")
    run = J.read("jev", SINGLE)[1]
    pr = J.read("jev", PAIR("opposite"))[1]
    print(f"   Q k=1: {run['requests']:,} requests, input tokens {run['input_tokens'] / pr['input_tokens']:.0%} of pairwise, time {run['seconds'] / pr['seconds']:.0%}")

    print("\nC  Which way it leans (Jev)")
    print(f"   {'question':<10}{'form':<10}{'judged yes':>11}{'truly yes':>10}{'mean p, truth yes':>18}{'truth no':>9}")
    for pred in J.PREDICATES:
        y = J.truth(pred)
        for label, x in (("pairwise", p(PAIR(pred))), ("Q, k=10", p(Q(pred)))):
            print(f"   {pred:<10}{label:<10}{J.decide(x).mean():>14.1%}{y.mean():>15.1%}{x[y].mean():>24.3f}{x[~y].mean():>14.3f}")

    print("\nD  Agreement with the answer derived from \"each review's own sentiment\" (phase 1's rewrite) (Jev)")
    for pred in J.PREDICATES:
        alt = J.rewritten(pred)
        print(f"   {pred:<10}pairwise {np.mean(J.decide(p(PAIR(pred))) == alt):.1%}, Q k=10 {np.mean(J.decide(p(Q(pred))) == alt):.1%}; the rewrite itself vs truth {np.mean(alt == J.truth(pred)):.1%}")
    cost = sum(J.read("jev", s)[1]["spent_usd"] for s in (*AGAIN, SINGLE))
    print(f"\ncost of the extra questions: ${cost:.3f}")


if __name__ == "__main__":
    if sys.argv[1:] == ["collect"]:
        collect()
    elif sys.argv[1:] == ["report"]:
        report()
    else:
        sys.exit(__doc__)
