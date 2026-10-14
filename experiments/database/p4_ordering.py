"""Phase 4, part two: when only Jev executes, is it better to order several semantic predicates or to pack them into one request?

Replays the Jev judgements collected by p4_collect.py and calls no model. The design was written before the run, see experiments/database/protocol/p4.txt.

Queries: every conjunction of 2 or 3 of the 21 predicates, 210 + 1,330 queries, over 3,949 sentences.
Plans:
    one by one   one request per row and predicate, the next predicate only if the previous one passed. Three orders: worst, hindsight best, and ascending pass rate estimated by Jev on 50 sampled rows
    packed       one request per row with all of the query's predicates
Cost is counted in input tokens: the measured mean tokens for 1 and for 21 questions, interpolated linearly in the number of questions.

    python experiments/database/p4_ordering.py
"""

from __future__ import annotations

import itertools
import json

import numpy as np

from p4_collect import PREDICATES, sentences
from paths import OUT as RESULTS

OUT = RESULTS / "p4"
N_SAMPLE, SEED = 50, 20261007


def main() -> None:
    rows = sentences()
    names = list(PREDICATES)
    single = {r["sid"]: r["p"] for r in map(json.loads, (OUT / "mams.jev.jsonl").read_text().splitlines())}
    passed = np.array([[single[r["sid"]][k] >= 0.5 for k in names] for r in rows])
    n = len(rows)
    runs = {s: json.loads((OUT / f"mams.{s}.run.json").read_text()) for s in ("jev", "jev.packed")}
    tok1 = runs["jev"]["input_tokens"] / runs["jev"]["requests"]
    tok21 = runs["jev.packed"]["input_tokens"] / runs["jev.packed"]["requests"]
    per_question = (tok21 - tok1) / (len(names) - 1)
    tokens = lambda k: tok1 + (k - 1) * per_question  # input tokens of a request with k questions
    print(f"measured: {tok1:.0f} tokens for 1 question, {tok21:.0f} for 21 questions, about {per_question:.0f} per extra question")

    rng = np.random.default_rng(SEED)
    sample_rate = passed[rng.choice(n, N_SAMPLE, replace=False)].mean(axis=0)  # the optimizer only sees these 50 rows

    def sequential(order: tuple[int, ...]) -> int:
        """Total requests when asking one by one in this order."""
        alive, requests = np.ones(n, dtype=bool), 0
        for j in order:
            requests += alive.sum()
            alive &= passed[:, j]
        return int(requests)

    out = {}
    for size in (2, 3):
        ratios = {k: [] for k in ("worst", "chosen", "packed_tokens", "packed_requests", "chosen_is_best")}
        for combo in itertools.combinations(range(len(names)), size):
            costs = {order: sequential(order) for order in itertools.permutations(combo)}
            best, worst = min(costs.values()), max(costs.values())
            chosen = costs[tuple(sorted(combo, key=lambda j: sample_rate[j]))]
            ratios["worst"].append(worst / best)
            ratios["chosen"].append(chosen / best)
            ratios["chosen_is_best"].append(chosen == best)
            ratios["packed_tokens"].append(n * tokens(size) / (best * tok1))
            ratios["packed_requests"].append(n / best)
        q = lambda xs: (float(np.median(xs)), float(np.quantile(xs, 0.1)), float(np.quantile(xs, 0.9)))
        out[size] = {k: q(v) for k, v in ratios.items() if k != "chosen_is_best"}
        out[size]["chosen_is_best"] = float(np.mean(ratios["chosen_is_best"]))
        out[size]["packed_cheaper_than_best"] = float(np.mean(np.array(ratios["packed_tokens"]) < 1))
        print(f"\nqueries with {size} predicates ({len(ratios['worst'])}), ratio to the hindsight-best order: median (10th to 90th percentile)")
        for key, label in (("worst", "worst order, requests and tokens"), ("chosen", "order chosen from Jev's sample"), ("packed_requests", "packed, requests"), ("packed_tokens", "packed, tokens")):
            m, lo, hi = out[size][key]
            print(f"  {label:34s}{m:6.2f} ({lo:.2f} to {hi:.2f})")
        print(f"  the sampled order is exactly the best in {out[size]['chosen_is_best']:.1%}; packing uses fewer tokens than the best order in {out[size]['packed_cheaper_than_best']:.1%} of queries")
    (OUT / "ordering.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
