"""Phase 8 post-hoc analysis: everything here was added after seeing the results and is not in the criteria fixed beforehand (experiments/database/protocol/p8.txt). Replays the collected results only and calls no model.

    python experiments/database/p8_posthoc.py > out/database/p8/posthoc.txt

A  Is the multiple choice's higher F1 due to the multiple-choice question itself, or only to forcing "a left record matches at most one"? Apply the same constraint to the yes/no probabilities
   (keep only each left record's highest-probability candidate at ≥0.5) and compare again.
B  The multiple choice changes when candidates are reordered: does averaging each option's probability over several orders and then choosing pull it back?
C  The whole-table account: tokens, cost, time. Pairwise numbers come from the 5 October run log of experiments/table_match/; packed ones from this run's fresh requests, scaled by request count.
D  For the same candidate pairs, the input tokens each of the two backends bills.
"""

from __future__ import annotations

import json

import numpy as np

import p8_join as J
from paths import ROOT

USD = 42 / 1e9  # Jev's list price per input token


def best_only(judged: dict) -> dict:
    """Keep only each left record's highest-probability candidate among those judged a match."""
    top: dict[str, tuple] = {}
    for (lid, rid), (yes, p) in judged.items():
        if yes and p is not None and p > top.get(lid, (None, -1))[1]:
            top[lid] = (rid, p)
    return {(lid, rid): (top.get(lid, (None,))[0] == rid, p) for (lid, rid), (_, p) in judged.items()}


def line(label: str, judged: dict, dataset: str, lids: list[str], base: np.ndarray | None = None) -> np.ndarray:
    t = J.tally(judged, dataset, lids)
    p, r, f = J.prf(t.sum(0))
    several = sum(n > 1 for n in np.bincount([lids.index(l) for (l, _), (yes, _) in judged.items() if yes and l in set(lids)], minlength=len(lids)))
    diff = "" if base is None else "   F1 difference from the multiple choice {:+.1f} ({:+.1f} to {:+.1f})".format(*J.f1_difference(t, base))
    print(f"   {label:<22}{p:>8.1%}{r:>8.1%}{f:>8.1%}   left records with two or more matches {several:>4}{diff}")
    return t


def part_a() -> None:
    dataset = J.SINGLE_MATCH[0]
    left = J.load(dataset)[0]
    print("A  \"a left record matches at most one\" applied to the yes/no questions (abt-buy; precision / recall / F1)")
    for source, scope in (("jev", "full"), ("jev", "sub"), ("kev-4b", "sub")):
        lids = list(left) if scope == "full" else J.subset(dataset, left)
        print(f"   -- {source} ({'whole table' if scope == 'full' else 'subset, the same 400 records for both backends'})")
        choice = line("multiple choice", J.judgments(J.read(source, dataset, "C", "random", scope)[0], "C"), dataset, lids)
        forms = {"pairwise": J.reference(source, dataset, scope)} | {f: J.judgments(J.read(source, dataset, f, "random", scope)[0], f) for f in ("Q", "S")}
        for name, judged in forms.items():
            line(name, judged, dataset, lids)
            line(name + " + at most one", best_only(judged), dataset, lids, choice)


def part_b() -> None:
    dataset = J.SINGLE_MATCH[0]
    sub = J.subset(dataset, J.load(dataset)[0])
    print("\nB  Multiple choice: average the probabilities over several orders, then choose (abt-buy subset; F1)")
    for source in J.SOURCES:
        runs = {o: {r["lid"]: r for r in J.read(source, dataset, "C", o, "sub")[0]} for o in J.ORDERS}
        single = {o: float(J.prf(J.tally(J.judgments(list(rows.values()), "C"), dataset, sub).sum(0))[2]) for o, rows in runs.items()}
        for orders in (("rank", "reversed"), J.ORDERS):
            judged = {}
            for lid in sub:
                mean: dict[str, float] = {}
                for o in orders:
                    r = runs[o][lid]
                    for key, p in r["probs"].items():
                        name = "none" if key == "none" else r["rids"][int(key[1:]) - 1]
                        mean[name] = mean.get(name, 0.0) + p / len(orders)
                pick = max(mean, key=mean.get)
                judged |= {(lid, rid): (rid == pick, mean[rid]) for rid in runs[orders[0]][lid]["rids"]}
            f = float(J.prf(J.tally(judged, dataset, sub).sum(0))[2])
            print(f"   {source:<8}{' + '.join(orders):<26}averaged {f:.1%} ({len(orders)}x the requests); single orders: " + "  ".join(f"{o} {single[o]:.1%}" for o in orders))


def part_c() -> None:
    print("\nC  The whole-table account (Jev)")
    print(f"   {'dataset':<14}{'form':<9}{'requests':>8}{'input tokens':>13}{'cost':>9}{'time':>9}")
    for dataset in J.DATASETS:
        old = [json.loads(x) for x in (ROOT / "out" / "table_match" / f"{dataset}.jev-1.13.0.runs.jsonl").read_text().splitlines()]
        requests, tokens, seconds = sum(r["fresh"] for r in old), sum(r["input_tokens"] for r in old), sum(r["seconds"] for r in old)
        print(f"   {dataset:<14}{'pairwise':<9}{requests:>10,}{tokens:>13,}{tokens * USD:>9.3f}{seconds:>8.0f}s   (run log of 5 October)")
        for form in J.PACKED:
            got = J.read("jev", dataset, form, "random", "full")
            sub = J.read("jev", dataset, form, "random", "sub")
            if got is None:
                continue
            # In the whole-table step the subset's requests are cache hits; the time is the subset pass plus the rest asked fresh
            seconds = sub[1]["seconds"] + got[1]["seconds"]
            print(f"   {dataset:<14}{form:<6}{got[1]['requests']:>10,}{got[1]['input_tokens']:>13,}{got[1]['input_tokens'] * USD:>9.3f}{seconds:>8.0f}s   "
                  f"tokens {got[1]['input_tokens'] / tokens:.0%} of pairwise, time {seconds / old_seconds(old):.0%}")


def old_seconds(old: list[dict]) -> float:
    return sum(r["seconds"] for r in old)


def part_d() -> None:
    print("\nD  The same candidate pairs (subset), input tokens billed by each backend: mean per candidate pair")
    print(f"   {'dataset':<14}{'form':<9}{'jev':>8}{'kev-4b':>9}{'diff':>8}")
    for dataset in J.DATASETS:
        for form in ("pair", *J.PACKED):
            a, b = J.read("jev", dataset, form, "rank" if form == "pair" else "random", "sub"), J.read("kev-4b", dataset, form, "rank" if form == "pair" else "random", "sub")
            if a and b:
                x, y = a[1]["input_tokens"] / a[1]["pairs"], b[1]["input_tokens"] / b[1]["pairs"]
                print(f"   {dataset:<14}{form:<6}{x:>8.0f}{y:>9.0f}{x - y:>8.0f}")


if __name__ == "__main__":
    print("Phase 8 post-hoc analysis (added after seeing the results, not in the criteria fixed beforehand)\n")
    part_a()
    part_b()
    part_c()
    part_d()
