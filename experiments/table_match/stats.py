"""Statistics for the main experiment: confidence intervals for each scheme's F1, whether the differences between schemes are significant, and the exact cost of the strong model.

Confidence intervals use the bootstrap: 2,000 resamples with one left-table (Abt) record as the unit. A record's candidate pairs are correlated,
so records are resampled, not candidate pairs; a true match lost at blocking goes with the record it belongs to.

    python experiments/table_match/stats.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import DATASETS, KINDS, OUT, build_pairs, load_truth  # noqa: E402
from strong import DATASET, K, SMALL, STRONG, read, stratum  # noqa: E402

from kevdb import Task  # noqa: E402
from kevdb.cache import Cache, cache_key  # noqa: E402
from kevdb.llm import LLM_PRICES  # noqa: E402

JEV_USD_PER_TOKEN = 42 / 1e9
RESAMPLES = 2000


def main() -> None:
    truth = load_truth(DATASETS[DATASET]["truth"])
    pairs = build_pairs(DATASET, K)
    ids = [p["id"] for p in pairs]
    jev, small, strong = read("jev-1.13.0"), read(SMALL), read(STRONG)
    unsure = {i for i in ids if stratum(jev[i]["p"]) == "unsure"}

    # Each scheme's final decision for each candidate pair
    schemes = {
        "Jev only": {i: jev[i]["p"] >= 0.5 for i in ids},
        "DeepSeek (small) only": {i: small[i]["p"] >= 0.5 for i in ids},
        "Jev first + DeepSeek": {i: (small[i]["p"] if i in unsure else jev[i]["p"]) >= 0.5 for i in ids},
        "Jev first + Opus": {i: (strong[i]["p"] if i in unsure else jev[i]["p"]) >= 0.5 for i in ids},
    }
    label = {i: bool(jev[i]["label"]) for i in ids}

    # Aggregate per left record: its true matches and each scheme's TP / FP on it
    truth_per_left: dict[str, int] = defaultdict(int)
    for a, _ in truth:
        truth_per_left[a] += 1
    lefts = sorted({i.split("|")[0] for i in ids} | set(truth_per_left))
    per = {name: defaultdict(lambda: [0, 0]) for name in schemes}
    for name, pred in schemes.items():
        for i in ids:
            if pred[i]:
                per[name][i.split("|")[0]][0 if label[i] else 1] += 1

    def f1(name: str, sample: list[str]) -> float:
        tp = sum(per[name][a][0] for a in sample)
        fp = sum(per[name][a][1] for a in sample)
        n = sum(truth_per_left[a] for a in sample)
        precision, recall = tp / max(tp + fp, 1), tp / max(n, 1)
        return 2 * precision * recall / max(precision + recall, 1e-9)

    rng = random.Random(0)
    boots = [rng.choices(lefts, k=len(lefts)) for _ in range(RESAMPLES)]
    lo, hi = int(RESAMPLES * 0.025), int(RESAMPLES * 0.975) - 1
    print(f"{DATASET}, top {K} candidates per row, {len(ids):,} pairs; {RESAMPLES} bootstrap resamples over left records\n")
    print(f"{'scheme':<24}{'F1':>7}{'95% interval':>18}")
    dist = {}
    for name in schemes:
        dist[name] = [f1(name, s) for s in boots]
        ordered = sorted(dist[name])
        print(f"{name:<26}{f1(name, lefts):>7.1%}{ordered[lo]:>10.1%} ~ {ordered[hi]:.1%}")
    print("\nF1 differences between schemes (paired over the same resamples; significant only if the interval excludes 0)")
    for a, b in (("Jev only", "DeepSeek (small) only"), ("Jev first + DeepSeek", "DeepSeek (small) only"), ("Jev first + DeepSeek", "Jev only"),
                 ("Jev first + Opus", "Jev only"), ("Jev first + Opus", "DeepSeek (small) only"), ("Jev first + Opus", "Jev first + DeepSeek")):
        diffs = sorted(x - y for x, y in zip(dist[a], dist[b]))
        print(f"  {a} − {b}: {f1(a, lefts) - f1(b, lefts):+.1%}, interval {diffs[lo]:+.1%} to {diffs[hi]:+.1%}")

    print(f"\naccuracy on the {len(unsure)} pairs Jev is unsure about: Jev {sum(schemes['Jev only'][i] == label[i] for i in unsure) / len(unsure):.1%}, "
          f"DeepSeek {sum((small[i]['p'] >= 0.5) == label[i] for i in unsure) / len(unsure):.1%}, "
          f"Opus {sum((strong[i]['p'] >= 0.5) == label[i] for i in unsure) / len(unsure):.1%}")

    # Exact cost of the strong model: the cache stores the token counts of every request
    task = Task(name="x", input="", questions={"same": KINDS[DATASETS[DATASET]["kind"]]}, state_fields=["record_a", "record_b"])
    payloads = {"same": KINDS[DATASETS[DATASET]["kind"]].payload()}
    cache = Cache()
    usd_in, usd_out = (x / 1e6 for x in LLM_PRICES[STRONG])
    cost: dict[str, list[float]] = defaultdict(list)
    out_tokens: dict[str, list[int]] = defaultdict(list)
    jev_cost = 0.0
    for p in pairs:
        state = task.state_of(p)
        j = cache.get(cache_key("jev-1.13.0", state, payloads))
        jev_cost += j["usage"]["input_tokens"] * JEV_USD_PER_TOKEN
        s = cache.get(cache_key(f"{STRONG}#chat-v1#default", state, payloads))
        if s:
            cost[stratum(jev[p["id"]]["p"])].append(s["usage"]["input_tokens"] * usd_in + s["usage"]["output_tokens"] * usd_out)
            out_tokens[stratum(jev[p["id"]]["p"])].append(s["usage"]["output_tokens"])
    sizes = {s: sum(stratum(jev[i]["p"]) == s for i in ids) for s in ("unsure", "sure_yes", "sure_no")}
    print(f"\ncost of the strong model {STRONG} (from each request's actual tokens)")
    for s in sizes:
        print(f"  {s:<9} judged {len(cost[s]):>4} pairs, ${sum(cost[s]) / len(cost[s]):.4f} per pair, {sum(out_tokens[s]) / len(out_tokens[s]):.0f} output tokens on average (thinking included)")
    cascade = jev_cost + sum(cost["unsure"])
    all_strong = sum(cost["unsure"]) + sum(sizes[s] * sum(cost[s]) / len(cost[s]) for s in ("sure_yes", "sure_no"))
    print(f"  Jev only: ${jev_cost:.3f}")
    print(f"  Jev first + Opus: ${cascade:.2f} (measured)")
    print(f"  everything to Opus: about ${all_strong:.2f} (unsure part measured, confident part scaled from the sampled unit price), {all_strong / jev_cost:.0f}x the cost of Jev only")
    print(f"  the two-tier scheme costs {cascade / all_strong:.0%} of using Opus for everything")


if __name__ == "__main__":
    main()
