"""Strong-model comparison: a flagship LLM judges Abt-Buy's candidate pairs, to answer two questions.

1. Jev first, unsure pairs to the strong model: how good is the result? (Every pair Jev is unsure about goes to the strong model, so this is fully measured)
2. What if the strong model judged everything? (It is too expensive, so it judges only a random sample of the pairs Jev is sure about, scaled up proportionally)

    python experiments/table_match/strong.py run        # calls the strong model through the third-party platform
    python experiments/table_match/strong.py report
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import DATASETS, KINDS, OUT, build_pairs, load_truth  # noqa: E402

from kevdb import Task, run_task  # noqa: E402
from kevdb.cache import Cache  # noqa: E402
from kevdb.cli import load_dotenv  # noqa: E402
from kevdb.llm import LLM_PRICES, LLMClient  # noqa: E402

DATASET, K = "abt-buy", 5
STRONG = "claude-opus-5-5"
SMALL = "deepseek-v4.1-flash.none"
# How many of Jev's confident pairs per stratum go to the strong model. It always thinks adaptively and cannot turn it off, about $0.0046 per pair, so the samples are kept small
SAMPLE = {"sure_yes": 100, "sure_no": 150}
JEV_PER_PAIR, SMALL_PER_PAIR = 0.023 / 1000, 0.119 / 1000


def stratum(p: float) -> str:
    return "sure_yes" if p >= 0.9 else "sure_no" if p <= 0.1 else "unsure"


def read(name: str) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, (OUT / f"{DATASET}.{name}.jsonl").read_text().splitlines())}


def chosen_pairs() -> list[dict]:
    pairs = build_pairs(DATASET, K)
    jev = read("jev-1.13.0")
    rng = random.Random(0)
    by = {"unsure": [], "sure_yes": [], "sure_no": []}
    for pair in pairs:
        by[stratum(jev[pair["id"]]["p"])].append(pair)
    picked = by["unsure"] + rng.sample(by["sure_yes"], SAMPLE["sure_yes"]) + rng.sample(by["sure_no"], SAMPLE["sure_no"])
    return [{**p, "stratum": stratum(jev[p["id"]]["p"])} for p in picked]


async def run(limit: int | None) -> None:
    load_dotenv()
    pairs = chosen_pairs()[:limit]
    task = Task(name=f"dedup_{DATASET}", input="", questions={"same": KINDS[DATASETS[DATASET]["kind"]]},
                state_fields=["record_a", "record_b"], concurrency=8)
    async with LLMClient(provider="opencode", model=STRONG, timeout=120.0, reasoning=None) as client:
        results, stats = await run_task(task, pairs, client, Cache())
    rows = [{"id": p["id"], "label": p["label"], "stratum": p["stratum"], "p": r.get("same_p"), "error": r.get("_error")}
            for p, r in zip(pairs, results) if not r.get("_pending")]
    (OUT / f"{DATASET}.{STRONG}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    fresh = len(rows) - stats.cached - stats.errors
    summary = {"pairs": len(rows), "fresh": fresh, "errors": stats.errors, "pending": stats.pending, "seconds": round(stats.seconds, 1),
               "latency_p50": stats.latency(0.5), "input_tokens": stats.input_tokens, "output_tokens": stats.output_tokens, "cost_usd": round(stats.cost_usd, 4)}
    if fresh:
        with (OUT / f"{DATASET}.{STRONG}.runs.jsonl").open("a") as f:
            f.write(json.dumps(summary) + "\n")
    print(summary)


def f1(tp: float, fp: float, n_truth: int) -> tuple[float, float, float]:
    precision, recall = tp / max(tp + fp, 1e-9), tp / n_truth
    return precision, recall, 2 * precision * recall / max(precision + recall, 1e-9)


def report() -> None:
    truth = load_truth(DATASETS[DATASET]["truth"])
    jev, strong = read("jev-1.13.0"), read(STRONG)
    ids = {p["id"] for p in build_pairs(DATASET, K)}
    jev = {i: r for i, r in jev.items() if i in ids}
    sizes = {s: sum(stratum(r["p"]) == s for r in jev.values()) for s in ("unsure", "sure_yes", "sure_no")}
    judged = {s: [r for r in strong.values() if r["stratum"] == s and r["p"] is not None] for s in sizes}
    runs = [json.loads(line) for line in (OUT / f"{DATASET}.{STRONG}.runs.jsonl").read_text().splitlines()]
    per_pair = sum(r["cost_usd"] for r in runs) / sum(r["fresh"] for r in runs)
    print(f"strong model {STRONG}: {sum(len(v) for v in judged.values())} pairs measured, {sum(r['p'] is None for r in strong.values())} failed, "
          f"${per_pair * 1000:.2f} per 1,000 pairs (Jev ${JEV_PER_PAIR * 1000:.3f}), median latency {runs[-1]['latency_p50']:.2f}s")
    for s, rows in judged.items():
        acc = sum((r["p"] >= 0.5) == bool(r["label"]) for r in rows) / len(rows)
        jacc = sum((jev[r["id"]]["p"] >= 0.5) == bool(r["label"]) for r in rows) / len(rows)
        print(f"  {s:<9} {sizes[s]:>5} pairs in total, strong model judged {len(rows):>4}, accuracy {acc:.1%} (Jev on the same pairs {jacc:.1%})")

    # 1. Two-tier scheme: fully measured
    unsure_ids = {r["id"] for r in judged["unsure"]}
    pred = [i for i, r in jev.items() if (strong[i]["p"] >= 0.5 if i in unsure_ids else r["p"] >= 0.5)]
    tp = sum(jev[i]["label"] for i in pred)
    p, r, f = f1(tp, len(pred) - tp, len(truth))
    cost = len(jev) * JEV_PER_PAIR + sizes["unsure"] * per_pair
    print(f"\nJev first, {sizes['unsure']} unsure pairs to the strong model (measured): P {p:.1%} R {r:.1%} F1 {f:.1%}, cost ${cost:.2f}")

    # 2. Everything to the strong model: stratified estimate, interval from resampling
    def estimate(sample: dict[str, list[dict]]) -> tuple[float, float, float]:
        tp = fp = 0.0
        for s, rows in sample.items():
            weight = sizes[s] / len(rows)
            tp += weight * sum(r["p"] >= 0.5 and r["label"] for r in rows)
            fp += weight * sum(r["p"] >= 0.5 and not r["label"] for r in rows)
        return f1(tp, fp, len(truth))

    p, r, f = estimate(judged)
    rng = random.Random(1)
    boots = sorted(estimate({s: rng.choices(rows, k=len(rows)) for s, rows in judged.items()})[2] for _ in range(2000))
    print(f"everything to the strong model (stratified estimate): P {p:.1%} R {r:.1%} F1 {f:.1%}, 95% interval {boots[50]:.1%} to {boots[1949]:.1%}, "
          f"cost about ${len(jev) * per_pair:.2f} (from the measured unit price)")
    print(f"for comparison: Jev only ${len(jev) * JEV_PER_PAIR:.2f}; small model only ${len(jev) * SMALL_PER_PAIR:.2f}; "
          f"the strong model costs {per_pair / JEV_PER_PAIR:.0f}x Jev per pair")


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        asyncio.run(run(int(sys.argv[2]) if len(sys.argv) > 2 else None))
    else:
        report()
