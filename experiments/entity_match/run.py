"""Entity matching: ask the model whether two records describe the same entity.

Each record pair is one yes/no question. Jev returns the probability of "same", an LLM only yes or no.
Reuses the kevdb pipeline (concurrency, retries, cache, cost accounting).
Writes out/entity_match/<dataset>.<split>.<model>.jsonl (id, truth, probability) and a .stats.json with the same name (time, tokens, cost).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

from kevdb import JevClient, Question, Task, run_task
from kevdb.cache import Cache
from kevdb.cli import load_dotenv
from kevdb.llm import LLMClient

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "out" / "entity_match"
# An empty field at the end of a row is written "COL price VAL" with no trailing space
FIELD = re.compile(r"COL (\w+) VAL(?: |$)")

# The question is a zero-shot first draft. It may only be changed against the validation split; the test split only produces the final numbers.
KINDS = {
    "product": Question(
        "noul",
        "Do `record_a` and `record_b` describe the same product, i.e. the exact same item that a shopper could buy from either listing?",
        {
            "true": "Same brand and same model or edition. Differences in wording, missing fields, or price do not matter.",
            "false": "Different products, including a different model, size, capacity, color, edition, or version of a similar item.",
        },
    ),
    "publication": Question(
        "noul",
        "Do `record_a` and `record_b` refer to the same publication?",
        {
            "true": "The same paper: same title and authors, allowing for abbreviations, typos, and differently formatted venue names.",
            "false": "Different papers, including different papers by the same authors or with similar titles.",
        },
    ),
}
DATASETS = {"abt-buy": "product", "amazon-google": "product", "walmart-amazon": "product", "dblp-acm": "publication"}


def parse_record(text: str) -> dict[str, str]:
    """'COL name VAL sony ... COL price VAL 9.99' -> {'name': 'sony ...', 'price': '9.99'}; empty fields are dropped."""
    parts = FIELD.split(text.strip())[1:]
    return {name: value.strip() for name, value in zip(parts[::2], parts[1::2]) if value.strip()}


def load_pairs(dataset: str, split: str) -> list[dict]:
    rows = []
    for i, line in enumerate((HERE / "data" / dataset / f"{split}.txt").read_text(encoding="utf-8").splitlines()):
        left, right, label = line.split("\t")
        rows.append({"id": f"{split}-{i}", "record_a": parse_record(left), "record_b": parse_record(right), "label": int(label)})
    return rows


def make_task(dataset: str, concurrency: int) -> Task:
    return Task(
        name=f"em_{dataset}", input="", questions={"same": KINDS[DATASETS[dataset]]},
        # Jev uses the official API; only the --llm comparison models go through the third-party platform
        state_fields=["record_a", "record_b"], provider="typesafe", model="jev-1.13.0", concurrency=concurrency,
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(DATASETS))
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--sample", type=int, help="draw this many pairs at random (fixed seed, so Jev and the LLM get the same pairs)")
    parser.add_argument("--llm", metavar="MODEL", help="use an LLM on OpenCode instead; Jev if omitted")
    parser.add_argument("--reasoning", default="none")
    # The official API has its highest throughput at concurrency 16; LLM comparisons on the third-party platform use 8
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    load_dotenv()

    pairs = load_pairs(args.dataset, args.split)
    if args.sample and args.sample < len(pairs):
        pairs = random.Random(0).sample(pairs, args.sample)
    task = make_task(args.dataset, args.concurrency)
    if args.llm:
        client, label = LLMClient(provider="opencode", model=args.llm, timeout=120.0, reasoning=None if args.reasoning == "default" else args.reasoning), f"{args.llm}.{args.reasoning}"
    else:
        client, label = JevClient(provider=task.provider, model=task.model), task.model
    async with client:
        results, stats = await run_task(task, pairs, client, Cache())

    rows = [{"id": p["id"], "label": p["label"], "p": r.get("same_p"), "error": r.get("_error")} for p, r in zip(pairs, results) if not r.get("_pending")]
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{args.dataset}.{args.split}.{label}"
    stem.with_name(stem.name + ".jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    fresh = len(rows) - stats.cached - stats.errors
    summary = {
        "pairs": len(rows), "fresh": fresh, "cached": stats.cached, "errors": stats.errors, "pending": stats.pending,
        "seconds": round(stats.seconds, 1), "latency_p50": stats.latency(0.5), "latency_p95": stats.latency(0.95),
        "input_tokens": stats.input_tokens, "output_tokens": stats.output_tokens, "cost_usd": round(stats.cost_usd, 5),
    }
    # One line is appended per run: after an interrupted run is resumed, cost and tokens are the sum of each run's new requests
    if fresh:
        with stem.with_name(stem.name + ".runs.jsonl").open("a") as f:
            f.write(json.dumps(summary) + "\n")
    print(f"{args.dataset} {args.split} / {label}: {summary}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
