"""Ask a model whether a candidate plan will be faster than the default plan.

For each query, every candidate plan that differs from the default is one yes/no question (noul). Jev returns a probability, an LLM only yes or no.
Reuses the kevdb pipeline (concurrency, retries, cache, cost accounting) and writes out/query_plan/decisions.<model>.<representation>.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from headroom import load  # noqa: E402

from kevdb import JevClient, Question, Task, run_task  # noqa: E402
from kevdb.cache import Cache  # noqa: E402
from kevdb.cli import load_dotenv  # noqa: E402
from kevdb.llm import LLMClient  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "out" / "query_plan"
ESTIMATE = re.compile(r"\(cost=[\d.]+\.\.[\d.]+ rows=(\d+) width=\d+\)")
BUCKETS = [(10, "under 10"), (1_000, "tens to hundreds"), (100_000, "thousands"), (10_000_000, "hundreds of thousands to millions"), (float("inf"), "tens of millions or more")]

# The default plan always has the lowest estimated cost, so a model that only reads the cost figures will always pick the default.
# The three representations separate a model reading the numbers from a model using knowledge beyond them.
REPS = {
    "full": "The cost and row numbers in the plans are the planner's estimates and can be badly wrong; the default plan always has the lower estimated cost.",
    "bucketed": "The row counts in the plans are the planner's estimates, shown as rough ranges, and can be badly wrong.",
    "bare": "The plans show only the operators and conditions, without the planner's estimates.",
}


def render(plan_text: str, rep: str) -> str:
    """Jev is poor at comparing raw numbers (the official docs suggest bucketing), so there are versions without numbers and with bucketed numbers."""
    if rep == "full":
        return plan_text
    if rep == "bare":
        return ESTIMATE.sub("", plan_text).replace("  \n", "\n")

    def bucket(match: re.Match) -> str:
        rows = int(match.group(1))
        return f"(estimated rows: {next(name for limit, name in BUCKETS if rows < limit)})"

    return ESTIMATE.sub(bucket, plan_text)


def build_items(truth: dict[str, dict[str, dict]], rep: str) -> list[dict]:
    queries = HERE / "data" / "job-queries"
    items = []
    for name, arms in truth.items():
        default = arms["default"]
        sql = (queries / f"{name}.sql").read_text().strip()
        seen = {default["plan_key"]}
        for arm, record in arms.items():
            if record["plan_key"] in seen:
                continue
            seen.add(record["plan_key"])
            items.append({
                "id": f"{name}:{arm}", "query": name, "arm": arm,
                "sql": sql,
                "default_plan": render(default["plan_text"], rep),
                "alternative_plan": render(record["plan_text"], rep),
                "default_ms": default["ms"], "alt_ms": record["ms"],
            })
    return items


def make_task(rep: str) -> Task:
    question = Question(
        "noul",
        "`default_plan` and `alternative_plan` are two PostgreSQL execution plans for the query in `sql`; both return the same result. "
        f"{REPS[rep]} Will `alternative_plan` finish faster than `default_plan` when actually executed?",
        {"true": "The alternative plan has the shorter real execution time.", "false": "The default plan is at least as fast."},
    )
    return Task(
        name="qo_pairwise", input="", questions={"alt_faster": question},
        state_fields=["sql", "default_plan", "alternative_plan"],
        # Jev uses the official API; only the --llm comparison models go through the third-party platform
        provider="typesafe", model="jev-1.13.0", concurrency=16, max_field_chars=30_000,
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", choices=list(REPS), default="full")
    parser.add_argument("--llm", metavar="MODEL", help="use an LLM on OpenCode instead, e.g. deepseek-v4.1-flash; Jev if omitted")
    parser.add_argument("--reasoning", default="none", help="with --llm: none / low / default")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    load_dotenv()

    task = make_task(args.rep)
    items = build_items(load(), args.rep)[: args.limit]
    if args.llm:
        client = LLMClient(provider="opencode", model=args.llm, timeout=180.0, reasoning=None if args.reasoning == "default" else args.reasoning)
        label = f"{args.llm}.{args.reasoning}"
    else:
        client = JevClient(provider=task.provider, model=task.model)
        label = task.model
    async with client:
        results, stats = await run_task(task, items, client, Cache())

    keep = ("id", "query", "arm", "default_ms", "alt_ms")
    rows = [{**{k: item[k] for k in keep}, "p": r.get("alt_faster_p"), "error": r.get("_error")} for item, r in zip(items, results) if not r.get("_pending")]
    path = OUT / f"decisions.{label}.{args.rep}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{label} / {args.rep}: {len(rows)} questions, {stats.errors} failed, {stats.cached} cache hits, {stats.pending} pending")
    print(f"  time {stats.seconds:.1f}s, median latency {stats.latency(0.5) or 0:.2f}s")
    print(f"  input tokens {stats.input_tokens:,}, output tokens {stats.output_tokens:,}, about ${stats.cost_usd:.4f}")
    print(f"  → {path}")


if __name__ == "__main__":
    asyncio.run(main())
