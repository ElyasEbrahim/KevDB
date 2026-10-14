"""Command-line entry point: kevdb run <task.toml> / kevdb sample / kevdb eval / kevdb ping"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import Counter
from pathlib import Path

from .cache import Cache
from .client import PROVIDERS, JevAPIError, JevClient
from .evaluate import evaluate, format_report, sample_for_gold
from .io import Row, read_rows, write_csv, write_jsonl
from .llm import LLMClient
from .mock import mock_transport
from .pipeline import Stats, Task, run_task


def load_dotenv(path: str = ".env") -> None:
    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def make_client(
    provider: str | None, model: str | None, mock: bool, llm: str | None = None, reasoning: str = "none"
) -> JevClient:
    if mock:
        return JevClient("mock", provider=provider, model=model, transport=mock_transport())
    if llm:
        # Generative models are much slower than Jev, so allow a longer timeout
        return LLMClient(provider="opencode", model=llm, timeout=120.0, reasoning=None if reasoning == "default" else reasoning)
    return JevClient(provider=provider, model=model)


def progress(done: int, total: int) -> None:
    if done == total or done % 25 == 0:
        print(f"\r  {done}/{total}", end="", file=sys.stderr, flush=True)


def summarize(task: Task, results: list[Row], stats: Stats, mock: bool) -> str:
    processed = stats.rows - stats.pending
    fresh = processed - stats.cached - stats.errors
    lines = [
        f"task {task.name}: {stats.rows} rows, {len(task.questions)} questions per row, model {', '.join(sorted(stats.models)) or '-'}",
        f"  time {stats.seconds:.1f}s ({processed / max(stats.seconds, 1e-9):.1f} rows/s)",
        f"  new requests {fresh}, cache hits {stats.cached}, failures {stats.errors}",
        f"  request latency: median {stats.latency(0.5) or 0:.2f}s, p95 {stats.latency(0.95) or 0:.2f}s",
        f"  input tokens {stats.input_tokens:,}, output tokens {stats.output_tokens:,}, about ${stats.cost_usd:.4f}",
        f"  to review: {stats.review} rows ({stats.review / max(processed, 1):.0%})",
    ]
    if stats.pending:
        hours = (stats.quota_retry_after or 0) / 3600
        lines.insert(1, f"  ⚠ quota exhausted: only {processed}/{stats.rows} rows done, {stats.pending} rows pending. "
                        f"The server asks to retry in about {hours:.1f} hours; then re-run the same command to resume (finished rows come from the cache).")
    for name, q in task.questions.items():
        counts = Counter(r[name] for r in results if name in r)
        if q.type == "score":
            values = [r[name] for r in results if name in r]
            lines.append(f"  {name}: mean {sum(values) / max(len(values), 1):.2f}")
        else:
            lines.append(f"  {name}: " + "  ".join(f"{k}={v}" for k, v in counts.most_common()))
    if mock:
        lines.append("  ⚠ these are --mock results: answers are random and tokens are estimated; they only show that the pipeline runs.")
    return "\n".join(lines)


async def cmd_run(args: argparse.Namespace) -> int:
    task = Task.load(args.task)
    if args.input:
        task.input, task.records_path = args.input, args.records_path
    if args.state_fields:
        task.state_fields = args.state_fields.split(",")
    if args.concurrency:
        task.concurrency = args.concurrency
    rows = read_rows(task.input, task.records_path)
    if args.limit:
        rows = rows[: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Fake results stay out of the cache so they cannot pollute later real runs
    cache = None if args.mock or args.no_cache else Cache()
    async with make_client(task.provider, task.model, args.mock, args.llm, args.reasoning) as client:
        results, stats = await run_task(task, rows, client, cache, progress)
    print(file=sys.stderr)

    # Output file names follow the input file, so different datasets do not overwrite each other
    merged = [{**row, **result} for row, result in zip(rows, results) if not result.get("_pending")]
    tag = args.tag or (args.llm and f"{args.llm}.{args.reasoning}")
    stem = f"{task.name}.{Path(task.input).stem}" + (f".{tag}" if tag else "") + (".mock" if args.mock else "")
    write_jsonl(out_dir / f"{stem}.jsonl", merged)
    write_csv(out_dir / f"{stem}.csv", merged)
    write_jsonl(out_dir / f"{stem}.review.jsonl", [r for r in merged if r["_review"]])
    print(summarize(task, results, stats, args.mock))
    print(f"  output: {out_dir / stem}.jsonl / .csv / .review.jsonl")
    return 2 if stats.pending else 0


async def cmd_sample(args: argparse.Namespace) -> int:
    """Draw rows to label from a prediction file. The output holds only the original fields, no model answers, so labelling stays blind."""
    task = Task.load(args.task)
    pred = read_rows(args.pred)
    picked = sample_for_gold(pred, task.id_field, args.class_field, args.n, args.min_per_class, args.seed)
    keep = [task.id_field, *task.state_fields, "uniform"]
    write_jsonl(args.out, [{k: r[k] for k in keep if k in r} for r in picked])
    print(f"sampled {len(picked)} rows ({sum(r['uniform'] for r in picked)} uniform, {sum(not r['uniform'] for r in picked)} added for rare classes) → {args.out}")
    return 0


async def cmd_eval(args: argparse.Namespace) -> int:
    task = Task.load(args.task)
    print(f"predictions {args.pred}  vs  gold {args.gold}")
    print(format_report(evaluate(task, read_rows(args.pred), read_rows(args.gold))))
    return 0


async def cmd_ping(args: argparse.Namespace) -> int:
    """Send one minimal real request to check that the key works."""
    async with make_client(args.provider, args.model, mock=False) as client:
        resp = await client.ask(
            "Help! My payouts have been failing for 3 days.",
            {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}},
        )
    print(
        f"OK  provider={client.provider}  model={resp['model']}"
        f"  is_urgent={resp['answers']['is_urgent']['noul']:.2f}  usage={resp['usage']}"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="kevdb", description="Batch data labelling with decision models")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="process one data file according to a task configuration")
    run.add_argument("task", help="task configuration (.toml)")
    run.add_argument("--mock", action="store_true", help="use the offline fake backend (no key needed, answers are meaningless)")
    run.add_argument("--input", help="use another data file (overrides input in the task configuration)")
    run.add_argument("--records-path", help="with --input: key of the record list inside a .json file")
    run.add_argument("--limit", type=int, help="process only the first N rows")
    run.add_argument("--out", default="out", help="output directory")
    run.add_argument("--no-cache", action="store_true", help="do not read or write the answer cache")
    run.add_argument("--state-fields", help="comma-separated; overrides state_fields in the task configuration (for ablations)")
    run.add_argument("--tag", help="added to the output file names so official results are not overwritten")
    run.add_argument("--concurrency", type=int, help="override the concurrency in the task configuration")
    run.add_argument("--llm", metavar="MODEL", help="answer the same questions with a generative LLM on OpenCode instead (baseline), e.g. deepseek-v4.1-flash")
    run.add_argument(
        "--reasoning", default="none",
        help="with --llm: value of reasoning_effort. none (default) turns thinking off; some models do not accept none, use low; default omits it and uses the model default",
    )
    run.set_defaults(func=cmd_run)

    sample = sub.add_parser("sample", help="sample rows from predictions to build a file for labelling")
    sample.add_argument("task")
    sample.add_argument("--pred", required=True, help=".jsonl written by run")
    sample.add_argument("--out", required=True)
    sample.add_argument("--class-field", required=True, help="choice question whose predicted classes decide which rare classes get extra rows")
    sample.add_argument("--n", type=int, default=150, help="number of rows drawn uniformly at random")
    sample.add_argument("--min-per-class", type=int, default=8)
    sample.add_argument("--seed", type=int, default=7)
    sample.set_defaults(func=cmd_sample)

    ev = sub.add_parser("eval", help="compare predictions with a gold set")
    ev.add_argument("task")
    ev.add_argument("--pred", required=True)
    ev.add_argument("--gold", required=True)
    ev.set_defaults(func=cmd_eval)

    ping = sub.add_parser("ping", help="check that an API key works")
    ping.add_argument("--provider", choices=list(PROVIDERS), help="default: the provider that has a key configured")
    ping.add_argument("--model", help="default: the provider's default model")
    ping.set_defaults(func=cmd_ping)

    args = parser.parse_args()
    load_dotenv()
    try:
        sys.exit(asyncio.run(args.func(args)))
    except* JevAPIError as group:
        print(f"\nerror: {group.exceptions[0]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
