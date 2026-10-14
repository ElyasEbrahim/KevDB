"""Text classification: how accurate Jev's multiple-choice questions (choice) are on public benchmarks with human labels.

    python experiments/classify/run.py agnews                 # Jev, official API, full test split
    python experiments/classify/run.py agnews --sample 500 --llm deepseek-v4.1-flash --concurrency 8   # comparison, through the third-party platform
    python experiments/classify/run.py --report               # summary

The questions are zero-shot first drafts that give only the class names (plus one sentence each for AG News) and were never edited against the test split.
Writes out/classify/<dataset>.<model>.jsonl (id, truth, prediction, confidence).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import sys
from pathlib import Path

from kevdb import JevClient, Question, Task, run_task
from kevdb.cache import Cache
from kevdb.cli import load_dotenv
from kevdb.llm import LLMClient

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "out" / "classify"
AG_CLASSES = {
    "World": "International news, politics, conflicts, and world affairs.",
    "Sports": "Sports events, teams, athletes, and results.",
    "Business": "Companies, markets, the economy, and finance.",
    "Sci/Tech": "Science, technology, computing, and the internet.",
}


def load(dataset: str) -> tuple[list[dict], Question, list[str]]:
    """Return (rows, question, fields shown to the model)."""
    if dataset == "agnews":
        names = list(AG_CLASSES)
        with (HERE / "data" / "agnews_test.csv").open(encoding="utf-8", newline="") as f:
            rows = [{"id": f"ag-{i}", "title": title, "description": body, "label": names[int(k) - 1]} for i, (k, title, body) in enumerate(csv.reader(f))]
        question = Question("choice", "Which news category does the article in `title` and `description` belong to?", AG_CLASSES)
        return rows, question, ["title", "description"]
    with (HERE / "data" / "banking77_test.csv").open(encoding="utf-8", newline="") as f:
        rows = [{"id": f"b77-{i}", "text": r["text"], "label": r["category"]} for i, r in enumerate(csv.DictReader(f))]
    # The 77 intents get class names only, no descriptions: the standard zero-shot setting
    labels = sorted({r["label"] for r in rows})
    question = Question("choice", "Which banking customer-service intent best matches the customer's message in `text`?", {label: None for label in labels})
    return rows, question, ["text"]


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def report() -> None:
    for dataset in ("agnews", "banking77"):
        files = sorted(OUT.glob(f"{dataset}.*.jsonl"))
        files = [f for f in files if not f.name.endswith(".runs.jsonl")]
        if not files:
            continue
        print(f"\n== {dataset}")
        by_model = {f.name.removeprefix(dataset + ".").removesuffix(".jsonl"): {r["id"]: r for r in read(f) if r["pred"] is not None} for f in files}
        for model, rows in by_model.items():
            values = list(rows.values())
            acc = sum(r["pred"] == r["label"] for r in values) / len(values)
            runs_path = OUT / f"{dataset}.{model}.runs.jsonl"
            runs = read(runs_path) if runs_path.exists() else []
            fresh = sum(r["fresh"] for r in runs) or 1
            cost, seconds = sum(r["cost_usd"] for r in runs), sum(r["seconds"] for r in runs)
            print(f"  {model:<28} {len(values):>5} rows  accuracy {acc:.1%}  ${cost / fresh * 1000:.3f} per 1,000  {fresh / max(seconds, 1e-9):.1f} rows/s"
                  f"  median latency {runs[-1]['latency_p50']:.2f}s" if runs else f"  {model:<28} {len(values):>5} rows  accuracy {acc:.1%}")
        jev = by_model.get("jev-1.13.0")
        if not jev:
            continue
        # Confidence routing: auto-accept only rows with high enough confidence, and see how many pass and how accurate they are
        values = list(jev.values())
        print("  Jev confidence routing:", "  ".join(
            f"≥{t}: {sum(r['conf'] >= t for r in values) / len(values):.0%} pass, accuracy {sum(r['pred'] == r['label'] for r in values if r['conf'] >= t) / max(sum(r['conf'] >= t for r in values), 1):.1%}"
            for t in (0.5, 0.7, 0.9)))
        for model, rows in by_model.items():
            if model == "jev-1.13.0":
                continue
            common = [i for i in rows if i in jev]
            print(f"  same {len(common)} rows: Jev {sum(jev[i]['pred'] == jev[i]['label'] for i in common) / len(common):.1%}"
                  f"  {model} {sum(rows[i]['pred'] == rows[i]['label'] for i in common) / len(common):.1%}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", choices=["agnews", "banking77"])
    parser.add_argument("--sample", type=int, help="draw this many rows at random (fixed seed)")
    parser.add_argument("--llm", metavar="MODEL", help="use an LLM on the third-party platform as the comparison; the official Jev API if omitted")
    parser.add_argument("--reasoning", default="none")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        return report()
    load_dotenv()

    rows, question, fields = load(args.dataset)
    if args.sample and args.sample < len(rows):
        rows = random.Random(0).sample(rows, args.sample)
    task = Task(name=f"cls_{args.dataset}", input="", questions={"label_pred": question}, state_fields=fields,
                provider="typesafe", model="jev-1.13.0", concurrency=args.concurrency)
    if args.llm:
        client, label = LLMClient(provider="opencode", model=args.llm, timeout=120.0, reasoning=None if args.reasoning == "default" else args.reasoning), args.llm
    else:
        client, label = JevClient(provider=task.provider, model=task.model), task.model
    async with client:
        results, stats = await run_task(task, rows, client, Cache())

    out = [{"id": row["id"], "label": row["label"], "pred": r.get("label_pred"), "conf": r.get("label_pred_conf"), "error": r.get("_error")}
           for row, r in zip(rows, results) if not r.get("_pending")]
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{args.dataset}.{label}"
    stem.with_name(stem.name + ".jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    fresh = len(out) - stats.cached - stats.errors
    summary = {"rows": len(out), "fresh": fresh, "errors": stats.errors, "pending": stats.pending, "seconds": round(stats.seconds, 1),
               "latency_p50": stats.latency(0.5), "input_tokens": stats.input_tokens, "output_tokens": stats.output_tokens, "cost_usd": round(stats.cost_usd, 5)}
    if fresh:
        with stem.with_name(stem.name + ".runs.jsonl").open("a") as f:
            f.write(json.dumps(summary) + "\n")
    answered = [r for r in out if r["pred"] is not None]
    print(f"{args.dataset} / {label}: {summary}  accuracy {sum(r['pred'] == r['label'] for r in answered) / max(len(answered), 1):.1%}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
