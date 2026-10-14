"""End-to-end matching of two raw tables: SQL blocking → pairwise model judgements → merge by probability, evaluated against the complete human mapping.

    python experiments/table_match/pipeline.py block abt-buy            # blocking; reports how many true matches the candidates keep (no model call)
    python experiments/table_match/pipeline.py judge abt-buy            # Jev judges every candidate pair
    python experiments/table_match/pipeline.py judge abt-buy --llm deepseek-v4.1-flash
    python experiments/table_match/pipeline.py eval abt-buy             # end-to-end precision / recall / F1, cost, time
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "entity_match"))
from run import KINDS  # noqa: E402  the same question version as the entity-matching experiment, not adapted to this data

from kevdb import JevClient, Task, run_task  # noqa: E402
from kevdb.cache import Cache  # noqa: E402
from kevdb.cli import load_dotenv  # noqa: E402
from kevdb.llm import LLMClient  # noqa: E402

OUT = HERE.parents[1] / "out" / "table_match"
DATA = HERE / "data"

# Per dataset: left table, right table, mapping file, and each one's id column and the columns shown to the model
DATASETS = {
    "abt-buy": {
        "kind": "product",
        "left": ("Abt-Buy/Abt.csv", "latin-1", "id", ["name", "description", "price"]),
        "right": ("Abt-Buy/Buy.csv", "utf-8", "id", ["name", "description", "manufacturer", "price"]),
        "truth": ("Abt-Buy/abt_buy_perfectMapping.csv", "idAbt", "idBuy"),
        # Abt's descriptions are long and drown the query; querying by product name alone raises the true matches kept by the top 5 candidates from 87.9% to 96.1%
        "query": ["name"],
    },
    "dblp-scholar": {
        "kind": "publication",
        "left": ("DBLP-Scholar/DBLP1.csv", "latin-1", "id", ["title", "authors", "venue", "year"]),
        "right": ("DBLP-Scholar/Scholar.csv", "utf-8-sig", "id", ["title", "authors", "venue", "year"]),
        "truth": ("DBLP-Scholar/DBLP-Scholar_perfectMapping.csv", "idDBLP", "idScholar"),
    },
}
TOKEN = re.compile(r"[a-z0-9]{2,}")


def load_table(spec: tuple) -> dict[str, dict[str, str]]:
    path, encoding, id_col, fields = spec
    with (DATA / path).open(encoding=encoding, newline="") as f:
        return {row[id_col]: {k: row[k].strip() for k in fields if row[k].strip()} for row in csv.DictReader(f)}


def load_truth(spec: tuple) -> set[tuple[str, str]]:
    path, left_col, right_col = spec
    with (DATA / path).open(encoding="utf-8", newline="") as f:
        return {(row[left_col], row[right_col]) for row in csv.DictReader(f)}


def tokens(record: dict[str, str], fields: list[str] | None = None, limit: int = 60) -> list[str]:
    text = " ".join(v for k, v in record.items() if fields is None or k in fields).lower()
    # Model numbers appear as "PS-LX350H" or "PSLX350H": words containing digits also count once with punctuation removed, so both spellings match
    glued = " ".join(re.sub(r"[^a-z0-9]", "", word) for word in text.split() if any(c.isdigit() for c in word))
    return list(dict.fromkeys(TOKEN.findall(text + " " + glued)))[:limit]


def block(left: dict, right: dict, k: int, query_fields: list[str] | None = None) -> dict[str, list[str]]:
    """For each left record, find the k most textually similar right records. Uses SQLite full-text search (BM25), no model."""
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE docs USING fts5(id UNINDEXED, text)")
    db.executemany("INSERT INTO docs VALUES (?, ?)", ((rid, " ".join(tokens(r, limit=10_000))) for rid, r in right.items()))
    candidates = {}
    for lid, record in left.items():
        query = " OR ".join(f'"{t}"' for t in tokens(record, query_fields))
        rows = db.execute("SELECT id FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?", (query, k)) if query else []
        candidates[lid] = [r[0] for r in rows]
    return candidates


def candidates_path(name: str) -> Path:
    return OUT / f"{name}.candidates.json"


def cmd_block(args: argparse.Namespace) -> None:
    spec = DATASETS[args.dataset]
    left, right, truth = load_table(spec["left"]), load_table(spec["right"]), load_truth(spec["truth"])
    wide = block(left, right, max(args.k, 20), spec.get("query"))
    print(f"{args.dataset}: {len(left):,} left × {len(right):,} right = {len(left) * len(right):,} possible pairs, {len(truth):,} true matches")
    print(f"{'top k per row':>14}{'candidates':>12}{'matches kept':>14}")
    for k in (1, 3, 5, 10, 20):
        kept = sum(b in wide[a][:k] for a, b in truth)
        print(f"{k:>14}{sum(len(c[:k]) for c in wide.values()):>12,}{kept / len(truth):>13.1%}")
    OUT.mkdir(parents=True, exist_ok=True)
    candidates_path(args.dataset).write_text(json.dumps({a: c[: args.k] for a, c in wide.items()}))
    print(f"using k={args.k} → {candidates_path(args.dataset)}")


def load_candidates(name: str, k: int | None = None) -> dict[str, list[str]]:
    """Candidates are stored from most to least similar, so taking the first k means "look only at the k most similar per row"."""
    return {a: bs[:k] for a, bs in json.loads(candidates_path(name).read_text()).items()}


def build_pairs(name: str, k: int | None = None) -> list[dict]:
    spec = DATASETS[name]
    left, right, truth = load_table(spec["left"]), load_table(spec["right"]), load_truth(spec["truth"])
    return [
        {"id": f"{a}|{b}", "record_a": left[a], "record_b": right[b], "label": int((a, b) in truth)}
        for a, bs in load_candidates(name, k).items() for b in bs
    ]


async def cmd_judge(args: argparse.Namespace) -> None:
    load_dotenv()
    pairs = build_pairs(args.dataset, args.k)[: args.limit]
    task = Task(
        name=f"dedup_{args.dataset}", input="", questions={"same": KINDS[DATASETS[args.dataset]["kind"]]},
        # Jev uses the official API; only the --llm comparison models go through the third-party platform
        state_fields=["record_a", "record_b"], provider="typesafe", model="jev-1.13.0", concurrency=args.concurrency,
    )
    if args.llm:
        client, label = LLMClient(provider="opencode", model=args.llm, timeout=120.0, reasoning=None if args.reasoning == "default" else args.reasoning), f"{args.llm}.{args.reasoning}"
    else:
        client, label = JevClient(provider=task.provider, model=task.model), task.model
    async with client:
        results, stats = await run_task(task, pairs, client, Cache())
    rows = [{"id": p["id"], "label": p["label"], "p": r.get("same_p"), "error": r.get("_error")} for p, r in zip(pairs, results) if not r.get("_pending")]
    stem = OUT / f"{args.dataset}.{label}"
    # When the same model runs again with another k, keep existing results and only add the new pairs
    existing = stem.with_name(stem.name + ".jsonl")
    seen = {r["id"] for r in rows}
    if existing.exists():
        rows += [r for r in map(json.loads, existing.read_text().splitlines()) if r["id"] not in seen]
    existing.write_text("".join(json.dumps(r) + "\n" for r in rows))
    fresh = len(rows) - stats.cached - stats.errors
    summary = {"pairs": len(rows), "fresh": fresh, "cached": stats.cached, "errors": stats.errors, "pending": stats.pending,
               "seconds": round(stats.seconds, 1), "latency_p50": stats.latency(0.5),
               "input_tokens": stats.input_tokens, "output_tokens": stats.output_tokens, "cost_usd": round(stats.cost_usd, 5)}
    if fresh:
        with stem.with_name(stem.name + ".runs.jsonl").open("a") as f:
            f.write(json.dumps(summary) + "\n")
    print(f"{args.dataset} / {label}: {summary}")


def cmd_eval(args: argparse.Namespace) -> None:
    truth = load_truth(DATASETS[args.dataset]["truth"])
    blocked = {(a, b) for a, bs in load_candidates(args.dataset, args.k).items() for b in bs}
    ids = {f"{a}|{b}" for a, b in blocked}
    print(f"{args.dataset} (top {args.k or 'all'} candidates per row): {len(truth):,} true matches, {len(blocked):,} candidate pairs, "
          f"blocking kept {len(truth & blocked) / len(truth):.1%}")
    for path in sorted(OUT.glob(f"{args.dataset}.*.jsonl")):
        if path.name.endswith(".runs.jsonl"):
            continue
        rows = [r for r in map(json.loads, path.read_text().splitlines()) if r["id"] in ids]
        answered = [r for r in rows if r["p"] is not None]
        if not answered:
            continue
        runs_path = path.with_name(path.name.replace(".jsonl", ".runs.jsonl"))
        runs = [json.loads(line) for line in runs_path.read_text().splitlines()] if runs_path.exists() else []
        fresh = sum(r["fresh"] for r in runs) or 1
        cost, seconds = sum(r["cost_usd"] for r in runs), sum(r["seconds"] for r in runs)
        print(f"\n== {path.name.removeprefix(args.dataset + '.').removesuffix('.jsonl')}  (judged {len(answered):,} / {len(blocked):,} pairs, {len(rows) - len(answered)} failed)")
        # The same model may have judged more candidates (a larger k); scale by unit price and rate to the current range
        rate = fresh / max(seconds, 1e-9)
        print(f"   these {len(answered):,} pairs: about ${cost / fresh * len(answered):.3f} (${cost / fresh * 1000:.3f} per 1,000 pairs), "
              f"about {len(answered) / rate / 60:.1f} minutes ({rate:.1f} pairs per second)")
        print(f"   {'thr':>5}{'matches':>8}{'prec':>8}{'recall':>8}{'F1':>7}")
        for t in (0.3, 0.5, 0.7, 0.9):
            predicted = [r for r in answered if r["p"] >= t]
            tp = sum(r["label"] for r in predicted)
            precision, recall = tp / max(len(predicted), 1), tp / len(truth)
            print(f"   {t:>6.1f}{len(predicted):>10,}{precision:>10.1%}{recall:>9.1%}{2 * precision * recall / max(precision + recall, 1e-9):>8.1%}")
        unsure = [r for r in answered if 0.1 < r["p"] < 0.9]
        if len({r["p"] for r in answered}) > 2:
            sure = [r for r in answered if not 0.1 < r["p"] < 0.9]
            print(f"   unsure (probability between 0.1 and 0.9): {len(unsure) / len(answered):.1%}; accuracy on the other {len(sure):,} pairs "
                  f"{sum((r['p'] >= 0.5) == bool(r['label']) for r in sure) / len(sure):.2%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("block"); b.add_argument("dataset", choices=list(DATASETS)); b.add_argument("--k", type=int, default=5)
    j = sub.add_parser("judge"); j.add_argument("dataset", choices=list(DATASETS))
    j.add_argument("--llm"); j.add_argument("--reasoning", default="none"); j.add_argument("--concurrency", type=int, default=16); j.add_argument("--limit", type=int)
    j.add_argument("--k", type=int, help="judge only the top k candidates per row (default: all)")
    e = sub.add_parser("eval"); e.add_argument("dataset", choices=list(DATASETS))
    e.add_argument("--k", type=int, help="evaluate on the top k candidates only")
    args = parser.parse_args()
    if args.cmd == "block":
        cmd_block(args)
    elif args.cmd == "judge":
        asyncio.run(cmd_judge(args))
    else:
        cmd_eval(args)


if __name__ == "__main__":
    main()
