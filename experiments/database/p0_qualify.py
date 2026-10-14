"""Phase 0 qualification test: which LLM takes over the rows Jev is unsure about?

Data: the 637 Abt-Buy product pairs Jev is unsure about (probability between 0.1 and 0.9), plus a sample of 250 pairs it is sure about.
The same pairs as the earlier Opus run (experiments/table_match/strong.py), with human labels; the question and prompt are also the same, only the route differs.

Selection rule (fixed before seeing any result; this file's checksum was recorded in experiments/database/protocol/p0.txt before the run):
1. The main metric is accuracy on the 637 unsure pairs against the human labels. Parse failures and failed requests count as wrong.
2. Qualifying bar: no more than 3 points below Opus on the same pairs.
3. Among qualifying routes, take the one with the lowest actual cost per call; if costs are within 10%, take the lower median latency.
4. If none qualifies: take the most accurate one and state in the report that this limits the cascade's quality ceiling.

    python experiments/database/p0_qualify.py run            # runs each route, results go to out/database/p0/
    python experiments/database/p0_qualify.py report
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

from kevdb import Task
from kevdb.cli import load_dotenv

from kevdb import llm_backend
from paths import OUT as RESULTS, ROOT

sys.path.insert(0, str(ROOT / "experiments" / "table_match"))
import strong  # noqa: E402  pair selection identical to the Opus run

OUT = RESULTS / "p0"
# The 8 candidate routes of the qualification test (routes added to routes.py later are not among them)
CANDIDATES = ["v4pro.off", "v4pro.low", "v41flash.off", "v41flash.low", "qwen38flash.off", "qwen38flash.low", "luna.off", "luna.low"]
MARGIN = 0.03
CONCURRENCY = 16


def pairs() -> list[dict]:
    return strong.chosen_pairs()


def run(names: list[str]) -> None:
    load_dotenv()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pairs()
    question = {"same": strong.KINDS["product"].payload()}
    task = Task(name="qualify", input="", questions={}, state_fields=["record_a", "record_b"])
    items = [(task.state_of(row), question) for row in rows]
    for name in names:
        backend = llm_backend(name, concurrency=CONCURRENCY)
        started = time.perf_counter()
        answers = backend.ask_many(items)
        seconds = time.perf_counter() - started
        meter = backend.meter
        with (OUT / f"qualify.{name}.jsonl").open("w") as f:
            for row, a in zip(rows, answers):
                same = None if "_error" in a else a["answers"]["same"]
                f.write(json.dumps({
                    "id": row["id"], "label": row["label"], "stratum": row["stratum"],
                    "pred": None if same is None else int(same["noul"] >= 0.5), "p_true": same and same.get("p_true"),
                    "error": a.get("_error"), "latency": a.get("_latency"), **(a.get("usage") or {}),
                }) + "\n")
        summary = {"route": name, "model": backend.model, "pairs": len(rows), "fresh": meter.fresh, "errors": meter.errors,
                   "seconds": round(seconds, 1), "concurrency": CONCURRENCY, "cost_usd": round(meter.cost, 5)}
        (OUT / f"qualify.{name}.run.json").write_text(json.dumps(summary, ensure_ascii=False) + "\n")
        print(summary, flush=True)
        backend.close()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p + z * z / (2 * n) - half) / (1 + z * z / n), (p + z * z / (2 * n) + half) / (1 + z * z / n)


def mcnemar(a: list[bool], b: list[bool]) -> float:
    """Exact paired test: only the pairs where one is right and the other wrong; two-sided p-value."""
    only_a, only_b = sum(x and not y for x, y in zip(a, b)), sum(y and not x for x, y in zip(a, b))
    n, k = only_a + only_b, min(only_a, only_b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def median(xs: list[float]) -> float:
    return sorted(xs)[len(xs) // 2]


def report() -> None:
    earlier = ROOT / "out" / "table_match"
    read = lambda path: {r["id"]: r for r in map(json.loads, Path(path).read_text().splitlines())}
    opus, jev, zen = read(earlier / "abt-buy.claude-opus-5-5.jsonl"), read(earlier / "abt-buy.jev-1.13.0.jsonl"), read(earlier / "abt-buy.deepseek-v4.1-flash.none.jsonl")
    unsure = [i for i, r in opus.items() if r["stratum"] == "unsure"]
    sure = [i for i, r in opus.items() if r["stratum"] != "unsure"]
    label = {i: opus[i]["label"] for i in opus}
    right = lambda rows, i, field="p": rows[i].get(field) is not None and int(rows[i][field] >= 0.5) == label[i]
    opus_ok, jev_ok = [right(opus, i) for i in unsure], [right(jev, i) for i in unsure]
    opus_acc = sum(opus_ok) / len(unsure)
    bar = opus_acc - MARGIN
    print(f"on the {len(unsure)} unsure pairs: Opus {opus_acc:.1%}, bar {bar:.1%}; {len(sure)} sampled confident pairs")
    print(f"{'route':18s}{'accuracy (95% CI)':>24s}{'vs Opus p':>12s}{'vs Jev p':>11s}{'confident':>10s}{'fail':>5s}{'$ per 1k':>9s}{'p50/p95 latency':>16s}{'think tok':>10s}")
    for name, rows, field in (("Jev (reference)", jev, "p"), ("DeepSeek via Zen (ref.)", zen, "p"), ("Opus 5.5 (reference)", opus, "p")):
        ok = [right(rows, i, field) for i in unsure]
        lo, hi = wilson(sum(ok), len(ok))
        print(f"{name:18s}{sum(ok) / len(ok):>10.1%} ({lo:.1%}-{hi:.1%})")
    table = []
    for name in CANDIDATES:
        path = OUT / f"qualify.{name}.jsonl"
        if not path.exists():
            continue
        rows = read(path)
        ok = [right(rows, i, "pred") for i in unsure]
        lo, hi = wilson(sum(ok), len(ok))
        done = [r for r in rows.values() if r["error"] is None]
        per_call = sum(r["cost"] for r in done) / len(done)
        lat = [r["latency"] for r in done]
        row = {"route": name, "acc": sum(ok) / len(ok), "per_call": per_call, "latency": median(lat)}
        table.append(row)
        print(f"{name:18s}{row['acc']:>10.1%} ({lo:.1%}-{hi:.1%}){mcnemar(ok, opus_ok):>12.3f}{mcnemar(ok, jev_ok):>11.3f}"
              f"{sum(right(rows, i, 'pred') for i in sure) / len(sure):>9.1%}{len(rows) - len(done):>5d}{per_call * 1000:>9.3f}"
              f"{median(lat):>7.2f}/{sorted(lat)[int(len(lat) * 0.95)]:.2f}s{sum(r['reasoning_tokens'] for r in done) / len(done):>10.0f}")
    if not table:
        return
    ok = [r for r in table if r["acc"] >= bar]
    if ok:
        cheapest = min(r["per_call"] for r in ok)
        pick = min((r for r in ok if r["per_call"] <= cheapest * 1.1), key=lambda r: r["latency"])
        print(f"\n{len(ok)} qualify: {', '.join(r['route'] for r in ok)}. Chosen by the rule: {pick['route']}")
    else:
        pick = max(table, key=lambda r: r["acc"])
        print(f"\nno route reaches the bar. By the rule, the most accurate is chosen: {pick['route']} ({pick['acc']:.1%}), which limits the cascade's quality ceiling")


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        run(sys.argv[2:] or CANDIDATES)
    elif sys.argv[1:2] == ["report"]:
        report()
    else:
        sys.exit(__doc__)
