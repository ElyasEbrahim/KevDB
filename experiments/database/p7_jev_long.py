"""Phase 7 reference: the official Jev API answers the same three questions on the same 40 long documents. A comparison column only, with no threshold.

This is the only paid step of phase 7 (approved by the owner on 2026-10-12, about $0.03): 120 requests of about 6,000 tokens each.
Like the local models' off arm, each question is asked once at concurrency 1, so request times include no queueing; a cloud API has no "already read" text to reuse,
so every question sends and bills the whole document again. Jev's answers are only compared with human labels and never used to calibrate or train any model.

    python experiments/database/p7_jev_long.py run       # needs TYPESAFE_API_KEY in .env. Answered requests are cached, so re-running costs nothing
    python experiments/database/p7_jev_long.py check     # are all files there, are there failed requests? Counting only

With the environment variable P7_REHEARSAL=1 it asks made-up documents instead (about $0.003).
"""

from __future__ import annotations

import json
import sys
import time

from kevdb.cli import load_dotenv

from kevdb import jev_backend
from p7_data import NAMESPACE, SHARED, cache_set

OUT = SHARED / "jev"


def run() -> None:
    load_dotenv()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, questions, _, _ = cache_set("long")
    backend = jev_backend(concurrency=1, cache_path=OUT / "cache.sqlite", namespace=NAMESPACE + "long:")
    record = {"model": backend.model, "rows": len(rows), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "passes": []}
    for i, (name, q) in enumerate(questions.items(), 1):
        items = [(r["state"], {"q": q}) for r in rows]
        backend.reset()
        started = time.perf_counter()
        for _ in range(4):  # failed requests are not cached; asking again only re-asks the failures
            answers = backend.ask_many(items)
            if not backend.meter.errors:
                break
        seconds, m = time.perf_counter() - started, backend.meter
        with (OUT / f"long.{i}.jsonl").open("w") as f:
            for r, a in zip(rows, answers):
                ok = "_error" not in a
                f.write(json.dumps({"id": r["id"], "answers": a["answers"] if ok else None, "tokens": a["usage"]["input_tokens"] if ok else None,
                                    "latency": a["_latency"] if ok else None, "cost": a["usage"]["cost"] if ok else None, "error": a.get("_error")}) + "\n")
        # Cache hits are not re-timed: seconds is this pass's real time only when fresh == requests; each request's latency is always the one measured when it was first asked
        record["passes"].append({"question": name, "requests": m.calls, "fresh": m.fresh, "errors": m.errors, "seconds": round(seconds, 3), "input_tokens": m.input_tokens,
                                 "cost": m.cost, "latency_p50": m.latency(0.5), "latency_p95": m.latency(0.95)})
        print(f"Jev long documents, question {i} {name}: {m.calls} requests ({m.fresh} new, {m.errors} failed), {seconds:.1f} s, ${m.cost:.4f}", flush=True)
    record["spent"] = backend.spent
    record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    backend.close()
    # One line is appended per run: when resuming, the cost is the sum of each run's new requests
    with (OUT / "runs.jsonl").open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"new spending this run: ${record['spent']:.4f}")


def check() -> None:
    rows = cache_set("long")[0]
    log = OUT / "runs.jsonl"
    if not log.exists():
        sys.exit(f"missing: {log}")
    runs = [json.loads(line) for line in log.read_text().splitlines()]
    print(f"{'question':12s}{'rows':>6s}{'rows without answer':>20s}")
    for i, p in enumerate(runs[-1]["passes"], 1):
        got = [json.loads(line) for line in (OUT / f"long.{i}.jsonl").read_text().splitlines()]
        print(f"{p['question']:12s}{len(got):>8d}{sum(r['answers'] is None for r in got):>14d}" + ("" if len(got) == len(rows) else f"  should be {len(rows)} rows"))
    print(f"new spending in total: ${sum(r['spent'] for r in runs):.4f} ({len(runs)} runs)")


if __name__ == "__main__":
    command = {"run": run, "check": check}.get(sys.argv[1] if len(sys.argv) == 2 else "")
    if not command:
        sys.exit(__doc__)
    command()
