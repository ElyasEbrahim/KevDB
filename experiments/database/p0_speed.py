"""Phase 0 route speed test: throughput and latency of the same requests at different concurrency levels, used to set the concurrency of the later phases.

The requests are yes/no questions about movie reviews (the most common kind in phase 1). Each concurrency level uses its own cache namespace, so every request is real.

    python experiments/database/p0_speed.py jev v41flash.off --n 300 --concurrency 8 16 32
"""

from __future__ import annotations

import argparse
import json
import time

import duckdb
from kevdb.cli import load_dotenv

from kevdb import jev_backend, llm_backend
from paths import DATA, OUT as RESULTS

OUT = RESULTS / "p0"
QUESTION = {"q": {"type": "noul", "instructions": "Is the movie review in `review` positive?"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backends", nargs="+", help="jev or an OpenRouter route name")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[8, 16, 32])
    args = parser.parse_args()
    load_dotenv()
    texts = [t for (t,) in duckdb.sql(
        f"SELECT DISTINCT reviewText FROM read_csv('{DATA}/sembench/files/movie/data/sf_2000/Reviews.csv') ORDER BY reviewText LIMIT {args.n}"
    ).fetchall()]
    items = [({"review": t}, QUESTION) for t in texts]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "speed.jsonl").open("a") as log:
        for name in args.backends:
            for c in args.concurrency:
                backend = jev_backend(concurrency=c) if name == "jev" else llm_backend(name, concurrency=c)
                backend.namespace = f"speed.{c}.{time.time():.0f}:"
                started = time.perf_counter()
                backend.ask_many(items)
                seconds = time.perf_counter() - started
                m = backend.meter
                row = {"backend": name, "concurrency": c, "requests": m.calls, "errors": m.errors, "seconds": round(seconds, 1),
                       "per_second": round((m.calls - m.errors) / seconds, 1), "latency_p50": round(m.latency(0.5), 2),
                       "latency_p95": round(m.latency(0.95), 2), "cost_usd": round(m.cost, 5)}
                log.write(json.dumps(row) + "\n")
                print(row, flush=True)
                backend.close()


if __name__ == "__main__":
    main()
