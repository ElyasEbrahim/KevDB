"""Send the same requests through the official Jev API and through the third-party platform OpenCode, and compare latency, throughput and how often each is rate limited.

To see rate limiting clearly there are no retries and no cache: each request is sent once, and every rejection is counted.
Results are appended to out/bench/route_speed.jsonl, one summary line per (route, concurrency).

    python experiments/table_match/route_speed.py --concurrency 8 16
    python experiments/table_match/route_speed.py --report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "table_match"))
from pipeline import KINDS, build_pairs  # noqa: E402

from kevdb import JevAPIError, JevClient  # noqa: E402
from kevdb.cli import load_dotenv  # noqa: E402

OUT = ROOT / "out" / "bench" / "route_speed.jsonl"
ROUTES = {"typesafe": "jev-1.13.0", "opencode": "jev-1.13"}
COOLDOWN = 20  # pause between rounds so one round's rate-limit window does not affect the next


async def run_cell(route: str, concurrency: int, pairs: list[dict]) -> dict:
    question = {"same": KINDS["product"].payload()}
    gate = asyncio.Semaphore(concurrency)
    latencies, statuses, answers, tokens = [], [], {}, 0

    async with JevClient(provider=route, model=ROUTES[route], max_retries=0, max_wait=1e9) as client:
        async def one(pair: dict) -> None:
            nonlocal tokens
            async with gate:
                started = time.perf_counter()
                try:
                    resp = await client.ask({"record_a": pair["record_a"], "record_b": pair["record_b"]}, question)
                except JevAPIError as e:
                    statuses.append(e.status)
                    return
                latencies.append(time.perf_counter() - started)
                statuses.append(200)
                answers[pair["id"]] = resp["answers"]["same"]["noul"]
                tokens += resp["usage"]["input_tokens"]

        started = time.perf_counter()
        await asyncio.gather(*(one(p) for p in pairs))
        wall = time.perf_counter() - started

    ok = statuses.count(200)
    ordered = sorted(latencies)
    return {
        "route": route, "model": ROUTES[route], "concurrency": concurrency, "requests": len(pairs),
        "ok": ok, "rate_limited": statuses.count(429), "other_errors": len(pairs) - ok - statuses.count(429),
        "wall_s": round(wall, 2), "ok_per_s": round(ok / wall, 2),
        "latency_p50": round(statistics.median(ordered), 3) if ordered else None,
        "latency_p95": round(ordered[int(len(ordered) * 0.95)], 3) if ordered else None,
        "input_tokens": tokens, "answers": answers,
    }


def report() -> None:
    cells = [json.loads(line) for line in OUT.read_text().splitlines()]
    print(f"{'route':<10}{'requests':>9}{'conc':>5}{'ok':>6}{'limited':>8}{'other err':>10}{'ok/s':>8}{'p50':>8}{'p95':>8}{'total':>8}")
    for c in sorted(cells, key=lambda c: (c["requests"], c["concurrency"], c["route"])):
        print(f"{c['route']:<12}{c['requests']:>8}{c['concurrency']:>6}{c['ok']:>8}{c['rate_limited']:>8}{c['other_errors']:>9}"
              f"{c['ok_per_s']:>11.1f}{c['latency_p50'] or 0:>10.2f}s{c['latency_p95'] or 0:>9.2f}s{c['wall_s']:>9.0f}s")
    # Do the two routes give the same probability for the same record pair?
    by_route: dict[str, dict[str, float]] = {}
    for c in cells:
        by_route.setdefault(c["route"], {}).update(c["answers"])
    if len(by_route) == 2:
        a, b = by_route["typesafe"], by_route["opencode"]
        common = sorted(set(a) & set(b))
        diffs = [abs(a[i] - b[i]) for i in common]
        print(f"\n{len(common)} pairs answered on both routes: same decision {sum((a[i] >= .5) == (b[i] >= .5) for i in common) / len(common):.1%}, "
              f"identical probability {sum(d == 0 for d in diffs) / len(common):.1%}, mean difference {sum(diffs) / len(common):.4f}, max difference {max(diffs):.3f}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, nargs="*", default=[])
    parser.add_argument("--requests", type=int, default=400)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    if args.concurrency:
        pairs = random.Random(0).sample(build_pairs("abt-buy"), args.requests)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        for i, concurrency in enumerate(args.concurrency):
            # Alternate which route goes first, so the same one never always runs first
            for route in (list(ROUTES) if i % 2 == 0 else list(ROUTES)[::-1]):
                cell = await run_cell(route, concurrency, pairs)
                with OUT.open("a") as f:
                    f.write(json.dumps(cell) + "\n")
                print(f"{route} concurrency {concurrency}: ok {cell['ok']}/{cell['requests']}, rate limited {cell['rate_limited']}, "
                      f"{cell['ok_per_s']} per second, median latency {cell['latency_p50']}s", flush=True)
                await asyncio.sleep(COOLDOWN)
    if args.report:
        report()


if __name__ == "__main__":
    asyncio.run(main())
