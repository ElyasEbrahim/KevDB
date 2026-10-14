"""Collect the ground truth: the plan and the actual runtime of every JOB query under every set of optimizer switches (arm).

Writes out/query_plan/truth.jsonl, one line per (query, arm). Can be interrupted and resumed; existing combinations are skipped.
This step calls no model; it only talks to the local PostgreSQL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import psycopg

DSN = "postgresql://job:job@127.0.0.1:5434/imdb"
HERE = Path(__file__).resolve().parent
QUERIES = HERE / "data" / "job-queries"
OUT = HERE.parents[1] / "out" / "query_plan" / "truth.jsonl"

# Each arm turns off some operators to force the optimizer into a different plan. default is the database's own choice.
TOGGLES = ["enable_nestloop", "enable_hashjoin", "enable_mergejoin", "enable_indexscan", "enable_seqscan", "enable_indexonlyscan"]
ARMS: dict[str, list[str]] = {
    "default": [],
    "no_nestloop": ["enable_nestloop"],
    "no_mergejoin": ["enable_mergejoin"],
    "no_mergejoin_indexscan": ["enable_mergejoin", "enable_indexscan"],
    "hash_only": ["enable_nestloop", "enable_mergejoin", "enable_indexscan"],
}


def apply_arm(cur: psycopg.Cursor, arm: str, timeout_ms: int) -> None:
    for toggle in TOGGLES:
        cur.execute(f"SET {toggle} = {'off' if toggle in ARMS[arm] else 'on'}")
    cur.execute(f"SET statement_timeout = {timeout_ms}")


def explain(cur: psycopg.Cursor, sql: str) -> tuple[dict, str]:
    cur.execute("EXPLAIN (FORMAT JSON) " + sql)
    plan = cur.fetchone()[0][0]["Plan"]
    cur.execute("EXPLAIN " + sql)
    return plan, "\n".join(row[0] for row in cur.fetchall())


def execute(cur: psycopg.Cursor, sql: str) -> tuple[float, float] | None:
    """Return (execution ms, planning ms), or None on timeout. TIMING OFF avoids the per-node timing overhead."""
    try:
        cur.execute("EXPLAIN (ANALYZE, TIMING OFF, FORMAT JSON) " + sql)
    except psycopg.errors.QueryCanceled:
        return None
    result = cur.fetchone()[0][0]
    return result["Execution Time"], result["Planning Time"]


def measure(cur: psycopg.Cursor, sql: str, runs: int) -> dict:
    """One warm-up run (not counted), then `runs` timed runs; return the median. Stop at the first timeout."""
    times: list[float] = []
    planning = None
    for i in range(runs + 1):
        result = execute(cur, sql)
        if result is None:
            return {"timed_out": True, "exec_ms": times}
        if i > 0:
            times.append(round(result[0], 2))
            planning = round(result[1], 2)
    return {"timed_out": False, "exec_ms": times, "median_ms": statistics.median(times), "planning_ms": planning}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="timed runs per plan (plus one warm-up)")
    parser.add_argument("--max-timeout", type=int, default=300, help="timeout for a single execution (seconds)")
    parser.add_argument("--only", nargs="*", help="run only these queries, e.g. 1a 17e")
    args = parser.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {(r["query"], r["arm"]) for r in map(json.loads, OUT.read_text().splitlines())}
    files = sorted(QUERIES.glob("[0-9]*.sql"), key=lambda p: (int(p.stem[:-1]), p.stem[-1]))
    if args.only:
        files = [f for f in files if f.stem in args.only]

    started = time.time()
    with psycopg.connect(DSN, autocommit=True) as conn, conn.cursor() as cur, OUT.open("a") as out:
        for n, path in enumerate(files, 1):
            sql = path.read_text().strip().rstrip(";")
            if all((path.stem, arm) in done for arm in ARMS):
                continue
            seen: dict[str, tuple[str, dict]] = {}
            default_ms = None
            for arm in ARMS:
                # Non-default arms need not wait long: more than 10x slower than the default plan counts as a timeout
                limit_s = args.max_timeout if default_ms is None else min(args.max_timeout, max(15, default_ms * 10 / 1000))
                apply_arm(cur, arm, int(limit_s * 1000))
                plan, plan_text = explain(cur, sql)
                key = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16]
                # If the switches did not change the plan, reuse the existing timing
                if key in seen:
                    same_as, timing = seen[key]
                else:
                    same_as, timing = None, measure(cur, sql, args.runs)
                    seen[key] = (arm, timing)
                record = {
                    "query": path.stem, "arm": arm, "plan_key": key, "timeout_s": round(limit_s, 1),
                    "est_cost": plan["Total Cost"], "est_rows": plan["Plan Rows"], "plan_text": plan_text, "plan": plan,
                    **timing,
                }
                if same_as:
                    record["same_plan_as"] = same_as
                if arm == "default" and not timing["timed_out"]:
                    default_ms = timing["median_ms"]
                if (path.stem, arm) not in done:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                shown = "timeout" if timing["timed_out"] else f"{timing['median_ms']:.0f}ms"
                print(f"[{n}/{len(files)}] {path.stem:>4} {arm:<24} {shown:>10}  elapsed {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
