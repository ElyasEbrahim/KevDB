"""Ground-truth analysis: before involving any model, how much is there to win by choosing plans at all?

The default optimizer's total runtime is the opponent to beat; the hindsight optimum (every query chosen right) is the ceiling.
If the two are too close, comparing which model chooses better is meaningless.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

TRUTH = Path(__file__).resolve().parents[2] / "out" / "query_plan" / "truth.jsonl"
# Faster / slower than the default plan by more than this factor counts as "clearly"
MARGIN = 1.2


def load(path: Path = TRUTH) -> dict[str, dict[str, dict]]:
    """Return {query: {arm: record}}. Timed-out records use the timeout as their runtime (a lower bound on the real one)."""
    by_query: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        r["ms"] = r["timeout_s"] * 1000 if r["timed_out"] else r["median_ms"]
        by_query[r["query"]][r["arm"]] = r
    return dict(by_query)


def summarize(truth: dict[str, dict[str, dict]]) -> dict:
    arms = list(next(iter(truth.values())))
    total = {arm: sum(q[arm]["ms"] for q in truth.values()) for arm in arms}
    oracle = sum(min(r["ms"] for r in q.values()) for q in truth.values())
    wins = defaultdict(int)
    improvable, distinct_plans = 0, []
    for q in truth.values():
        best = min(q, key=lambda a: q[a]["ms"])
        wins[best] += 1
        improvable += q["default"]["ms"] > MARGIN * q[best]["ms"]
        distinct_plans.append(len({r["plan_key"] for r in q.values()}))
    return {
        "queries": len(truth),
        "total_s": {arm: round(ms / 1000, 1) for arm, ms in total.items()},
        "oracle_s": round(oracle / 1000, 1),
        "best_arm_counts": dict(wins),
        "improvable": improvable,
        "mean_distinct_plans": round(sum(distinct_plans) / len(distinct_plans), 2),
        "timeouts": {arm: sum(q[arm]["timed_out"] for q in truth.values()) for arm in arms},
    }


def main() -> None:
    truth = load()
    s = summarize(truth)
    default = s["total_s"]["default"]
    print(f"{s['queries']} queries, {s['mean_distinct_plans']} distinct candidate plans per query on average")
    print(f"\n{'policy':<28}{'total':>10}{'vs default':>11}{'timeouts':>9}")
    for arm, seconds in s["total_s"].items():
        print(f"{'always ' + arm:<28}{seconds:>9.1f}s{seconds / default:>9.2f}x{s['timeouts'][arm]:>8}")
    print(f"{'hindsight optimum (ceiling)':<28}{s['oracle_s']:>9.1f}s{s['oracle_s'] / default:>9.2f}x")
    print(f"\nqueries where the default plan is more than {MARGIN}x slower than the best: {s['improvable']} / {s['queries']}")
    print("times each arm is fastest:", s["best_arm_counts"])
    worst = sorted(truth.items(), key=lambda kv: kv[1]["default"]["ms"] - min(r["ms"] for r in kv[1].values()), reverse=True)[:10]
    print("\nthe 10 queries where the default plan loses most (default → best):")
    for name, q in worst:
        best = min(q, key=lambda a: q[a]["ms"])
        print(f"  {name:>4}: {q['default']['ms']:>9.0f}ms → {q[best]['ms']:>8.0f}ms  ({best})")


if __name__ == "__main__":
    main()
