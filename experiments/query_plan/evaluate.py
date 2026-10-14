"""Evaluate plan selection: how accurate the model's judgements are, and whether the whole workload gets faster or slower when plans are chosen by them.

Policy: within each query, take the candidate with the highest probability of being faster than the default; use it if that probability reaches the threshold, otherwise use the default plan.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from headroom import MARGIN, load  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "out" / "query_plan"
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def auc(scored: list[tuple[float, bool]]) -> float | None:
    """Probability that a random candidate that really is faster scores higher than a random one that is not. 0.5 is chance."""
    pos = [s for s, y in scored if y]
    neg = [s for s, y in scored if not y]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def apply_policy(decisions: list[dict], default_ms: dict[str, float], threshold: float) -> dict:
    """Total runtime when plans are chosen with this threshold, and the number of queries clearly faster / slower than the default plan."""
    best: dict[str, dict] = {}
    for d in decisions:
        if d["p"] is not None and d["p"] >= threshold and d["p"] > best.get(d["query"], {"p": -1})["p"]:
            best[d["query"]] = d
    chosen = {q: best[q]["alt_ms"] if q in best else ms for q, ms in default_ms.items()}
    return {
        "total_s": sum(chosen.values()) / 1000,
        "switched": len(best),
        "faster": sum(chosen[q] * MARGIN < ms for q, ms in default_ms.items()),
        "slower": sum(chosen[q] > ms * MARGIN for q, ms in default_ms.items()),
    }


def report(path: Path, default_ms: dict[str, float]) -> None:
    decisions = [json.loads(line) for line in path.read_text().splitlines()]
    answered = [d for d in decisions if d["p"] is not None]
    truth = [(d["p"], d["alt_ms"] < d["default_ms"]) for d in answered]
    base = sum(y for _, y in truth) / len(truth)
    accuracy = sum((p >= 0.5) == y for p, y in truth) / len(truth)
    score = auc(truth)
    print(f"\n== {path.name.removeprefix('decisions.').removesuffix('.jsonl')}")
    print(f"  {len(answered)} questions ({len(decisions) - len(answered)} failed); the candidate really is faster in {base:.0%}")
    print(f"  accuracy {accuracy:.0%} (always guessing \"not faster\" gives {1 - base:.0%}), AUC {score:.2f}" if score is not None else f"  accuracy {accuracy:.0%}")
    default_total = sum(default_ms.values()) / 1000
    print(f"  {'thr':>6}{'switched':>9}{'total':>10}{'vs default':>11}{'faster':>8}{'slower':>8}")
    for t in THRESHOLDS:
        r = apply_policy(answered, default_ms, t)
        print(f"  {t:>6.1f}{r['switched']:>10}{r['total_s']:>10.1f}s{r['total_s'] / default_total:>9.2f}x{r['faster']:>10}{r['slower']:>10}")


def main() -> None:
    truth = load()
    default_ms = {q: arms["default"]["ms"] for q, arms in truth.items()}
    default_total = sum(default_ms.values()) / 1000
    oracle = sum(min(r["ms"] for r in arms.values()) for arms in truth.values()) / 1000
    print(f"{len(truth)} queries. Default plans take {default_total:.1f}s in total, the hindsight optimum {oracle:.1f}s ({oracle / default_total:.2f}x)")

    # Reference: pick a random candidate (default included) for each query
    rng = random.Random(0)
    trials = [sum(rng.choice(list({r["plan_key"]: r["ms"] for r in arms.values()}.values())) for arms in truth.values()) / 1000 for _ in range(200)]
    print(f"random candidate plan: {sum(trials) / len(trials):.1f}s on average ({sum(trials) / len(trials) / default_total:.2f}x)")

    files = sorted(OUT.glob("decisions.*.jsonl")) if len(sys.argv) < 2 else [Path(p) for p in sys.argv[1:]]
    for path in files:
        report(path, default_ms)


if __name__ == "__main__":
    main()
