"""Reference baseline: a logistic regression on plan features that predicts whether a candidate plan is faster than the default.

It stands for the most basic version of the classic route of learning from this database's own execution feedback (a minimal stand-in for Bao-style work)
and answers one question: is the signal that separates good plans from bad ones present in the plans at all? Jev and the LLMs are zero-shot and it is not,
so this is a reference, not a fair comparison.

Leave-one-template-out cross-validation (1a/1b/1c belong to template 1), so near-duplicate queries of one template cannot leak answers.
Pure Python with no extra dependencies. The output format matches decide.py, so evaluate.py can evaluate it directly.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from headroom import load  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "out" / "query_plan"
NODES = ["Nested Loop", "Hash Join", "Merge Join", "Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan"]


def walk(plan: dict):
    yield plan
    for child in plan.get("Plans", []):
        yield from walk(child)


def plan_features(plan: dict) -> list[float]:
    nodes = list(walk(plan))
    counts = [sum(n["Node Type"] == t for n in nodes) for t in NODES]
    return [*counts, math.log10(plan["Total Cost"] + 1), math.log10(max(n["Plan Rows"] for n in nodes) + 1)]


def build(truth: dict[str, dict[str, dict]]) -> list[dict]:
    rows = []
    for name, arms in truth.items():
        default = arms["default"]
        seen = {default["plan_key"]}
        for arm, record in arms.items():
            if record["plan_key"] in seen:
                continue
            seen.add(record["plan_key"])
            d, a = plan_features(default["plan"]), plan_features(record["plan"])
            rows.append({
                "id": f"{name}:{arm}", "query": name, "arm": arm, "template": int(name[:-1]),
                "default_ms": default["ms"], "alt_ms": record["ms"],
                # Features of the default plan + the candidate's difference from the default
                "x": d + [ai - di for ai, di in zip(a, d)],
                "y": record["ms"] < default["ms"],
            })
    return rows


def fit(xs: list[list[float]], ys: list[bool], l2: float = 1.0, steps: int = 800, lr: float = 0.1):
    """Logistic regression with standardisation and L2 regularisation, full-batch gradient descent. Returns a prediction function."""
    n, k = len(xs), len(xs[0])
    mean = [sum(x[j] for x in xs) / n for j in range(k)]
    std = [math.sqrt(sum((x[j] - mean[j]) ** 2 for x in xs) / n) or 1.0 for j in range(k)]
    z = [[(x[j] - mean[j]) / std[j] for j in range(k)] for x in xs]
    w, b = [0.0] * k, 0.0
    for _ in range(steps):
        ps = [1 / (1 + math.exp(-(b + sum(wj * xj for wj, xj in zip(w, x))))) for x in z]
        errs = [p - y for p, y in zip(ps, ys)]
        w = [wj - lr * (sum(e * x[j] for e, x in zip(errs, z)) / n + l2 * wj / n) for j, wj in enumerate(w)]
        b -= lr * sum(errs) / n

    def predict(x: list[float]) -> float:
        return 1 / (1 + math.exp(-(b + sum(wj * (xj - m) / s for wj, xj, m, s in zip(w, x, mean, std)))))

    return predict


def main() -> None:
    rows = build(load())
    out = []
    for template in sorted({r["template"] for r in rows}):
        train = [r for r in rows if r["template"] != template]
        predict = fit([r["x"] for r in train], [r["y"] for r in train])
        for r in rows:
            if r["template"] == template:
                out.append({k: r[k] for k in ("id", "query", "arm", "default_ms", "alt_ms")} | {"p": round(predict(r["x"]), 4), "error": None})
    path = OUT / "decisions.learned-logreg.cv.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"{len(out)} questions, leave-one-out over {len({r['template'] for r in rows})} templates → {path}")


if __name__ == "__main__":
    main()
