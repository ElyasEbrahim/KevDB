"""Gold set: sampling, comparing predictions, and computing accuracy, calibration and threshold sweeps."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from .io import Row
from .pipeline import Task
from .questions import Question

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
BUCKETS = ((0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01))


def sample_for_gold(pred: list[Row], id_field: str, class_field: str, n: int, min_per_class: int, seed: int) -> list[Row]:
    """Draw n rows uniformly at random (uniform=True, used for overall metrics),
    then top up predicted classes with fewer than min_per_class rows (uniform=False, used only for that class's precision)."""
    rng = random.Random(seed)
    uniform = rng.sample(pred, min(n, len(pred)))
    chosen = {r[id_field] for r in uniform}
    picked = [{**r, "uniform": True} for r in uniform]
    counts = Counter(r[class_field] for r in uniform)
    for label in sorted({r[class_field] for r in pred}):
        pool = [r for r in pred if r[class_field] == label and r[id_field] not in chosen]
        rng.shuffle(pool)
        for r in pool[: max(0, min_per_class - counts[label])]:
            chosen.add(r[id_field])
            picked.append({**r, "uniform": False})
    return picked


def confidence(name: str, q: Question, row: Row) -> float:
    if q.type == "noul":
        p = row[f"{name}_p"]
        return max(p, 1 - p)
    return row[f"{name}_conf"]


def is_correct(name: str, q: Question, pred: Row, gold: Row, lenient: bool = True) -> bool:
    """lenient: alternatives listed in the gold column <question>_alt also count as correct (for rows that genuinely fit two classes)."""
    if q.type == "score":
        return round(pred[name]) == gold[name]
    return pred[name] == gold[name] or (lenient and pred[name] in gold.get(f"{name}_alt", []))


def evaluate(task: Task, pred: list[Row], gold: list[Row]) -> dict[str, dict[str, Any]]:
    """One report per question. Overall metrics use the uniform rows only; per-class precision uses every gold row.
    Questions left blank in the gold set (too uncertain to label) are not scored. Everything except accuracy_strict uses the lenient rule."""
    by_id = {r[task.id_field]: r for r in pred}
    pairs = [(by_id[g[task.id_field]], g) for g in gold if g[task.id_field] in by_id]
    uniform = [(p, g) for p, g in pairs if g.get("uniform", True)]
    report: dict[str, dict[str, Any]] = {}
    for name, q in task.questions.items():
        scored = [(confidence(name, q, p), is_correct(name, q, p, g)) for p, g in uniform if name in g]
        if not scored:
            continue
        out: dict[str, Any] = {
            "n": len(scored),
            "accuracy": _mean(ok for _, ok in scored),
            "accuracy_strict": _mean(is_correct(name, q, p, g, lenient=False) for p, g in uniform if name in g),
        }
        out["calibration"] = [
            {"range": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": len(hit), "accuracy": _mean(hit), "mean_conf": _mean(c for c, _ in scored if lo <= c < hi)}
            for lo, hi in BUCKETS
            if (hit := [ok for c, ok in scored if lo <= c < hi])
        ]
        out["thresholds"] = [
            {"review_below": t, "auto": len(kept) / len(scored), "auto_accuracy": _mean(kept)}
            for t in THRESHOLDS
            if (kept := [ok for c, ok in scored if c >= t])
        ]
        if q.type == "score":
            out["mae"] = _mean(abs(p[name] - g[name]) for p, g in uniform if name in g)
            out["within_1"] = _mean(abs(round(p[name]) - g[name]) <= 1 for p, g in uniform if name in g)
        else:
            labels = sorted({str(g[name]) for _, g in pairs if name in g} | {str(p[name]) for p, _ in pairs})
            out["per_class"] = {
                label: {
                    "precision": _mean(is_correct(name, q, p, g) for p, g in pairs if name in g and str(p[name]) == label),
                    "predicted": sum(str(p[name]) == label for p, g in pairs if name in g),
                    "recall": _mean(str(p[name]) == label for p, g in uniform if name in g and str(g[name]) == label),
                    "actual": sum(str(g[name]) == label for _, g in uniform if name in g),
                }
                for label in labels
            }
            out["confusions"] = Counter(
                (str(g[name]), str(p[name])) for p, g in uniform if name in g and not is_correct(name, q, p, g)
            ).most_common(8)
        report[name] = out
    return report


def _mean(values: Any) -> float | None:
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else None


def format_report(report: dict[str, dict[str, Any]]) -> str:
    def pct(x: float | None) -> str:
        return "  - " if x is None else f"{x:4.0%}"

    lines = []
    for name, r in report.items():
        lines.append(f"\n[{name}]  n={r['n']}  accuracy {pct(r['accuracy'])} (strict {pct(r['accuracy_strict'])})" + (
            f"  MAE {r['mae']:.2f}  within 1 level {pct(r['within_1'])}" if "mae" in r else ""))
        lines.append("  calibration (confidence bin: rows / mean confidence / actual accuracy)")
        lines += [f"    {b['range']}: {b['n']:>3}  {b['mean_conf']:.2f}  {pct(b['accuracy'])}" for b in r["calibration"]]
        lines.append("  threshold sweep (review_below: share auto-accepted / accuracy of the auto-accepted rows)")
        lines += [f"    {t['review_below']:.1f}: {pct(t['auto'])}  {pct(t['auto_accuracy'])}" for t in r["thresholds"]]
        if "per_class" in r:
            lines.append("  per class (precision [predicted] / recall [actual])")
            lines += [f"    {label:<13} {pct(c['precision'])} [{c['predicted']:>3}]   {pct(c['recall'])} [{c['actual']:>3}]"
                      for label, c in r["per_class"].items()]
            if r["confusions"]:
                lines.append("  common errors (gold → predicted): " + "  ".join(f"{g}→{p}×{k}" for (g, p), k in r["confusions"]))
    return "\n".join(lines)
