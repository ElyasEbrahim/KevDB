"""Entity matching evaluation: precision, recall and F1 against human labels.

Jev gives probabilities, so besides the fixed threshold of 0.5 we also report "choose the best threshold on the validation split, then apply it to the test split".
Thresholds are chosen on the validation split only; the test split takes no part in any tuning.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "out" / "entity_match"
DATASETS = ["abt-buy", "amazon-google", "walmart-amazon", "dblp-acm"]


def load(name: str) -> list[dict]:
    path = OUT / f"{name}.jsonl"
    return [r for r in map(json.loads, path.read_text().splitlines()) if r["p"] is not None] if path.exists() else []


def prf(rows: list[dict], threshold: float = 0.5) -> tuple[float, float, float]:
    tp = sum(r["p"] >= threshold and r["label"] for r in rows)
    fp = sum(r["p"] >= threshold and not r["label"] for r in rows)
    fn = sum(r["p"] < threshold and r["label"] for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def best_threshold(rows: list[dict]) -> float:
    return max((t / 100 for t in range(5, 100, 5)), key=lambda t: prf(rows, t)[2])


def average_precision(rows: list[dict]) -> float:
    """Sort all pairs by probability, highest first; the closer to 1, the higher the matching pairs rank. Measures ranking quality, independent of any threshold."""
    ranked = sorted(rows, key=lambda r: -r["p"])
    hits, total = 0, 0.0
    for i, r in enumerate(ranked, 1):
        if r["label"]:
            hits += 1
            total += hits / i
    return total / max(hits, 1)


def spend(name: str) -> dict:
    path = OUT / f"{name}.runs.jsonl"
    runs = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return {k: sum(r[k] for r in runs) for k in ("fresh", "input_tokens", "output_tokens", "cost_usd", "seconds")}


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "jev-1.13"
    print(f"model {model}, against human labels\n")
    print(f"{'dataset':<16}{'pairs':>6}{'matches':>8} | {'thr 0.5: P':>10}{'R':>6}{'F1':>7} | {'valid thr':>10}{'P':>6}{'R':>6}{'F1':>7} | {'AP':>6}")
    for d in DATASETS:
        test, valid = load(f"{d}.test.{model}"), load(f"{d}.valid.{model}")
        if not test:
            continue
        p, r, f = prf(test)
        line = f"{d:<18}{len(test):>6}{sum(x['label'] for x in test) / len(test):>9.0%}  | {p:>11.1%}{r:>7.1%}{f:>8.1%} |"
        if valid:
            t = best_threshold(valid)
            p2, r2, f2 = prf(test, t)
            line += f"{t:>12.2f}{p2:>9.1%}{r2:>7.1%}{f2:>8.1%} |"
        print(line + f"{average_precision(test):>8.3f}")
    total = {k: sum(spend(f"{d}.test.{model}")[k] for d in DATASETS) for k in ("fresh", "input_tokens", "cost_usd")}
    if total["fresh"]:
        print(f"\ntest splits in total: {total['fresh']:,} requests, {total['input_tokens']:,} input tokens, ${total['cost_usd']:.3f}"
              f" (${total['cost_usd'] / total['fresh'] * 1000:.3f} per 1,000 pairs)")


if __name__ == "__main__":
    main()
