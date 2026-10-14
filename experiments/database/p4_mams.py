"""Pre-check for phase 4, part two: the 21 predicates on MAMS restaurant reviews. Reads the judgements collected by p4_collect.py only and calls no model.

The three questions and the decision rules were written before collection, see experiments/database/protocol/p4.txt:
A. Is the LLM stronger than Jev on this data (escalation and plan ordering only if its mean accuracy is at least 2 points higher)?
B. Do the judgements change when the 21 questions are packed into one Jev request?
C. How well does a 50-row Jev sample estimate the selectivity of the 21 predicates?

    python experiments/database/p4_mams.py
"""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.stats import rankdata

from p4_collect import PREDICATES, sentences
from paths import OUT as RESULTS

OUT = RESULTS / "p4"
DRAWS, N_SAMPLE, SEED = 1000, 50, 20261007


def read(source: str) -> dict[str, dict[str, float]]:
    return {r["sid"]: r["p"] for r in map(json.loads, (OUT / f"mams.{source}.jsonl").read_text().splitlines())}


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos = y == 1
    if pos.all() or not pos.any():
        return float("nan")
    return (rankdata(score)[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def f1(pred: np.ndarray, y: np.ndarray) -> float:
    tp = (pred & (y == 1)).sum()
    return 2 * tp / max(pred.sum() + (y == 1).sum(), 1)


def mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    only_a, only_b = int((a & ~b).sum()), int((b & ~a).sum())
    n, k = only_a + only_b, min(only_a, only_b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def qerr(est: np.ndarray | float, truth: float, floor: float) -> np.ndarray:
    return np.maximum(np.maximum(est, floor) / max(truth, floor), max(truth, floor) / np.maximum(est, floor))


def main() -> None:
    rows = sentences()
    sid = [r["sid"] for r in rows]
    names = list(PREDICATES)
    Y = np.array([[r["truth"][k] for k in names] for r in rows])
    single, packed, llm = read("jev"), read("jev.packed"), read("luna.low")
    P = np.array([[single[s][k] for k in names] for s in sid])
    PK = np.array([[packed[s][k] for k in names] for s in sid])
    sample = np.array([s in llm for s in sid])
    L = np.array([[llm[s][k] for k in names] for s in np.array(sid)[sample]])
    runs = {s: json.loads((OUT / f"mams.{s}.run.json").read_text()) for s in ("jev", "jev.packed", "luna.low")}
    out: dict = {}

    print(f"A. Is the LLM stronger than Jev: {sample.sum()} sentences × {len(names)} predicates, against human labels")
    print(f"{'predicate':20s}{'selectivity':>12s}{'Jev acc':>10s}{'LLM acc':>10s}{'diff':>7s}{'paired p':>9s}{'Jev F1':>8s}{'LLM F1':>8s}{'Jev AUC':>8s}")
    ys, ps = Y[sample], P[sample]
    table = []
    for j, k in enumerate(names):
        jev_ok, llm_ok = (ps[:, j] >= 0.5) == (ys[:, j] == 1), (L[:, j] >= 0.5) == (ys[:, j] == 1)
        row = {"predicate": k, "selectivity": Y[:, j].mean(), "jev_acc": jev_ok.mean(), "llm_acc": llm_ok.mean(), "p": mcnemar(jev_ok, llm_ok),
               "jev_f1": f1(ps[:, j] >= 0.5, ys[:, j]), "llm_f1": f1(L[:, j] >= 0.5, ys[:, j]), "jev_auc": auc(P[:, j], Y[:, j])}
        table.append(row)
        print(f"{k:20s}{row['selectivity']:>8.1%}{row['jev_acc']:>11.1%}{row['llm_acc']:>12.1%}{100 * (row['llm_acc'] - row['jev_acc']):>+8.1f}{row['p']:>9.3f}"
              f"{row['jev_f1']:>9.3f}{row['llm_f1']:>9.3f}{row['jev_auc']:>9.3f}")
    mean = lambda key: float(np.mean([r[key] for r in table]))
    gap = 100 * (mean("llm_acc") - mean("jev_acc"))
    for kind in ("mention", "negative", "positive"):
        part = [r for r in table if r["predicate"].startswith(kind)]
        print(f"  {kind:9s}mean: Jev accuracy {np.mean([r['jev_acc'] for r in part]):.1%} / F1 {np.mean([r['jev_f1'] for r in part]):.3f}, "
              f"LLM accuracy {np.mean([r['llm_acc'] for r in part]):.1%} / F1 {np.mean([r['llm_f1'] for r in part]):.3f}")
    print(f"mean over 21 predicates: Jev {mean('jev_acc'):.2%}, LLM {mean('llm_acc'):.2%}, LLM higher by {gap:+.2f} points; F1: Jev {mean('jev_f1'):.3f}, LLM {mean('llm_f1'):.3f}"
          f" → {'2 points reached: do escalation and plan ordering' if gap >= 2 else 'below 2 points: by the rule, no escalation; study Jev executing alone'}")
    out["A"] = {"table": table, "mean_gap_pp": gap, "escalate": bool(gap >= 2)}

    print(f"\nB. 21 questions packed into one request (all {len(rows)} sentences)")
    acc_single, acc_packed = ((P >= 0.5) == (Y == 1)).mean(axis=0), ((PK >= 0.5) == (Y == 1)).mean(axis=0)
    f1_single = np.array([f1(P[:, j] >= 0.5, Y[:, j]) for j in range(len(names))])
    f1_packed = np.array([f1(PK[:, j] >= 0.5, Y[:, j]) for j in range(len(names))])
    agree = ((P >= 0.5) == (PK >= 0.5)).mean(axis=0)
    print(f"  accuracy: one by one {acc_single.mean():.2%}, packed {acc_packed.mean():.2%} (per-predicate difference {100 * (acc_packed - acc_single).min():+.1f} to {100 * (acc_packed - acc_single).max():+.1f} points)")
    print(f"  F1: one by one {f1_single.mean():.3f}, packed {f1_packed.mean():.3f} (per-predicate difference {(f1_packed - f1_single).min():+.3f} to {(f1_packed - f1_single).max():+.3f})")
    print(f"  the two forms agree on {agree.mean():.2%} of judgements (lowest predicate {agree.min():.2%}), mean probability difference {np.abs(P - PK).mean():.3f}")
    tok_single, tok_packed = runs["jev"]["input_tokens"] / len(rows), runs["jev.packed"]["input_tokens"] / len(rows)
    print(f"  input tokens per sentence: one by one {tok_single:.0f} (21 requests), packed {tok_packed:.0f} (1 request), {tok_single / tok_packed:.1f}x fewer; "
          f"median request latency: one by one {runs['jev']['latency_p50']:.2f} s, packed {runs['jev.packed']['latency_p50']:.2f} s; "
          f"whole table: one by one {runs['jev']['seconds']:.0f} s, packed {runs['jev.packed']['seconds']:.0f} s")
    out["B"] = {"acc_single": acc_single.mean(), "acc_packed": acc_packed.mean(), "f1_single": f1_single.mean(), "f1_packed": f1_packed.mean(),
                "agreement": agree.mean(), "tokens_single": tok_single, "tokens_packed": tok_packed,
                "per_predicate": {k: {"acc_single": a, "acc_packed": b, "f1_single": c, "f1_packed": d} for k, a, b, c, d in zip(names, acc_single, acc_packed, f1_single, f1_packed)}}

    print(f"\nC. Selectivity estimated from {N_SAMPLE} sampled rows of the whole table, {DRAWS} draws per predicate. Error factor: median / 90th percentile / share within 2x")
    rng = np.random.default_rng(SEED)
    draws = np.array([rng.choice(len(rows), N_SAMPLE, replace=False) for _ in range(DRAWS)])
    floor = 0.5 / N_SAMPLE
    errs = {k: [] for k in ("default 1/3", "Jev sample, threshold (main)", "Jev sample, probability", "human-label sample", "Jev full, threshold", "vs Jev's own pass rate")}
    for j in range(len(names)):
        truth, hard = Y[:, j].mean(), (P[:, j] >= 0.5)
        errs["default 1/3"].append(np.full(DRAWS, qerr(1 / 3, truth, floor)))
        errs["Jev sample, threshold (main)"].append(qerr(hard[draws].mean(axis=1), truth, floor))
        errs["Jev sample, probability"].append(qerr(P[:, j][draws].mean(axis=1), truth, floor))
        errs["human-label sample"].append(qerr(Y[:, j][draws].mean(axis=1), truth, floor))
        errs["Jev full, threshold"].append(np.full(DRAWS, qerr(hard.mean(), truth, floor)))
        # When Jev itself executes the predicate, the optimizer needs to predict how many rows Jev will pass, not the human truth
        errs["vs Jev's own pass rate"].append(qerr(hard[draws].mean(axis=1), hard.mean(), floor))
    out["C"] = {}
    for k, e in errs.items():
        e = np.concatenate(e)
        out["C"][k] = {"median": float(np.median(e)), "p90": float(np.quantile(e, 0.9)), "within2": float((e <= 2).mean())}
        print(f"  {k:22s}{np.median(e):>7.2f}{np.quantile(e, 0.9):>9.2f}{(e <= 2).mean():>9.1%}")
    low = [j for j in range(len(names)) if Y[:, j].mean() < 0.05]
    e = np.concatenate([errs["Jev sample, threshold (main)"][j] for j in low])
    g = np.concatenate([errs["human-label sample"][j] for j in low])
    print(f"  the {len(low)} predicates with true selectivity below 5%: Jev sample median {np.median(e):.2f}, within 2x {(e <= 2).mean():.1%}; human-label sample median {np.median(g):.2f}, within 2x {(g <= 2).mean():.1%}")
    print("  Jev's full-table pass rate vs human truth: " + "; ".join(f"{k} {(P[:, j] >= 0.5).mean():.3f}/{Y[:, j].mean():.3f}" for j, k in enumerate(names) if j % 3 == 0))
    (OUT / "mams.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
