"""Phase 4, part one: can Jev give the optimizer the selectivity of semantic predicates? Replays phase 2's judgements and calls no model.

Settings and stopping rule were written before the run, see experiments/database/protocol/p4.txt.

    python experiments/database/p4_selectivity.py
"""

from __future__ import annotations

import json

import numpy as np

from p3_aggregates import load
from paths import OUT as RESULTS

OUT = RESULTS / "p4"
N_SAMPLE, DRAWS, SEED = 50, 200, 20261007


def qerr(est: np.ndarray | float, truth: float, floor: float) -> np.ndarray:
    est, truth = np.maximum(est, floor), max(truth, floor)
    return np.maximum(est / truth, truth / est)


def main() -> None:
    t = load()
    rng = np.random.default_rng(SEED)
    ids, counts = np.unique(t["id"], return_counts=True)
    names = ["default 1/3", "Jev sample, threshold (main)", "Jev sample, probability", "LLM sample", "human-label sample", "Jev full, threshold"]
    out: dict = {}
    for predicate, flip in (("positive", False), ("negative", True)):
        y, p, llm = (1 - t["y"], 1 - t["p"], 1 - t["llm"]) if flip else (t["y"], t["p"], t["llm"])
        errs = {k: [] for k in names}
        truths = []
        for movie in ids[counts >= 30]:
            ix = np.flatnonzero(t["id"] == movie)
            n = min(N_SAMPLE, len(ix))
            truth, floor = y[ix].mean(), 0.5 / n
            truths.append(truth)
            draws = np.array([rng.choice(ix, n, replace=False) for _ in range(DRAWS)])
            errs["default 1/3"].append(np.full(DRAWS, qerr(1 / 3, truth, floor)))
            errs["Jev sample, threshold (main)"].append(qerr((p[draws] >= 0.5).mean(axis=1), truth, floor))
            errs["Jev sample, probability"].append(qerr(p[draws].mean(axis=1), truth, floor))
            errs["LLM sample"].append(qerr(llm[draws].mean(axis=1), truth, floor))
            errs["human-label sample"].append(qerr(y[draws].mean(axis=1), truth, floor))
            errs["Jev full, threshold"].append(np.full(DRAWS, qerr((p[ix] >= 0.5).mean(), truth, floor)))
        truths = np.array(truths)
        print(f"\npredicate \"{predicate}\": {len(truths)} subsets, true selectivity {truths.min():.2f}-{truths.max():.2f} (median {np.median(truths):.2f})")
        print(f"{'method':28s}{'median q-err':>13s}{'90th pct':>10s}{'within 2x':>10s}")
        out[predicate] = {}
        for k in names:
            e = np.concatenate(errs[k])
            out[predicate][k] = {"median": float(np.median(e)), "p90": float(np.quantile(e, 0.9)), "within2": float((e <= 2).mean())}
            print(f"{k:20s}{np.median(e):>12.2f}{np.quantile(e, 0.9):>13.2f}{(e <= 2).mean():>13.1%}")
        out[predicate]["_errors"] = {k: np.concatenate(errs[k]) for k in names}
    pooled = np.concatenate([out[pr]["_errors"]["Jev sample, threshold (main)"] for pr in ("positive", "negative")])
    baseline = np.concatenate([out[pr]["_errors"]["default 1/3"] for pr in ("positive", "negative")])
    labels = np.concatenate([out[pr]["_errors"]["human-label sample"] for pr in ("positive", "negative")])
    print(f"\nboth predicates pooled: median error factor of the Jev sample {np.median(pooled):.2f} (default 1/3: {np.median(baseline):.2f}, human-label sample: {np.median(labels):.2f})"
          f" → {'above 2: by the rule, no plan ordering' if np.median(pooled) > 2 else 'at most 2: plan ordering may proceed'}")
    for pr in ("positive", "negative"):
        del out[pr]["_errors"]
    out["pooled_primary_median"] = float(np.median(pooled))
    (OUT / "selectivity.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
