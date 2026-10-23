"""Phase 14 evaluation: serve each padded seed-1 checkpoint and ask it what experiments/database/protocol/p14.txt asks. A thin layer over
phase 10's p10_eval.py, as in phases 11-13 (phase 13's p13_eval.py with seed 1 and Kev-0.8B only): serving (sandboxed kev.serve on port
8015, identity checks), caching, request rows, re-ask passes,
checkpoint checks and inventory are phase 10's, pointed at phase 14's folders and runs by configure(model).

    python experiments/database/p14_eval.py arms kev-0.8b                          the model's runs in the order p14_run.sh takes them
    python experiments/database/p14_eval.py arm kev-0.8b loo-walmart-amazon.seed1  serve open_models/p14/kev-0.8b/loo-walmart-amazon.seed1, ask its sets
    python experiments/database/p14_eval.py trained kev-0.8b loo-walmart-amazon.seed1   exit 0 if the checkpoint counts
    python experiments/database/p14_eval.py verify                                 phase 10's inputs and training files against its frozen checksums
    python experiments/database/p14_eval.py check kev-0.8b                         inventory: counts, attempts, deviations; no answer is looked at

Request sets of a run <phase 10 arm>.seed<N>: those of the phase 10 arm without its end-to-end and order readings: test.<held-out dataset>,
the pools of the three datasets it trains on, and the guards g1-g3. Phase 10's unpadded seed-1 answers are in out/database/p10, seeds 2
and 3 in out/database/p13.
"""

from __future__ import annotations

import os
import sys

REHEARSAL = os.environ.get("P14_REHEARSAL") == "1"
if REHEARSAL:
    os.environ["P10_REHEARSAL"] = "1"

import p10_eval as E  # noqa: E402
from paths import OUT as RESULTS, ROOT  # noqa: E402

RUNS = ROOT / "open_models" / ("p14_rehearsal" if REHEARSAL else "p14")
OUT = RESULTS / ("p14_rehearsal" if REHEARSAL else "p14")
P13_OUT = RESULTS / ("p13_rehearsal" if REHEARSAL else "p13")  # seeds 2 and 3
P10_OUT = RESULTS / ("p10_rehearsal" if REHEARSAL else "p10")
NAMESPACE = "p14.rehearsal:" if REHEARSAL else "p14:"
SEEDS = (1,)
# Phase 13's Kev-0.8B arms, whose seeds 2 and 3 passed the movie-review guard where phase 10's seed 1 did not
CLOSE = {"kev-0.8b": ("loo-amazon-google", "loo-walmart-amazon", "loo-dblp-acm")}
MODELS = tuple(CLOSE)


def arms(model: str) -> tuple[str, ...]:
    return tuple(f"{arm}.seed{seed}" for arm in CLOSE[model] for seed in SEEDS)


def phase10_arm(arm: str) -> str:
    return arm.rsplit(".seed", 1)[0]


def seed(arm: str) -> int:
    return int(arm.rsplit(".seed", 1)[1])


def sets(arm: str) -> list[str]:
    return [s for s in _p10_sets(phase10_arm(arm)) if not s.startswith(("e2e.", "order."))]


def training_records(arm: str) -> int:
    with (P10_OUT / "train" / f"{phase10_arm(arm)}.jsonl").open() as f:
        return sum(1 for _ in f) + E.REPLAY


def cache_id(model: str, arm: str) -> str:
    return _p10_cache_id(model, arm).replace("p10/", "p14/", 1)


_p10_sets = E.sets
_p10_cache_id = E.cache_id


def configure(model: str) -> None:
    """Point phase 10's evaluation code at phase 14 and this model's runs. Called by this script's and p14_report.py's entry points only,
    never on import, so that phase 10's to 13's code and tests are unaffected wherever they are loaded together."""
    E.RUNS = RUNS
    E.D.OUT = OUT
    E.NAMESPACE = NAMESPACE
    E.ARMS = arms(model)
    E.TRAINED = arms(model)
    E.sets = sets
    E.cache_id = cache_id
    E.training_records = training_records


def verify() -> int:
    """Phase 10's inputs and the training files of the arms run here, against phase 10's frozen checksums."""
    bad = 0
    for rel, digest in E.INPUTS.items():
        got = E.sha256(ROOT / rel)[0] if (ROOT / rel).exists() else "missing"
        if got != digest:
            print(f"input changed: {rel} ({got[:16]})")
            bad += 1
    frozen = E.BUILT_REHEARSAL if REHEARSAL else E.BUILT
    used = sorted({a for m in MODELS for a in CLOSE[m]})
    for arm in used:
        rel = f"train/{arm}.jsonl"
        got = E.sha256(P10_OUT / rel) if (P10_OUT / rel).exists() else ("missing", 0)
        if got != frozen[rel]:
            print(f"phase 10's training file differs from the frozen one: {rel} ({got[0][:16]}, {got[1]} lines)")
            bad += 1
    total = len(E.INPUTS) + len(used)
    print(f"verify: {total - bad} of {total} files as frozen")
    return bad


def main(argv: list[str]) -> int:
    command, rest = (argv[:1] or [""])[0], argv[1:]
    if rest[:1] and rest[0] in MODELS:
        configure(rest[0])
    if command == "arms" and len(rest) == 1 and rest[0] in MODELS:
        print(" ".join(arms(rest[0])))
        return 0
    if command == "arm" and len(rest) == 2 and rest[0] in MODELS and rest[1] in arms(rest[0]):
        return E.evaluate(*rest)
    if command == "trained" and len(rest) == 2 and rest[0] in MODELS and rest[1] in arms(rest[0]):
        reason = E.not_trained(*rest)
        print(reason or "complete")
        return 1 if reason else 0
    if command == "verify" and not rest:
        return 1 if verify() else 0
    if command == "check" and len(rest) == 1 and rest[0] in MODELS:
        return 1 if E.check(rest[0]) else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    sys.exit(main(sys.argv[1:]))
