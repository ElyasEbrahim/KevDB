"""Phase 16 run: every cell of experiments/database/protocol/p16.txt (population x cheap model x query x design x budget), 2,000
replications each. Replays cached answers only and calls no model; free. One JSON summary per cell; finished cells are skipped, so an
interrupted run resumes.

    python experiments/database/p16_run.py [--workers 14] [--reps 2000] [--parts primary,S1,...] [--only GLOB] [--list]
    P16_REHEARSAL=1 python experiments/database/p16_run.py      the same on simulated labels, into out/database/p16_rehearsal
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass

import numpy as np


import p16_data as D
import p16_methods as M

REPS = 2000
ALL = ("U", "C", "QU", "QC", "QP", "G", "abae", "labels", "global")
CORE = ("U", "C", "QU", "QC", "QP", "G")
D1_QUERIES = ("Q1", "Q2", "Q3a", "Q3b", "Q4a", "Q4b-1", "Q4b-2", "Q4b-2t")
SUBSET = {"D1": None, "D1-5k": "5k", "D1-1.85k": "1.85k", "D2": None, "D2-1k": "1k", "D2-546": "546"}
STATIC = ("U", "C", "QU", "QC", "labels", "global")


@dataclass(frozen=True)
class Cell:
    part: str  # primary, S1 ... S5
    pop: str
    cheap: str
    query: str
    design: str  # U C QU QC QP G D abae labels global; T3 (S5)
    budget: int
    pilot: float = M.PILOT
    second: str = ""  # S4: the second model whose answers replace the labels; S5: the tier-2 model
    m: float = 0.0  # S5: second-model calls per human label

    @property
    def name(self) -> str:
        extra = (f".pilot{self.pilot:g}" if self.pilot != M.PILOT else "") + (f".{self.second}" if self.second else "") + \
                (f".m{self.m:g}" if self.m else "")
        return f"{self.part}/{self.pop}.{self.cheap}/{self.query}/{self.design}{extra}.B{self.budget}"


def budgets_of(query: str, pop: str) -> tuple[tuple, int]:
    budgets, primary = D.GRID[query]
    family = pop.split("-")[0]
    if pop == family:
        return budgets, primary
    n = SIZES[pop]
    scale = lambda bs: tuple(max(20, round(b * n / D.FULL[family])) for b in bs)  # noqa: E731
    return scale(budgets), scale((primary,))[0]


SIZES = {"D1": 15388, "D1-5k": 5001, "D1-1.85k": 1851, "D2": 3949, "D2-1k": 1000, "D2-546": 546}


def cells() -> list[Cell]:
    out = []
    add = lambda part, pop, cheap, q, designs, bs, **kw: out.extend(  # noqa: E731
        Cell(part, pop, cheap, q, d, b, **kw) for d in designs for b in bs)
    for q in D1_QUERIES:  # primary
        add("primary", "D1", "jev", q, ALL, budgets_of(q, "D1")[0])
    add("primary", "D1", "jev", "Q2", ("D",), budgets_of("Q2", "D1")[0])
    add("primary", "D2", "jev", "Q6", ALL, budgets_of("Q6", "D2")[0])
    for cheap in ("jev", "kev-4b"):  # S1
        for q in ("Q1", "Q2", "Q3b", "Q4a", "Q4b-2"):
            add("S1", "D1-5k", cheap, q, CORE, budgets_of(q, "D1-5k")[0])
        add("S1", "D2-1k", cheap, "Q6", CORE, budgets_of("Q6", "D2-1k")[0])
    for share in (0.15, 0.5):  # S2
        for q in D1_QUERIES:
            add("S2", "D1", "jev", q, ("QP", "G"), (budgets_of(q, "D1")[1],), pilot=share)
        add("S2", "D2", "jev", "Q6", ("QP", "G"), (budgets_of("Q6", "D2")[1],), pilot=share)
    add("S3", "D1", "jev", "Q2b", ALL, budgets_of("Q2b", "D1")[0])
    for pop, second, queries in (("D1", "luna", ("Q1", "Q2", "Q3b")), ("D1-5k", "kev-9b", ("Q1", "Q2", "Q3b")),
                                 ("D1-5k", "deepseek.yesno", ("Q1", "Q2", "Q3b")), ("D1-1.85k", "opus", ("Q1", "Q2", "Q3b")),
                                 ("D2-1k", "luna", ("Q6",)), ("D2-1k", "deepseek", ("Q6",)), ("D2-546", "opus", ("Q6",))):
        for q in queries:  # S4: B* and the largest budget, scaled
            grid, primary = budgets_of(q, pop)
            add("S4", pop, "jev", q, ("U", "QP"), (primary, grid[-1]), second=second)
    for pop, second, queries in (("D1", "luna", ("Q1", "Q2")), ("D1-5k", "kev-9b", ("Q1", "Q2")), ("D1-5k", "deepseek", ("Q1", "Q2")),
                                 ("D2-1k", "luna", ("Q6",)), ("D2-1k", "deepseek", ("Q6",))):
        for q in queries:  # S5
            for m in (3.0, 10.0):
                add("S5", pop, "jev", q, ("T3",), budgets_of(q, pop)[0], second=second, m=m)
    return out


_CACHE: dict = {}


def population(name: str, cheap: str):
    if (name, cheap) not in _CACHE:
        loader = D.d1 if name.startswith("D1") else D.d2
        pop = loader(SUBSET[name], cheap)
        assert pop.N == SIZES[name], (name, pop.N)
        _CACHE[(name, cheap)] = (pop, D.queries(pop))
    return _CACHE[(name, cheap)]


class Accumulator:
    def __init__(self, q: D.Query, truths: dict, q5: bool):
        K = q.K
        self.h, self.truths, self.q5 = q.h, truths, q5
        self.reps = 0
        self.err = {t: np.zeros(K) for t in truths}
        self.err2 = {t: np.zeros(K) for t in truths}
        self.cover = {t: np.zeros(K) for t in truths}
        self.loss = {t: [] for t in truths}
        self.halfwidth, self.has2 = np.zeros(K), np.zeros(K)
        self.labels, self.second_calls = [], []
        self.wrong, self.decided, self.any_wrong = [], [], []
        self.family = 1 - 0.025 / K  # Q5: family-wise 95% over the movies

    def add(self, theta: np.ndarray, V: np.ndarray, df: np.ndarray, certain: np.ndarray, labels: int, has2: np.ndarray,
            second_calls: int = 0):
        lo, hi = M.interval(theta, V, df, certain)
        for t, truth in self.truths.items():
            e = theta - truth
            self.err[t] += e
            self.err2[t] += e**2
            self.cover[t] += (lo <= truth + 1e-12) & (truth <= hi + 1e-12)
            self.loss[t].append(float(np.mean((e / self.h) ** 2)))
        self.halfwidth += (hi - lo) / 2
        self.has2 += has2
        self.labels.append(int(labels))
        self.second_calls.append(int(second_calls))
        self.reps += 1
        if self.q5:
            truth = self.truths["human"]
            self.wrong.append(int((((truth >= 0.65) & (theta < 0.60)) | ((truth < 0.55) & (theta >= 0.60))).sum()))
            flo, fhi = M.interval(theta, V, df, certain, self.family, clip=False)
            fresh, rotten = flo > 0.60, fhi < 0.60
            self.decided.append(int((fresh | rotten).sum()))
            self.any_wrong.append(int(((fresh & (truth < 0.60)) | (rotten & (truth >= 0.60))).any()))

    def summary(self) -> dict:
        n = self.reps
        out = {"reps": n, "labels_mean": float(np.mean(self.labels)), "second_calls_mean": float(np.mean(self.second_calls)),
               "halfwidth": (self.halfwidth / n).tolist(), "has2": (self.has2 / n).tolist(), "by_truth": {}}
        for t, truth in self.truths.items():
            cover = self.cover[t] / n
            loss = np.array(self.loss[t])
            out["by_truth"][t] = {"truth": truth.tolist(), "bias": (self.err[t] / n).tolist(),
                                  "rmse": np.sqrt(self.err2[t] / n).tolist(), "coverage": cover.tolist(),
                                  "coverage_mean": float(cover.mean()), "coverage_p10": float(np.quantile(cover, 0.10)),
                                  "loss_mean": float(loss.mean()), "loss_se": float(loss.std(ddof=1) / np.sqrt(n))}
        if self.q5:
            out["q5"] = {"wrong_mean": float(np.mean(self.wrong)), "decided_mean": float(np.mean(self.decided)),
                         "any_wrong_decided": float(np.mean(self.any_wrong))}
        return out


def _abae(pop: D.Population, q: D.Query, cell: Cell, r: int, U1: np.ndarray, U2: np.ndarray):
    spec = q.abae
    if spec.get("per_predicate"):  # Q6: one ABae per predicate, B/21 sentences each, each from its own sample
        shares = M._largest_remainder(np.full(pop.C, cell.budget / pop.C), cell.budget, np.arange(pop.C) / pop.C)
        theta, V, take = np.zeros(pop.C), np.zeros(pop.C), np.zeros(pop.N, bool)
        for c in range(pop.C):
            u1, u2 = M.uniforms(pop.index, r, pop.N, stream=1000 + c)
            t, v, mask = M.abae(pop.y[:, c], pop.p[:, c], int(shares[c]), u1, u2)
            theta[c], V[c], take = t[0], v[0], take | mask
        return theta, V, take
    tg, vg, take = M.abae(pop.y[:, 0], pop.p[:, 0], cell.budget, U1, U2, spec.get("groups"), spec.get("members"), spec.get("stat"))
    combine = spec.get("combine")
    if spec.get("groups") is None or combine is None:
        return tg, vg, take
    vg = np.where(np.isfinite(vg), vg, 1e6)
    return combine.T @ tg, (combine**2).T @ vg, take


def run_cell(cell: Cell, reps: int) -> dict:
    started = time.perf_counter()
    pop, qs = population(cell.pop, cell.cheap)
    q = qs[cell.query]
    y = pop.y
    truths = {"human": q.truth(pop.y)}
    if cell.part == "S4":  # the second model's yes/no answers replace the human labels
        y = (pop.second[cell.second] >= 0.5).astype(float)
        truths["second"] = q.truth(y)
    ctx = M.Context(y, pop.p, pop.fold, pop.bins, q.cells, q.A, q.h, q.AG, q.hG, q.relative)
    acc = Accumulator(q, truths, q5=cell.query == "Q2")
    pi_static = M.static_pi(ctx, cell.design, cell.budget) if cell.design in STATIC else None
    for r in range(reps):
        U1, U2 = M.uniforms(pop.index, r, pop.N)
        calls = 0
        if cell.design in ("U", "C", "QU", "QC", "QP", "G", "D"):
            if cell.design in ("U", "C", "QU", "QC"):
                xi, pi, yhat = M.single_stage(ctx, pi_static, U1)
            else:
                xi, pi, yhat = M.two_stage(ctx, cell.design, cell.budget, U1, U2, cell.pilot)
            theta, V = M.common(ctx, xi, pi, yhat)
            df = ctx.labelled(xi & (pi < 1)) - 1  # rows drawn with certainty add no variance
        elif cell.design == "labels":
            xi = U1 < pi_static
            theta, V = M.labels_only(ctx, xi, pi_static)
            df = ctx.labelled(xi) - 1
        elif cell.design == "global":
            xi = U1 < pi_static
            theta, V = M.global_calibration(ctx, xi)
            df = np.full(q.K, xi.sum() - 2.0)
        elif cell.design == "abae":
            theta, V, xi = _abae(pop, q, cell, r, U1, U2)
            df = ctx.labelled(xi) - 1
        elif cell.design == "T3":
            U3, _ = M.uniforms(pop.index, r, pop.N, stream=3)
            theta, V, xi, calls = M.three_tier(ctx, pop.second[cell.second], cell.budget, cell.m, U1, U2, U3)
            df = ctx.labelled(xi) - 1
        else:
            raise ValueError(cell.design)
        acc.add(theta, V, df, ctx.labelled(~xi) == 0, int(xi.sum()), ctx.labelled(xi) >= 2, calls)
    out = {"cell": asdict(cell), "name": cell.name, "outputs": list(q.outputs), "h": q.h.tolist(), "N": pop.N}
    out.update(acc.summary())
    out["seconds"] = round(time.perf_counter() - started, 1)
    return out


def path_of(cell: Cell):
    return D.RESULTS / "cells" / (cell.name + ".json")


def work(cell: Cell, reps: int) -> tuple[str, float]:
    result = run_cell(cell, reps)
    target = path_of(cell)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    os.replace(tmp, target)
    return cell.name, result["seconds"]


def cost(cell: Cell) -> float:
    """Rough relative cost, to start the heaviest cells first."""
    return SIZES[cell.pop] * (3 if cell.design in ("QP", "G", "D", "T3", "abae") else 1) * (21 if cell.pop.startswith("D2") else 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--parts", default="primary,S1,S2,S3,S4,S5")
    ap.add_argument("--only", default="*")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    todo = [c for c in cells() if c.part in parts and fnmatch.fnmatch(c.name, args.only)]
    if args.list:
        for c in todo:
            print(c.name)
        print(len(todo), "cells")
        return
    pending = sorted((c for c in todo if not path_of(c).exists()), key=cost, reverse=True)
    stamp = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())  # noqa: E731
    print(f"{stamp()} {'rehearsal (simulated labels)' if D.REHEARSAL else 'formal run'}: {len(todo)} cells, {len(todo) - len(pending)} "
          f"already done, {len(pending)} to run, {args.reps} replications, {args.workers} workers -> {D.RESULTS}", flush=True)
    started, done = time.perf_counter(), 0
    if args.workers <= 1:
        for c in pending:
            name, seconds = work(c, args.reps)
            done += 1
            print(f"{stamp()} {done}/{len(pending)} {name} {seconds}s", flush=True)
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            futures = {pool.submit(work, c, args.reps): c for c in pending}
            for future in as_completed(futures):
                name, seconds = future.result()
                done += 1
                print(f"{stamp()} {done}/{len(pending)} {name} {seconds}s", flush=True)
    print(f"{stamp()} all {len(todo)} cells done in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
