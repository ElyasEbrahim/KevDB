"""Reliable aggregates from a cheap model plus a few human labels, spent on the rows the query's answer depends on.

A cheap model has answered every row with a probability p; a person labels some rows. The labels correct the model's answers,
so each estimate is unbiased against the human labels whatever the model's bias, and comes with a 95% interval. Which rows
get a label decides how many labels are needed. A row is chosen with a probability proportional to its influence on the
query's outputs (its weight divided by its group's total, summed over the groups it counts in) times the model's
uncertainty sqrt(p(1 - p)). A fifth of the budget is spread uniformly, so that no row's chance is negligible, even where the
model is confidently wrong.

Phase 16 of experiments/database measured this design on 15,388 movie reviews and 3,949 restaurant sentences: for
per-group shares, uniform or confidence-only sampling needs 7 to 12 times its labels for the same error, and group-only
sampling up to 20 times when a join repeats rows heavily. Spending part of the budget on a human pilot to locate the errors
did not do better than the model's uncertainty, so there is no pilot stage. See experiments/database/README.md, "Phase 16".

    plan = allocation.plan(con, "reviews", key="id", p="p", group="movie", budget=2000)
    plan.rows()                       # the rows to label: key, p, inclusion_probability (a pyarrow table)
    plan.estimate({key: label, ...})  # per group: estimate, low, high, labelled, rows (a pyarrow table)

Outputs are shares: sum(weight x label) / sum(weight) over the rows of each group (weight 1 unless `weight` is given).
`group` may be a list column, so that a row counts in several groups, as when a join repeats it.
Only the rows the plan chose may be labelled and used; every one of them must be labelled.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

import duckdb
import numpy as np
import pyarrow as pa

MIX = 0.2  # share of the budget spread uniformly over the rows


def folds(keys: list, salt: str = "kevdb:") -> np.ndarray:
    """Each row's fold for the cross-fitted correction: parity of the first hex digit of sha256(salt + key)."""
    return np.array([int(hashlib.sha256(f"{salt}{k}".encode()).hexdigest()[0], 16) % 2 for k in keys], dtype=np.int64)


def water_fill(u: np.ndarray, target: float) -> np.ndarray:
    """pi_i = min(1, lam u_i) with sum pi = target; rows with u_i = 0 get 0."""
    u = np.asarray(u, float)
    pi = np.zeros_like(u)
    positive = u > 0
    if target <= 0 or not positive.any():
        return pi
    if target >= positive.sum():
        pi[positive] = 1.0
        return pi
    capped = np.zeros_like(positive)
    while True:
        free = positive & ~capped
        lam = (target - capped.sum()) / u[free].sum()
        now = positive & (lam * u >= 1.0)
        if (now == capped).all():
            break
        capped = now
    pi[capped] = 1.0
    pi[free] = lam * u[free]
    return pi


def _line(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted least squares y = alpha + beta p; fewer than 10 rows or no spread in p gives the identity."""
    if len(p) < 10:
        return 0.0, 1.0
    pm, ym = np.average(p, weights=w), np.average(y, weights=w)
    sxx = np.sum(w * (p - pm) ** 2)
    if sxx <= 1e-12:
        return 0.0, 1.0
    beta = np.sum(w * (p - pm) * (y - ym)) / sxx
    return float(ym - beta * pm), float(beta)


def interval(theta: np.ndarray, V: np.ndarray, df: np.ndarray, certain: np.ndarray, level: float = 0.95):
    """theta +- t sqrt(V), t the Student's t quantile with df degrees of freedom, clipped to [0, 1]. V = 0 gives the point only if
    every row of the output was labelled (`certain`); otherwise V = 0, or V > 0 with df < 1, gives [0, 1]."""
    from scipy.stats import t as student_t

    df = np.asarray(df, float)
    q = np.where(df >= 1, student_t.ppf(1 - (1 - level) / 2, np.maximum(df, 1)), np.inf)
    with np.errstate(invalid="ignore"):
        half = np.where(V > 0, q * np.sqrt(V), np.where(certain, 0.0, np.inf))
    return np.clip(theta - half, 0, 1), np.clip(theta + half, 0, 1)


@dataclass
class Plan:
    """Which rows to label, with each row's inclusion probability, and how to turn their labels into estimates."""

    keys: list
    p: np.ndarray  # the cheap model's probability per row
    outputs: list  # the groups (or ["all"])
    member_row: np.ndarray  # one entry per (row, output) membership
    member_output: np.ndarray
    influence: np.ndarray  # weight / total weight of the output, per membership
    pi: np.ndarray  # inclusion probability per row
    chosen: np.ndarray  # rows to label
    fold: np.ndarray
    _index: dict = field(default_factory=dict, repr=False)

    def _sum(self, z: np.ndarray, squared: bool = False) -> np.ndarray:
        a = self.influence**2 if squared else self.influence
        return np.bincount(self.member_output, weights=a * z[self.member_row], minlength=len(self.outputs))

    def _count(self, mask: np.ndarray) -> np.ndarray:
        return np.bincount(self.member_output, weights=mask[self.member_row].astype(float), minlength=len(self.outputs))

    def rows(self) -> pa.Table:
        """The rows to label, with the model's probability and the row's inclusion probability."""
        at = np.flatnonzero(self.chosen)
        return pa.table({"key": [self.keys[i] for i in at], "p": self.p[at], "inclusion_probability": self.pi[at]})

    def estimate(self, labels: Mapping[Any, Any], level: float = 0.95) -> pa.Table:
        """labels: key -> 0/1 for every chosen row (other keys are ignored). One row per output: estimate, low, high, labelled, rows."""
        chosen = np.flatnonzero(self.chosen)
        missing = [self.keys[i] for i in chosen if self.keys[i] not in labels]
        if missing:
            raise KeyError(f"{len(missing)} chosen rows have no label, e.g. {missing[:3]}; label every chosen row")
        y = np.zeros(len(self.keys))
        y[chosen] = [float(labels[self.keys[i]]) for i in chosen]
        x = self.chosen
        yhat = np.empty(len(self.keys))
        for j in (0, 1):  # cross-fitting: a row's calibration line uses only the labels of the other fold
            train = x & (self.fold == 1 - j)
            alpha, beta = _line(self.p[train], y[train], 1.0 / self.pi[train])
            rows = self.fold == j
            yhat[rows] = alpha + beta * self.p[rows]
        res = y - yhat
        theta = self._sum(yhat + np.where(x, res / self.pi, 0.0))
        V = self._sum(np.where(x, res**2 * (1 - self.pi) / self.pi**2, 0.0), squared=True)
        df = self._count(x & (self.pi < 1)) - 1
        low, high = interval(theta, V, df, self._count(~x) == 0, level)
        return pa.table({"output": [str(o) for o in self.outputs], "estimate": theta, "low": low, "high": high,
                         "labelled": self._count(x).astype(np.int64), "rows": self._count(np.ones_like(x)).astype(np.int64)})


def plan(con: duckdb.DuckDBPyConnection, table: str, *, key: str, p: str, budget: float, group: str | None = None,
         weight: str | None = None, h: float | Mapping[Any, float] | None = None, seed: int = 0,
         uniforms: np.ndarray | None = None, salt: str = "kevdb:") -> Plan:
    """Choose about `budget` rows of `table` to label.

    table   a table or view, or a parenthesised query
    key     a unique key per row; p the cheap model's probability (a column, or a SQL function call such as kev_noul(...))
    group   optional: the GROUP BY column, or a list column when a row counts in several groups (as through a join)
    weight  optional: a non-negative weight per row (a measure, or the number of output rows a join makes of the row)
    h       optional: the 95% half-width each output should reach, to favour some outputs; equal for all by default
    These arguments are SQL expressions evaluated on `table` and inserted as written: do not pass untrusted text.
    Rows whose weight is 0 or whose group is NULL count in no output but keep the uniform fifth of the budget; leave them
    out of `table` if they are not part of the query. `uniforms` replaces the random draw (one number in [0, 1) per row)."""
    columns = [f"{key} AS k", f"{p} AS p", f"{group} AS g" if group else "NULL AS g", f"{weight} AS w" if weight else "1.0 AS w"]
    fetched = con.execute(f"SELECT {', '.join(columns)} FROM {table}").fetchall()
    keys = [r[0] for r in fetched]
    if len(set(keys)) != len(keys):
        raise ValueError("key is not unique")
    if any(r[1] is None for r in fetched):
        raise ValueError("p is NULL for some rows: the cheap model must answer every row")
    probs = np.array([r[1] for r in fetched], float)
    w = np.array([r[3] for r in fetched], float)
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("weight must be a non-negative number on every row")
    member_row, names = [], []
    for i, (_, _, g, _) in enumerate(fetched):
        for member in (g if isinstance(g, list) else [g]) if group else ["all"]:
            if member is not None:
                member_row.append(i)
                names.append(member)
    outputs = sorted(set(names), key=lambda v: (str(type(v)), v))
    index = {o: k for k, o in enumerate(outputs)}
    member_row = np.array(member_row, np.int64)
    member_output = np.array([index[n] for n in names], np.int64)
    totals = np.bincount(member_output, weights=w[member_row], minlength=len(outputs))
    influence = np.divide(w[member_row], totals[member_output], out=np.zeros(len(member_row)), where=totals[member_output] > 0)
    if h is None:
        hk = np.ones(len(outputs))
    elif isinstance(h, Mapping):
        hk = np.array([float(h[o]) for o in outputs])
    else:
        hk = np.full(len(outputs), float(h))
    N = len(keys)
    reach = np.sqrt(np.bincount(member_row, weights=influence**2 / hk[member_output] ** 2, minlength=N))
    score = reach * np.sqrt(probs * (1 - probs))
    fold = folds(keys, salt)
    pi = np.empty(N)
    for j in (0, 1):  # each fold carries its share of the budget
        rows = fold == j
        target = budget * rows.sum() / N
        pi[rows] = (1 - MIX) * water_fill(score[rows], target) + MIX * target / rows.sum()
    u = np.random.default_rng(seed).random(N) if uniforms is None else np.asarray(uniforms, float)
    return Plan(keys, probs, outputs, member_row, member_output, influence, pi, u < pi, fold)
