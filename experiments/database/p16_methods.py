"""Phase 16 designs and estimators (experiments/database/protocol/p16.txt). Pure functions of arrays: no data files, no model calls.

Every design returns the labelled set (xi), each row's inclusion probability (pi) and the cross-fitted calibrated prediction (yhat); the
common estimator turns them into an estimate and a variance per output. Rows of fold j only ever use quantities computed from the other
fold, so the estimate is exactly unbiased under every design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.stats import t as student_t

SEED = 20261031
PILOT = 0.3  # share of the budget spent on the uniform first stage (QP, G, D, abae)
MIX = 0.2  # share mixed in uniformly (every design except U, labels-only, global calibration and abae)
KAPPA = 10.0  # shrinkage of a cell's mean squared residual towards its score bin
Z95 = 1.959963984540054
NBINS = 5


def uniforms(population: int, replication: int, n: int, stream: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Common random numbers: U1, U2 of population d and replication r (p16.txt "REPLICATIONS"); extra streams for abae on Q6 and S5."""
    key = [SEED, population, replication] + ([] if stream is None else [stream])
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(key)))
    return rng.random(n), rng.random(n)


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


def mixture(pi: np.ndarray, target: float) -> np.ndarray:
    return (1 - MIX) * pi + MIX * target / len(pi)


def calibrate(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares y = alpha + beta p per predicate; fewer than 10 rows or no spread in p gives c(p) = p."""
    C = p.shape[1]
    if len(w) < 10:
        return np.zeros(C), np.ones(C)
    W = w.sum()
    pm, ym = w @ p / W, w @ y / W
    dp = p - pm
    sxx, sxy = w @ (dp * dp), w @ (dp * (y - ym))
    ok = sxx > 1e-12
    beta = np.where(ok, sxy / np.where(ok, sxx, 1.0), 1.0)
    alpha = np.where(ok, ym - beta * pm, 0.0)
    return alpha, beta


@dataclass
class Context:
    """What the designs and estimators of one (population, query) need, precomputed once."""

    y: np.ndarray  # (N, C) labels the designs draw (human labels; a second model's answers in S4)
    p: np.ndarray  # (N, C) cheap model
    fold: np.ndarray  # (N,)
    bins: np.ndarray  # (N, C) score bins of p
    cells: np.ndarray  # (N,) group key of the cells
    A: sp.csr_matrix  # (N, K)
    h: np.ndarray  # (K,)
    AG: sp.csr_matrix
    hG: np.ndarray
    relative: bool = False

    def __post_init__(self):
        self.N, self.C = self.y.shape
        self.K = self.A.shape[1]
        self.AT = self.A.T.tocsr()
        self.A2T = self.A.multiply(self.A).T.tocsr()
        self.support = (self.A > 0).astype(np.float64).T.tocsr()
        self.groups = int(self.cells.max()) + 1
        self.cellid = self.bins * self.groups + self.cells[:, None]  # (N, C)
        self.folds = [self.fold == j for j in (0, 1)]
        if self.C == 1:
            self.wQ = np.sqrt(np.asarray(self.A.multiply(self.A) @ (1.0 / self.h**2)).ravel())
            self.wG = np.sqrt(np.asarray(self.AG.multiply(self.AG) @ (1.0 / self.hG**2)).ravel())

    def apply(self, Z: np.ndarray, squared: bool = False) -> np.ndarray:
        """sum_i a_ik Z_{i,c(k)} (or with a_ik^2)."""
        AT = self.A2T if squared else self.AT
        if self.C == 1:
            return AT @ Z[:, 0]
        return np.asarray(AT.multiply(Z.T).sum(axis=1)).ravel()

    def labelled(self, xi: np.ndarray) -> np.ndarray:
        return self.support @ xi.astype(np.float64)


def fitted(ctx: Context, coef: dict) -> np.ndarray:
    """Rows of fold j use coef[j], which was fitted on fold 1-j."""
    yhat = np.empty_like(ctx.p)
    for j, rows in enumerate(ctx.folds):
        alpha, beta = coef[j]
        yhat[rows] = alpha + beta * ctx.p[rows]
    return yhat


# ---------------------------------------------------------------- single-stage designs (no label used to set pi)

def static_pi(ctx: Context, design: str, budget: float) -> np.ndarray:
    if design in ("U", "labels", "global"):
        return np.full(ctx.N, budget / ctx.N)
    if ctx.C == 1:
        conf = np.sqrt(ctx.p[:, 0] * (1 - ctx.p[:, 0]))
        u = {"C": conf, "QU": ctx.wQ, "QC": conf * ctx.wQ}[design]
    else:
        h = np.maximum(0.01, 0.25 * ctx.p.mean(axis=0)) if ctx.relative else ctx.h
        v = ctx.p * (1 - ctx.p)
        u = {"C": np.sqrt(v.sum(axis=1)), "QU": np.full(ctx.N, np.sqrt((1 / h**2).sum())),
             "QC": np.sqrt((v / h**2).sum(axis=1))}[design]
    pi = np.empty(ctx.N)
    for rows in ctx.folds:
        target = budget * rows.sum() / ctx.N
        pi[rows] = mixture(water_fill(u[rows], target), target)
    return pi


def single_stage(ctx: Context, pi: np.ndarray, U1: np.ndarray):
    xi = U1 < pi
    coef = {}
    for j in (0, 1):
        train = ctx.folds[1 - j] & xi
        coef[j] = calibrate(ctx.p[train], ctx.y[train], 1.0 / pi[train])
    return xi, pi, fitted(ctx, coef)


# ---------------------------------------------------------------- two-stage designs (QP, G, D)

def shrunk_s2(ctx: Context, train: np.ndarray, alpha: np.ndarray, beta: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Per (row, predicate) of `rows`: the stage-1 mean squared residual of its cell, shrunk towards its score bin (p16.txt)."""
    r2 = (ctx.y[train] - (alpha + beta * ctx.p[train])) ** 2
    out = np.empty((rows.sum(), ctx.C))
    ncell = NBINS * ctx.groups
    for c in range(ctx.C):
        cell, b = ctx.cellid[train, c], ctx.bins[train, c]
        sums = np.bincount(cell, weights=r2[:, c], minlength=ncell)
        counts = np.bincount(cell, minlength=ncell)
        bsum, bcount = np.bincount(b, weights=r2[:, c], minlength=NBINS), np.bincount(b, minlength=NBINS)
        m_all = r2[:, c].mean() if len(r2) else 1.0
        m_bin = np.where(bcount > 0, bsum / np.maximum(bcount, 1), m_all)
        s2 = (sums + KAPPA * np.repeat(m_bin, ctx.groups)) / (counts + KAPPA)
        out[:, c] = s2[ctx.cellid[rows, c]]
    return out


def scores(ctx: Context, design: str, s2: np.ndarray, rows: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    if ctx.C == 1:
        if design == "QP":
            w = ctx.wQ[rows]
        elif design == "G":
            w = ctx.wG[rows]
        else:  # D: requirement from the distance of the calibrated cheap share to the 60% line
            t = ctx.AT @ (alpha[0] + beta[0] * ctx.p[:, 0])
            hD = np.maximum(0.05, np.abs(t - 0.60))
            w = np.sqrt(np.asarray(ctx.A.multiply(ctx.A) @ (1.0 / hD**2)).ravel())[rows]
        return np.sqrt(s2[:, 0]) * w
    if design == "QP":
        h = np.maximum(0.01, 0.25 * (alpha + beta * ctx.p).mean(axis=0)) if ctx.relative else ctx.h
    else:
        h = ctx.hG
    return np.sqrt((s2 / h**2).sum(axis=1))


def two_stage(ctx: Context, design: str, budget: float, U1: np.ndarray, U2: np.ndarray, pilot: float = PILOT, details: bool = False):
    pi1 = pilot * budget / ctx.N
    xi1 = U1 < pi1
    pi2 = np.empty(ctx.N)
    coef = {}
    for j in (0, 1):
        train = ctx.folds[1 - j] & xi1
        coef[j] = calibrate(ctx.p[train], ctx.y[train], np.ones(train.sum()))
        rows = ctx.folds[j]
        u = scores(ctx, design, shrunk_s2(ctx, train, *coef[j], rows), rows, *coef[j])
        Nj = rows.sum()
        target = (budget * Nj / ctx.N - Nj * pi1) / (1 - pi1)
        pi2[rows] = mixture(water_fill(u, target), target)
    xi = xi1 | (U2 < pi2)
    pi = 1 - (1 - pi1) * (1 - pi2)
    yhat = fitted(ctx, coef)
    return (xi, pi, yhat, xi1) if details else (xi, pi, yhat)


# ---------------------------------------------------------------- estimators

def interval(theta: np.ndarray, V: np.ndarray, df: np.ndarray, certain: np.ndarray, level: float = 0.975, clip: bool = True):
    """theta +- t sqrt(V), t the `level` quantile of Student's t with df degrees of freedom. V = 0 gives the point only if every row of
    the output's support is labelled (`certain`); V = 0 otherwise (nothing labelled where it varies), or V > 0 with df < 1, gives [0, 1]."""
    df = np.asarray(df, float)
    q = np.where(df >= 1, student_t.ppf(level, np.maximum(df, 1)), np.inf)
    with np.errstate(invalid="ignore"):
        half = np.where(V > 0, q * np.sqrt(V), np.where(certain, 0.0, np.inf))
    lo, hi = theta - half, theta + half
    if clip:
        lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
    return lo, hi


def common(ctx: Context, xi: np.ndarray, pi: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    res = ctx.y - yhat
    x = xi[:, None]
    theta = ctx.apply(yhat + np.where(x, res / pi[:, None], 0.0))
    V = ctx.apply(np.where(x, res**2 * ((1 - pi) / pi**2)[:, None], 0.0), squared=True)
    return theta, V


def labels_only(ctx: Context, xi: np.ndarray, pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hajek estimate per output from the labelled rows only; an output without labelled rows: .5 and an infinite variance."""
    x = xi[:, None].astype(float)
    w = x / pi[:, None]
    den = ctx.apply(np.broadcast_to(w, ctx.y.shape))
    num = ctx.apply(w * ctx.y)
    theta = np.where(den > 0, num / np.where(den > 0, den, 1), 0.5)
    q = x * ((1 - pi) / pi**2)[:, None]
    s0 = ctx.apply(np.broadcast_to(q, ctx.y.shape), squared=True)
    s1, s2 = ctx.apply(q * ctx.y, squared=True), ctx.apply(q * ctx.y**2, squared=True)
    V = np.where(den > 0, (s2 - 2 * theta * s1 + theta**2 * s0) / np.where(den > 0, den, 1) ** 2, np.inf)
    return theta, np.maximum(V, 0.0)


def global_calibration(ctx: Context, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One calibration for the whole table, applied to every output; variance from the coefficients only (sandwich)."""
    alpha, beta = calibrate(ctx.p[xi], ctx.y[xi], np.ones(xi.sum()))
    theta = ctx.apply(alpha + beta * ctx.p)
    ones = ctx.apply(np.ones_like(ctx.p))
    pa = ctx.apply(ctx.p)
    V = np.zeros(ctx.K)
    for c in range(ctx.C):
        X = np.column_stack([np.ones(xi.sum()), ctx.p[xi, c]])
        e = ctx.y[xi, c] - X @ np.array([alpha[c], beta[c]])
        try:
            bread = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            V[:] = np.inf
            continue
        cov = bread @ (X.T * e**2) @ X @ bread
        ks = np.arange(ctx.K) if ctx.C == 1 else np.array([c])
        g = np.column_stack([ones[ks], pa[ks]])
        V[ks] = np.einsum("ki,ij,kj->k", g, cov, g)
    return theta, V


# ---------------------------------------------------------------- ABae and ABae-GroupBy as published (Kang et al. 2021)

def _largest_remainder(values: np.ndarray, total: int, tiebreak: np.ndarray) -> np.ndarray:
    base = np.floor(values).astype(np.int64)
    short = int(total - base.sum())
    if short > 0:
        order = np.lexsort((tiebreak, -(values - base)))
        base[order[:short]] += 1
    return base


def _capped_proportional(weights: np.ndarray, room: np.ndarray, total: int, tiebreak: np.ndarray) -> np.ndarray:
    """total labels in proportion to weights, at most room per stratum (the excess of full strata spread again); integers."""
    room = np.asarray(room, np.int64)
    total = min(int(total), int(room.sum()))
    if total == room.sum():
        return room.copy()
    w = np.where(room > 0, np.asarray(weights, float) + 1e-12, 0.0)
    capped = np.zeros(len(room), bool)
    while True:
        free = (room > 0) & ~capped
        lam = (total - room[capped].sum()) / w[free].sum()
        now = (room > 0) & (lam * w >= room)
        if (now == capped).all():
            break
        capped = now
    alloc = np.where(capped, room, lam * w)
    return np.minimum(room, _largest_remainder(alloc, total, tiebreak))


def _binary(values: np.ndarray) -> bool:
    return bool(np.isin(values, (0.0, 1.0)).all())


def _variance(x: np.ndarray, binary: bool, pooled: float) -> float:
    n = len(x)
    if binary:  # Bernoulli plug-in with one success and one failure added, so that a stratum of identical labels is not given 0
        m = (x.sum() + 1) / (n + 2)
        return m * (1 - m)
    return float(x.var(ddof=1)) if n >= 2 else pooled


def abae(y: np.ndarray, p: np.ndarray, budget: int, U1: np.ndarray, U2: np.ndarray, groups: np.ndarray | None = None,
         members: np.ndarray | None = None, stat: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-stage stratified sampling by proxy-score quantile strata (within groups), stratified estimator pooling both stages.
    Returns per-group estimates and variances (one value without groups) and the mask of labelled rows."""
    N = len(y)
    members = np.ones(N, bool) if members is None else members
    x = y if stat is None else stat * y
    binary = _binary(x[members])
    rows = np.flatnonzero(members)
    gcode = np.zeros(N, np.int64) if groups is None else groups
    present = np.unique(gcode[rows])
    G = len(present)
    n1_total = int(round(PILOT * budget))
    per_group = n1_total / G
    K = 1 if groups is not None else min(5, max(1, int(n1_total // 100)))
    assert groups is None or min(5, max(1, int(per_group // 100))) == 1, "ABae-GroupBy with more than one stratum per group"
    # strata: groups x K quantile strata of p (ties by the row order of the sorted keys)
    stratum = np.full(N, -1, np.int64)
    if K == 1:
        remap = np.full(gcode.max() + 1, -1, np.int64)
        remap[present] = np.arange(G)
        stratum[rows] = remap[gcode[rows]]
    else:
        order = rows[np.argsort(p[rows], kind="stable")]
        stratum[order] = np.minimum(np.arange(len(order)) * K // len(order), K - 1)
    S = stratum.max() + 1
    sizes = np.bincount(stratum[rows], minlength=S)
    # rows of each stratum in increasing U1; tie-break of integer allocations by U2 of each stratum's first row
    order = rows[np.lexsort((U1[rows], stratum[rows]))]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    tiebreak = U2[order[starts]]
    n1 = _capped_proportional(np.ones(S), sizes, n1_total, tiebreak)  # stage 1: equal over strata, capped by size
    rank = np.empty(N, np.int64)
    rank[order] = np.arange(len(order)) - np.repeat(starts, sizes)
    take1 = members & (rank < n1[np.maximum(stratum, 0)])
    pooled = float(x[take1].var(ddof=1)) if take1.sum() >= 2 else 1.0
    sd = np.array([np.sqrt(_variance(x[take1 & (stratum == s)], binary, pooled)) for s in range(S)])
    n2_total = budget - int(n1.sum())
    room = sizes - n1
    if groups is None:  # Neyman over strata
        n2 = _capped_proportional(sizes * sd, room, n2_total, tiebreak)
    else:  # minimax over groups: equalise (1 - n/N) sd^2 / n
        var = sd**2
        lo, hi = 1e-12, 1e3
        for _ in range(100):
            mid = np.sqrt(lo * hi)
            n = np.clip(sizes * var / (sizes * mid + var), n1, sizes)
            lo, hi = (lo, mid) if n.sum() < budget else (mid, hi)
        n = np.clip(sizes * var / (sizes * hi + var), n1, sizes)
        n2 = _capped_proportional(np.maximum(n - n1, 0), room, n2_total, tiebreak)
    n_s = n1 + n2
    take = members & (rank < n_s[np.maximum(stratum, 0)])
    mean = np.zeros(S)
    var_s = np.zeros(S)
    for s in range(S):
        xs = x[take & (stratum == s)]
        if len(xs):
            mean[s] = xs.mean()
            var_s[s] = xs.var(ddof=1) if len(xs) >= 2 else (xs.mean() * (1 - xs.mean()) if binary else pooled)
    fpc = np.where(n_s > 0, (1 - n_s / np.maximum(sizes, 1)) * var_s / np.maximum(n_s, 1), np.inf)
    if groups is None:
        if stat is None:
            weights = sizes / sizes.sum()
            return np.array([weights @ mean]), np.array([weights**2 @ fpc]), take
        total = stat[members].sum()
        return np.array([sizes @ mean / total]), np.array([(sizes**2) @ fpc / total**2]), take
    theta = np.where(n_s > 0, mean, 0.5)
    full = np.full(gcode.max() + 1, 0.5)
    fullV = np.full(gcode.max() + 1, np.inf)
    full[present], fullV[present] = theta, fpc
    return full, fullV, take


# ---------------------------------------------------------------- S5: three tiers

def three_tier(ctx: Context, q2: np.ndarray, budget: float, m: float, U1: np.ndarray, U2: np.ndarray, U3: np.ndarray):
    """Humans label QP's set H; the second model answers A, which contains H, with P(i in A) = min(1, m pi_i)."""
    xi, pi, yhat, pilot = two_stage(ctx, "QP", budget, U1, U2, details=True)
    piA = np.minimum(1.0, m * pi)
    extra = np.where(pi < 1, (piA - pi) / np.maximum(1 - pi, 1e-12), 0.0)
    xiA = xi | (U3 < extra)
    qt = np.empty_like(ctx.p)
    for j, rows in enumerate(ctx.folds):
        train = ctx.folds[1 - j] & pilot
        for c in range(ctx.C):
            if train.sum() < 10:
                qt[rows, c] = yhat[rows, c]
                continue
            X = np.column_stack([np.ones(train.sum()), ctx.p[train, c], q2[train, c]])
            coef = np.linalg.lstsq(X, ctx.y[train, c], rcond=None)[0]
            qt[rows, c] = coef[0] + coef[1] * ctx.p[rows, c] + coef[2] * q2[rows, c]
    d, e = qt - yhat, ctx.y - qt
    xa, xh = xiA[:, None], xi[:, None]
    PA, PH = piA[:, None], pi[:, None]
    theta = ctx.apply(yhat + np.where(xa, d / PA, 0.0) + np.where(xh, e / PH, 0.0))
    V = ctx.apply(np.where(xa, d**2 * (1 - PA) / PA**2, 0.0) + np.where(xh, e**2 * (1 - PH) / PH**2, 0.0)
                  + np.where(xh, 2 * d * e * (1 / PA - 1) / PH, 0.0), squared=True)
    return theta, np.maximum(V, 0.0), xi, int(xiA.sum())


# ---------------------------------------------------------------- labels needed

def labels_needed(budgets, losses, target: float) -> float:
    """Budget at which the loss curve reaches target: log loss linear in log budget on the grid, slope -1 beyond it."""
    b = np.log(np.asarray(budgets, float))
    L = np.log(np.maximum(np.asarray(losses, float), 1e-300))
    t = np.log(max(target, 1e-300))
    if L[0] <= t:
        return float(np.exp(b[0] + L[0] - t))
    for i in range(len(b) - 1):
        if L[i] > t >= L[i + 1]:
            return float(np.exp(b[i] + (L[i] - t) / (L[i] - L[i + 1]) * (b[i + 1] - b[i])))
    return float(np.exp(b[-1] + L[-1] - t))
