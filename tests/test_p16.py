"""Easy mistakes in phase 16: that every design's estimate is unbiased (cross-fitting, two stages, three tiers), that budgets are met, that
a fold's second stage never sees its own labels, the published ABae, the interval rules, the labels-needed interpolation and the cells of
the standard. Synthetic tables only: reads no data file, calls no model."""

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "database"))

import p16_data as D  # noqa: E402
import p16_methods as M  # noqa: E402
import p16_run as R  # noqa: E402


def table(n=400, groups=8, seed=0):
    """Rows in groups of uneven size; labels with a group effect; a biased, noisy cheap model; fixed folds."""
    rng = np.random.default_rng(seed)
    g = np.sort(rng.choice(groups, n, p=np.linspace(1, 3, groups) / np.linspace(1, 3, groups).sum()))
    effect = rng.normal(0, 0.8, groups)[g]
    q = 1 / (1 + np.exp(-(rng.normal(0.5, 1.5, n) + effect)))
    y = (rng.random(n) < q).astype(float)[:, None]
    p = np.clip(1 / (1 + np.exp(-(np.log(q / (1 - q)) * 0.8 - 0.6 + rng.normal(0, 0.7, n)))), 0.01, 0.99)[:, None]
    fold = D.folds("test", [str(i) for i in range(n)])
    bins = np.searchsorted(D.EDGES, p, side="right")
    return y, p, fold, bins, g


def context(kind="groups", seed=0):
    y, p, fold, bins, g = table(seed=seed)
    n = len(g)
    if kind == "groups":
        A = D._membership(g)
        h = np.full(A.shape[1], 0.1)
    else:  # a weighted scalar with a heavy-tailed fan-out
        k = 1 + np.floor(np.exp(1.5 * np.random.default_rng(seed + 1).standard_normal(n)))
        A = sp.csr_matrix((k / k.sum())[:, None])
        h = np.array([0.01])
    AG = D._membership(g) if kind == "groups" else sp.csr_matrix(np.full((n, 1), 1 / n))
    hG = np.full(AG.shape[1], 0.1 if kind == "groups" else 0.01)
    return M.Context(y, p, fold, bins, g, A, h, AG, hG)


def draws(ctx, design, budget, reps, seed=5):
    pi_static = M.static_pi(ctx, design, budget) if design in ("U", "C", "QU", "QC") else None
    est = []
    for r in range(reps):
        U1, U2 = M.uniforms(seed, r, ctx.N)
        if pi_static is not None:
            xi, pi, yhat = M.single_stage(ctx, pi_static, U1)
        else:
            xi, pi, yhat = M.two_stage(ctx, design, budget, U1, U2)
        est.append(M.common(ctx, xi, pi, yhat))
    return np.array([e[0] for e in est]), np.array([e[1] for e in est])


def test_water_fill_meets_the_target_and_caps_at_one():
    u = np.random.default_rng(1).lognormal(0, 2, 500)
    for target in (5, 50, 250, 499):
        pi = M.water_fill(u, target)
        assert pi.sum() == pytest.approx(target) and pi.max() <= 1 + 1e-12
        free = pi < 1
        assert np.allclose(pi[free] / u[free], (pi[free] / u[free])[0])
    assert M.water_fill(np.zeros(4), 2).sum() == 0 and M.water_fill(np.ones(4), 9).tolist() == [1, 1, 1, 1]
    assert M.mixture(M.water_fill(u, 50), 50).sum() == pytest.approx(50)


def test_integer_allocations_respect_room_and_total():
    room = np.array([3, 50, 0, 7, 100])
    out = M._capped_proportional(np.array([10.0, 1, 5, 1, 1]), room, 60, np.arange(5) / 5)
    assert out.sum() == 60 and (out <= room).all() and out[2] == 0 and out[0] == 3
    assert M._capped_proportional(np.ones(5), room, 999, np.arange(5) / 5).tolist() == room.tolist()


@pytest.mark.parametrize("kind", ["groups", "scalar"])
@pytest.mark.parametrize("design", ["U", "C", "QU", "QC", "QP", "G"])
def test_every_design_is_unbiased(kind, design):
    ctx = context(kind)
    theta = ctx.apply(ctx.y)
    est, _ = draws(ctx, design, 80, 3000)
    se = est.std(axis=0, ddof=1) / np.sqrt(len(est))
    assert (np.abs(est.mean(axis=0) - theta) <= 4.5 * se + 1e-9).all()


def test_the_decision_aware_design_is_unbiased():
    ctx = context("groups")
    est, _ = draws(ctx, "D", 80, 3000)
    se = est.std(axis=0, ddof=1) / np.sqrt(len(est))
    assert (np.abs(est.mean(axis=0) - ctx.apply(ctx.y)) <= 4.5 * se + 1e-9).all()


def test_the_variance_estimate_matches_the_spread():
    ctx = context("scalar")
    est, var = draws(ctx, "C", 120, 4000)
    assert var.mean() == pytest.approx(est.var(ddof=1), rel=0.15)


def test_expected_labels_equal_the_budget():
    ctx = context("groups")
    for design in ("U", "C", "QU", "QC"):
        assert M.static_pi(ctx, design, 70).sum() == pytest.approx(70)
    U1, U2 = M.uniforms(1, 0, ctx.N)
    for design in ("QP", "G", "D"):
        _, pi, _ = M.two_stage(ctx, design, 70, U1, U2)
        assert pi.sum() == pytest.approx(70)
        for rows in ctx.folds:  # each fold carries its share of the budget
            assert pi[rows].sum() == pytest.approx(70 * rows.sum() / ctx.N)


def test_a_folds_second_stage_never_sees_its_own_labels():
    ctx = context("groups")
    U1, U2 = M.uniforms(2, 3, ctx.N)
    _, pi, yhat = M.two_stage(ctx, "QP", 90, U1, U2)
    changed = ctx.y.copy()
    changed[ctx.folds[0]] = 1.0  # (flipping every label would keep the squared residuals, so it would prove nothing)
    other = M.Context(changed, ctx.p, ctx.fold, ctx.bins, ctx.cells, ctx.A, ctx.h, ctx.AG, ctx.hG)
    _, pi2, yhat2 = M.two_stage(other, "QP", 90, U1, U2)
    rows = ctx.folds[0]
    assert np.allclose(pi[rows], pi2[rows]) and np.allclose(yhat[rows], yhat2[rows])
    assert not np.allclose(pi[ctx.folds[1]], pi2[ctx.folds[1]])


def test_g_and_qp_coincide_when_the_outputs_are_the_groups():
    ctx = context("groups")
    U1, U2 = M.uniforms(3, 0, ctx.N)
    a, b = M.two_stage(ctx, "QP", 60, U1, U2), M.two_stage(ctx, "G", 60, U1, U2)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))


def test_three_tiers_are_unbiased():
    ctx = context("groups")
    rng = np.random.default_rng(9)
    q2 = np.where(rng.random(ctx.y.shape) < 0.85, ctx.y, 1 - ctx.y)
    theta = ctx.apply(ctx.y)
    est = []
    for r in range(3000):
        U1, U2 = M.uniforms(4, r, ctx.N)
        U3, _ = M.uniforms(4, r, ctx.N, stream=3)
        t, V, xi, calls = M.three_tier(ctx, q2, 60, 3.0, U1, U2, U3)
        assert calls >= xi.sum()
        est.append(t)
    est = np.array(est)
    se = est.std(axis=0, ddof=1) / np.sqrt(len(est))
    assert (np.abs(est.mean(axis=0) - theta) <= 4.5 * se + 1e-9).all()


def test_abae_is_exact_when_every_row_is_labelled_and_meets_its_budget():
    y, p, _, _, g = table()
    U1, U2 = M.uniforms(5, 0, len(g))
    theta, V, take = M.abae(y[:, 0], p[:, 0], len(g), U1, U2, g)
    truth = np.bincount(g, weights=y[:, 0]) / np.bincount(g)
    assert np.allclose(theta, truth) and np.allclose(V, 0) and take.all()
    theta, V, take = M.abae(y[:, 0], p[:, 0], 90, U1, U2, g)
    assert take.sum() == 90 and np.isfinite(V).all()
    scalar, var, take = M.abae(y[:, 0], p[:, 0], 350, U1, U2)
    assert take.sum() == 350 and 0 <= scalar[0] <= 1 and var[0] > 0


def test_labels_only_and_global_calibration_with_every_row_labelled():
    ctx = context("groups")
    every = np.ones(ctx.N, bool)
    theta, V = M.labels_only(ctx, every, np.ones(ctx.N))
    assert np.allclose(theta, ctx.apply(ctx.y)) and np.allclose(V, 0)
    theta, V = M.global_calibration(ctx, every)
    assert theta.shape == (ctx.K,) and (V > 0).all()


def test_interval_rules():
    theta = np.full(4, 0.4)
    lo, hi = M.interval(theta, np.array([0.0, 0.01, 0.01, 0.0]), np.array([5, 0, 9, -1]), np.array([True, False, False, False]))
    assert (lo[0], hi[0]) == (0.4, 0.4) and (lo[1], hi[1]) == (0.0, 1.0)
    assert hi[2] - 0.4 == pytest.approx(2.2621571627 * 0.1)
    assert (lo[3], hi[3]) == (0.0, 1.0)  # no variance only because nothing was labelled: not a point


def test_an_output_without_labels_gets_no_point_interval():
    ctx = context("groups")
    pi = M.static_pi(ctx, "U", 8)  # so few labels that some groups get none
    hits = 0
    for r in range(50):
        xi, pi_, yhat = M.single_stage(ctx, pi, M.uniforms(6, r, ctx.N)[0])
        theta, V = M.common(ctx, xi, pi_, yhat)
        empty = ctx.labelled(xi) == 0
        lo, hi = M.interval(theta, V, ctx.labelled(xi & (pi_ < 1)) - 1, ctx.labelled(~xi) == 0)
        hits += empty.sum()
        assert (lo[empty] == 0).all() and (hi[empty] == 1).all()
    assert hits > 0


def test_labels_needed_follows_the_curve_and_extrapolates_with_slope_minus_one():
    budgets = [100, 200, 400]
    losses = [8.0, 4.0, 2.0]
    assert M.labels_needed(budgets, losses, 4.0) == pytest.approx(200)
    assert M.labels_needed(budgets, losses, 2.828427) == pytest.approx(282.84, rel=1e-4)
    assert M.labels_needed(budgets, losses, 1.0) == pytest.approx(800)
    assert M.labels_needed(budgets, losses, 16.0) == pytest.approx(50)


def test_fanout_and_folds_follow_the_standard():
    pop = D.Population("D1", np.array(["b", "a"]), np.zeros((2, 1)), np.full((2, 1), 0.5),
                       {"z": np.array([0.3, -1.2]), "top": np.array([1, 0])})
    assert D.fanout(pop, 2.0, np.log(5)).tolist() == [1 + np.floor(np.exp(0.6 + np.log(5))), 1 + np.floor(np.exp(-2.4))]
    assert D.folds("D1", ["a", "b"]).tolist() == D.folds("D1", ["b", "a"]).tolist()[::-1]


def test_the_cells_are_those_of_the_standard():
    cells = R.cells()
    names = [c.name for c in cells]
    assert len(names) == len(set(names)) == 899
    primary = [c for c in cells if c.part == "primary"]
    for q in R.D1_QUERIES:
        budgets = D.GRID[q][0]
        assert {(c.design, c.budget) for c in primary if c.query == q and c.pop == "D1"} >= {(d, b) for d in R.ALL for b in budgets}
    assert {c.budget for c in primary if c.design == "D"} == set(D.GRID["Q2"][0])
    assert {(c.design, c.budget) for c in primary if c.query == "Q6"} == {(d, b) for d in R.ALL for b in D.GRID["Q6"][0]}
    assert R.budgets_of("Q2", "D1-5k") == ((162, 325, 650, 1300), 650)


def test_the_run_script_switches_to_its_own_rehearsal():
    script = (ROOT / "experiments" / "database" / "p16_run.sh").read_text()
    assert "P16_REHEARSAL=1" in script and "p16_rehearsal" in script and "p15" not in script.lower()
    source = (ROOT / "experiments" / "database" / "p16_data.py").read_text()
    assert 'os.environ.get("P16_REHEARSAL") == "1"' in source and '"p16_rehearsal" if REHEARSAL else "p16"' in source
