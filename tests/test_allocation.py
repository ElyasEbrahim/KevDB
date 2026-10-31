"""kevdb.allocation on small synthetic DuckDB tables: budgets, groups that overlap, weights, exactness when every row is labelled,
unbiasedness over many draws, refusals, and that it reproduces the design phase 16 froze. Reads no data file, calls no model."""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pytest

from kevdb import allocation

ROOT = Path(__file__).resolve().parents[1]


def table(n=300, seed=0):
    """Reviews in groups of uneven size, a biased cheap model, a weight and a list of tags per row."""
    rng = np.random.default_rng(seed)
    group = np.sort(rng.choice(6, n, p=np.array([1, 2, 3, 4, 6, 9]) / 25))
    q = 1 / (1 + np.exp(-(rng.normal(0.4, 1.4, n) + rng.normal(0, 0.7, 6)[group])))
    label = (rng.random(n) < q).astype(int)
    p = np.clip(1 / (1 + np.exp(-(0.8 * np.log(q / (1 - q)) - 0.5 + rng.normal(0, 0.6, n)))), 0.01, 0.99)
    weight = np.round(np.exp(rng.normal(0, 1, n)), 3)
    tags = [["a", "b"][: 1 + i % 2] if g < 3 else ["c"] for i, g in enumerate(group)]
    con = duckdb.connect()
    con.register("incoming", pa.table({"id": [f"r{i}" for i in range(n)], "grp": [f"g{g}" for g in group], "p": p,
                                       "label": label, "weight": weight, "tags": tags}))
    con.execute("CREATE TABLE reviews AS SELECT * FROM incoming")
    return con, {f"r{i}": int(v) for i, v in enumerate(label)}


def truth(con, by=None, weight="1.0"):
    if by is None:
        return {"all": con.execute(f"SELECT sum({weight} * label) / sum({weight}) FROM reviews").fetchall()[0][0]}
    if by == "tags":
        rows = con.execute("SELECT t, sum(label) / count(*) FROM (SELECT unnest(tags) AS t, label FROM reviews) GROUP BY t").fetchall()
    else:
        rows = con.execute(f"SELECT {by}, sum({weight} * label) / sum({weight}) FROM reviews GROUP BY {by}").fetchall()
    return dict(rows)


def test_the_budget_is_met_and_rows_lists_the_chosen_rows():
    con, labels = table()
    plan = allocation.plan(con, "reviews", key="id", p="p", group="grp", budget=60)
    assert plan.pi.sum() == pytest.approx(60) and (plan.pi > 0).all() and (plan.pi <= 1).all()
    rows = plan.rows()
    assert rows.num_rows == plan.chosen.sum() and set(rows.column("key").to_pylist()) <= set(labels)
    assert np.allclose(rows.column("inclusion_probability").to_numpy(), plan.pi[plan.chosen])


@pytest.mark.parametrize("group, weight", [(None, None), ("grp", None), (None, "weight"), ("grp", "weight"), ("tags", None)])
def test_labelling_every_row_gives_the_exact_answer(group, weight):
    con, labels = table()
    plan = allocation.plan(con, "reviews", key="id", p="p", group=group, weight=weight, budget=300)
    assert plan.chosen.all()
    out = plan.estimate(labels)
    expected = truth(con, group, weight or "1.0")
    got = dict(zip(out.column("output").to_pylist(), out.column("estimate").to_pylist()))
    assert got.keys() == {str(k) for k in expected}
    for k, v in expected.items():
        assert got[str(k)] == pytest.approx(v)
    assert np.allclose(out.column("low").to_numpy(), out.column("high").to_numpy())


@pytest.mark.parametrize("group, weight", [("grp", None), (None, "weight"), ("tags", None)])
def test_estimates_are_unbiased_over_many_draws(group, weight):
    con, labels = table()
    expected = truth(con, group, weight or "1.0")
    names = [str(k) for k in sorted(expected, key=str)]
    draws = []
    for seed in range(2000):
        out = allocation.plan(con, "reviews", key="id", p="p", group=group, weight=weight, budget=45, seed=seed).estimate(labels)
        draws.append(dict(zip(out.column("output").to_pylist(), out.column("estimate").to_pylist())))
    for name in names:
        values = np.array([d[name] for d in draws])
        assert abs(values.mean() - expected[name]) <= 4.5 * values.std(ddof=1) / np.sqrt(len(values)) + 1e-9


def test_groups_with_more_influence_get_more_labels():
    con, _ = table()
    plan = allocation.plan(con, "reviews", key="id", p="p", group="grp", budget=60)
    grp = np.array(con.execute("SELECT grp FROM reviews").fetchall())[:, 0]
    per_row = {g: plan.pi[grp == g].mean() for g in np.unique(grp)}
    assert per_row["g0"] > 3 * per_row["g5"]  # the smallest group is labelled far more densely than the largest


def test_refusals():
    con, labels = table()
    plan = allocation.plan(con, "reviews", key="id", p="p", budget=40)
    partial = dict(list({k: labels[k] for k in plan.rows().column("key").to_pylist()}.items())[1:])
    with pytest.raises(KeyError):
        plan.estimate(partial)
    con.execute("CREATE TABLE broken AS SELECT * FROM reviews")
    con.execute("UPDATE broken SET p = NULL WHERE id = 'r3'")
    with pytest.raises(ValueError, match="NULL"):
        allocation.plan(con, "broken", key="id", p="p", budget=40)
    with pytest.raises(ValueError, match="unique"):
        allocation.plan(con, "(SELECT * FROM reviews UNION ALL SELECT * FROM reviews)", key="id", p="p", budget=40)


def test_it_reproduces_the_design_phase_16_froze():
    sys.path.insert(0, str(ROOT / "experiments" / "database"))
    import p16_data as D
    import p16_methods as M

    con, labels = table()
    rows = con.execute("SELECT id, grp, p, label FROM reviews").fetchall()
    keys = [r[0] for r in rows]
    codes = np.unique([r[1] for r in rows], return_inverse=True)[1]
    y, p = np.array([[r[3]] for r in rows], float), np.array([[r[2]] for r in rows], float)
    fold = allocation.folds(keys, "p16|test|")
    assert (fold == D.folds("test", keys)).all()
    A = D._membership(codes)
    h = np.full(A.shape[1], 0.10)
    ctx = M.Context(y, p, fold, np.searchsorted(D.EDGES, p, side="right"), codes, A, h, A, h)
    pi = M.static_pi(ctx, "QC", 60)
    for r in range(20):
        U1, _ = M.uniforms(9, r, len(keys))
        xi, pi_, yhat = M.single_stage(ctx, pi, U1)
        theta, V = M.common(ctx, xi, pi_, yhat)
        lo, hi = M.interval(theta, V, ctx.labelled(xi & (pi_ < 1)) - 1, ctx.labelled(~xi) == 0)
        plan = allocation.plan(con, "reviews", key="id", p="p", group="grp", budget=60, h=0.10, uniforms=U1, salt="p16|test|")
        assert np.allclose(plan.pi, pi_) and (plan.chosen == xi).all()
        out = plan.estimate(labels)
        assert np.allclose(out.column("estimate").to_numpy(), theta)
        assert np.allclose(out.column("low").to_numpy(), lo) and np.allclose(out.column("high").to_numpy(), hi)
