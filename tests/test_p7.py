"""A few easy mistakes in the phase 7 analysis and plan. Calls no model and reads no collected results."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "database"))

import p6_report as R
import p7_cache
import p7_data
import p7_report


def test_scores_are_the_stage_6_metrics_and_missing_answers_count_as_wrong():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.2).astype(int)
    p = np.clip(np.where(y == 1, 0.7, 0.2) + rng.normal(0, 0.25, 500), 0, 1)
    p[:25] = np.nan
    got, want = p7_report.scores(p, y), R.four(p[:, None], y[:, None])
    assert np.allclose([got["acc"], got["auc"], got["brier"], got["ece_mass"]], want)
    assert got["missing_share"] == 0.05
    decide = np.where(np.isnan(p), False, p >= 0.5)
    tp = (decide & (y == 1)).sum()
    assert np.isclose(got["precision"], tp / decide.sum()) and np.isclose(got["recall"], tp / y.sum())


def test_match_rule_is_the_stage_6_rule():
    jev = {"acc": 0.90, "auc": 0.95, "brier": 0.08, "missing_share": 0.0}
    ok = {"acc": 0.881, "auc": 0.931, "brier": 0.099, "missing_share": 0.05}
    assert p7_report.matches(ok, jev)
    for key, value in (("acc", 0.879), ("auc", 0.929), ("brier", 0.101), ("missing_share", 0.051)):
        assert not p7_report.matches(ok | {key: value}, jev)


def test_answers_are_compared_by_decision_and_a_missing_answer_counts_as_different():
    noul = lambda p: {"q": {"type": "noul", "noul": p}}
    choice = lambda name, a: {"q": {"type": "choice", "choice": name, "probabilities": {"A": a, "B": 1 - a}}}
    score = lambda probs: {"q": {"type": "score", "score": 0.0, "probabilities": dict(enumerate(probs))}}
    a = [{"id": 1, "answers": noul(0.49)}, {"id": 2, "answers": noul(0.80)}, {"id": 3, "answers": choice("A", 0.6)}, {"id": 4, "answers": score([0.1, 0.6, 0.3])}, {"id": 5, "answers": noul(0.9)}]
    b = [{"id": 1, "answers": noul(0.51)}, {"id": 2, "answers": noul(0.85)}, {"id": 3, "answers": choice("A", 0.7)}, {"id": 4, "answers": score([0.1, 0.4, 0.5])}, {"id": 5, "answers": None}]
    got = p7_report.compare(a, b)
    assert got["n"] == 5 and got["same"] == 2  # row 1 crosses 0.5, row 4's most likely level changed, row 5 has no answer
    assert np.isclose(got["max_gap"], 0.2) and np.isclose(got["mean_gap"], (0.02 + 0.05 + 0.1 + 0.2) / 4)
    assert got["compared"] == 4
    # When merging, probability differences are weighted by rows answered on both sides; agreement uses all answers
    other = p7_report.compare(a[:2], b[:2])
    both = p7_report.pooled([got, other])
    assert both["n"] == 7 and np.isclose(both["agree"], 3 / 7) and np.isclose(both["mean_gap"], (0.02 + 0.05 + 0.1 + 0.2 + 0.02 + 0.05) / 6) and np.isclose(both["max_gap"], 0.2)


def test_cache_plan_alternates_the_arms_and_each_arm_gets_its_server_settings():
    plan = p7_cache.plan()
    main = [(s, arm, rep) for s, arm, rep in plan if s in p7_data.MAIN_SETS and arm != "default"]
    assert len(plan) == len(set(plan)) == 3 * 2 * p7_data.REPS + 3 + 2 * len(p7_data.SWEEP_SETS)
    assert [arm for s, arm, rep in main if s == "long"] == ["off", "table", "table", "off", "off", "table"]
    assert p7_data.server_settings("off", 300, 20) == {"KEV_PREFIX_CACHE": "0"} and p7_data.server_settings("default", 300, 20) == {}
    assert p7_data.server_settings("table", 300, 20) == {"KEV_PREFIX_CACHE": "320", "KEV_PREFIX_MAX_TOKENS": str(p7_data.MAX_TOKENS)}


def test_made_up_rows_are_distinct_and_documents_respect_the_budget():
    rows, warm = p7_data.made_reviews("rows", 50), p7_data.made_reviews("warm", 20)
    assert len({r["state"]["review"] for r in rows + warm}) == 70
    docs = p7_data.made_docs("doc", 3, 6000)
    assert len({d["state"]["reviews"] for d in docs}) == 6 and all(5000 < len(d["state"]["reviews"]) <= 6000 for d in docs)
    assert all(0 <= d["share"] <= 1 for d in docs) and {d["share"] >= 0.5 for d in docs} == {True, False}


def test_short_text_sets_take_the_whole_table_when_the_cache_budget_allows(monkeypatch):
    """The owner's decision of 2026-10-12: use as much memory as possible. Take the whole table if it fits the budget together with the warm-up rows, otherwise as many rows as fit."""
    monkeypatch.setattr(p7_data, "REHEARSAL", False)
    monkeypatch.setattr(p7_data, "BUDGET_STATES", 600)
    assert p7_data.n_short("movie") == p7_data.n_short("mams") == 600 - p7_data.WARM_SHORT
    monkeypatch.setattr(p7_data, "BUDGET_STATES", 4000)
    assert (p7_data.n_short("movie"), p7_data.n_short("mams")) == (1864, 3949)
    assert p7_data.server_settings("budget", 0, 0) == {"KEV_PREFIX_CACHE": "4000", "KEV_PREFIX_MAX_TOKENS": str(p7_data.MAX_TOKENS)}
    # The full-cache arm never needs more entries than the budget
    for name in ("movie", "mams"):
        assert p7_data.n_short(name) + p7_data.WARM_SHORT <= 4000


def test_capacity_budget_is_what_the_probe_measured():
    for model, budget in p7_data.CAPACITY.items():
        assert budget["states"] % 50 == 0 and budget["states"] > 0 and budget["bytes"] > 0, model
