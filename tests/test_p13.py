"""Easy mistakes in phase 13: which runs each model trains, that a run is phase 10's arm with only the seed changed (same data, same request
sets without the end-to-end readings, the same record count), and how a verdict over the seeds is read. Calls no model and reads no answer."""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "database"))

import p10_eval as E  # noqa: E402
import p13_eval as P  # noqa: E402

np = pytest.importorskip("numpy")
import p13_report as Q  # noqa: E402


def test_the_runs_are_the_close_arms_with_seeds_two_and_three():
    assert P.CLOSE == {"kev-0.8b": ("loo-amazon-google", "loo-walmart-amazon", "loo-dblp-acm"), "kev-4b": ("loo-amazon-google",)}
    assert P.MODELS == ("kev-0.8b", "kev-4b")  # the small model first
    assert P.arms("kev-4b") == ("loo-amazon-google.seed2", "loo-amazon-google.seed3")
    assert len(P.arms("kev-0.8b")) == 6 and all(P.seed(a) in (2, 3) for a in P.arms("kev-0.8b"))
    assert {P.phase10_arm(a) for a in P.arms("kev-0.8b")} == set(P.CLOSE["kev-0.8b"])


def test_a_run_is_asked_its_phase_10_arms_sets_without_the_end_to_end_readings():
    assert P.sets("loo-dblp-acm.seed2") == ["test.dblp-acm", "pool.abt-buy", "pool.amazon-google", "pool.walmart-amazon", "g1", "g2", "g3"]
    assert "e2e.dblp-scholar" in E.sets("loo-dblp-acm")  # phase 10 asked it; phase 13 does not
    for arm in P.arms("kev-0.8b"):
        assert set(P.sets(arm)) == {s for s in E.sets(P.phase10_arm(arm)) if not s.startswith(("e2e.", "order."))}


def test_a_run_must_train_on_phase_10s_records_plus_the_replay(monkeypatch, tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "loo-walmart-amazon.jsonl").write_text("{}\n" * 6000)
    monkeypatch.setattr(P, "P10_OUT", tmp_path)
    assert P.training_records("loo-walmart-amazon.seed3") == 6000 + E.REPLAY


def test_the_training_script_changes_only_the_seed():
    mine = (ROOT / "experiments" / "database" / "p13_train.sh").read_text()
    theirs = (ROOT / "experiments" / "database" / "p10_train.sh").read_text()
    command = lambda text: re.search(r"sandbox-exec -p .*?--out \"\$OUT\"", text, re.S).group(0)  # noqa: E731
    assert command(mine).replace('--seed "$SEED"', "--seed 1") == command(theirs)
    assert 'DATA="$ROOT/out/database/p10/train/$PHASE10_ARM.jsonl"' in mine and 'RUNS="$DIR/p13"' in mine
    run = (ROOT / "experiments" / "database" / "p13_run.sh").read_text()
    assert "p13_train.sh" in run and "experiments/database/p12_" not in run and "pgrep -P \"$job\" -f 'kev\\.train'" in run


def test_cache_names_never_reuse_phase_10s(monkeypatch):
    monkeypatch.setattr(P, "_p10_cache_id", lambda model, arm: f"p10/{model}/{arm}@0123")
    assert P.cache_id("kev-0.8b", "loo-dblp-acm.seed2") == "p13/kev-0.8b/loo-dblp-acm.seed2@0123"


def test_a_verdict_is_stable_only_if_all_seeds_agree():
    assert Q.stable([True, True, True]) and Q.stable([False, False, False]) and not Q.stable([True, False, True])
    x = {"on_par": False, "criteria": {"acc": True, "auc": False, "brier": True}, "guard": {"g1": {"pass": False}, "g2": {"pass": True}}}
    assert Q.verdict(x) == "below: auc, guard g1" and Q.verdict(x | {"on_par": True}) == "on par"


def test_importing_phase_13_leaves_phase_10s_evaluation_untouched():
    assert E.NAMESPACE.startswith("p10") and E.ARMS[-1] == "released" and "p13" not in str(E.RUNS) and E.sets is not P.sets
