"""Easy mistakes in phase 14: that a run is phase 10's Kev-0.8B arm with seed 1 and only the padding launcher added, which arms run, and how
the pre-registered reading follows from the number of arms on which padded seed 1 fails the movie-review guard. Calls no model."""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "database"))

import p10_eval as E  # noqa: E402
import p14_eval as P  # noqa: E402

np = pytest.importorskip("numpy")
import p14_report as Q  # noqa: E402


def test_the_runs_are_phase_13s_kev_08b_arms_with_seed_one():
    assert P.MODELS == ("kev-0.8b",) and P.SEEDS == (1,)
    assert P.arms("kev-0.8b") == ("loo-amazon-google.seed1", "loo-walmart-amazon.seed1", "loo-dblp-acm.seed1")
    assert P.sets("loo-dblp-acm.seed1") == ["test.dblp-acm", "pool.abt-buy", "pool.amazon-google", "pool.walmart-amazon", "g1", "g2", "g3"]


def test_the_training_command_is_phase_10s_with_seed_one_through_the_launcher():
    mine = (ROOT / "experiments" / "database" / "p14_train.sh").read_text()
    theirs = (ROOT / "experiments" / "database" / "p10_train.sh").read_text()
    command = lambda text: re.search(r"sandbox-exec -p .*?--out \"\$OUT\"", text, re.S).group(0)  # noqa: E731
    assert command(mine).replace('--seed "$SEED"', "--seed 1") == command(theirs)
    assert "p10_kev_train.py\" kev.train" in command(mine) and 'RUNS="$DIR/p14"' in mine
    assert '[ "$MODEL" = kev-0.8b ] && [ "$SEED" = 1 ]' in mine
    run = (ROOT / "experiments" / "database" / "p14_run.sh").read_text()
    assert "p14_train.sh" in run and "experiments/database/p13_" not in run and "pgrep -P \"$job\" -f 'kev\\.train'" in run


def test_cache_names_are_phase_14s(monkeypatch):
    monkeypatch.setattr(P, "_p10_cache_id", lambda model, arm: f"p10/{model}/{arm}@0123")
    assert P.cache_id("kev-0.8b", "loo-dblp-acm.seed1") == "p14/kev-0.8b/loo-dblp-acm.seed1@0123"


def test_the_reading_follows_the_pooled_interval():
    assert Q.reading((0.012, 0.004, 0.020)).startswith("B")  # interval above zero: padding moves seed 1
    assert Q.reading((0.002, -0.005, 0.009)).startswith("A")  # interval contains zero: padded seed 1 like phase 10's
    assert Q.reading((-0.012, -0.020, -0.004)).startswith("undetermined")


def test_the_four_conditions_read_the_right_folders():
    labels = [label for label, _ in Q.CONDITIONS]
    assert labels[0].startswith("seed 1, phase 10") and labels[1].startswith("seed 1, padded")
    assert Q.CONDITIONS[0][1]("loo-dblp-acm") == P.P10_OUT / "kev-0.8b" / "loo-dblp-acm"
    assert Q.CONDITIONS[1][1]("loo-dblp-acm") == P.OUT / "kev-0.8b" / "loo-dblp-acm.seed1"
    assert Q.CONDITIONS[3][1]("loo-dblp-acm") == P.P13_OUT / "kev-0.8b" / "loo-dblp-acm.seed3"


def test_importing_phase_14_leaves_phase_10s_evaluation_untouched():
    assert E.NAMESPACE.startswith("p10") and E.ARMS[-1] == "released" and "p14" not in str(E.RUNS) and E.sets is not P.sets


def test_every_phases_training_script_switches_to_its_own_rehearsal():
    """A training script copied from another phase's must test its own phase's rehearsal variable: phase 14's first copy tested phase 13's,
    so its rehearsal trained on real rows (stopped before the first step, see p14.txt)."""
    for phase in (11, 12, 13, 14):
        text = (ROOT / "experiments" / "database" / f"p{phase}_train.sh").read_text()
        switches = set(re.findall(r'\[ "\$\{(P\d+)_REHEARSAL:-\}" = 1 \]', text))
        assert switches == {f"P{phase}"}, (phase, switches)
