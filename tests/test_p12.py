"""Easy mistakes in phase 12: a record on both sides of the split, budgets that are not nested or overlap their temperature slice, a
validation sample other than the one Jev was asked, the order and request sets of the runs, what a checkpoint must contain to count, the
training recipe, and the n* line. Calls no model and reads no data file."""

import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "database"))

import p10_data as P10
import p12_data as D
import p12_eval as P

np = pytest.importorskip("numpy")
import p12_report as R  # noqa: E402


def made(n: int, prefix: str, records: int) -> list[dict]:
    """n pairs over a small set of records, so that records recur across pairs as in the real datasets."""
    rng = random.Random(prefix)
    return [{"id": f"{prefix}-{i}", "record_a": {"title": "a"}, "record_b": {"title": "b"}, "label": int(rng.random() < 0.2),
             "raw": (f"{prefix}a{rng.randrange(records)}", f"{prefix}b{rng.randrange(records)}")} for i in range(n)]


@pytest.fixture
def fake(monkeypatch):
    splits = {"train": made(6000, "r", 300), "valid": made(800, "r", 300), "test": made(800, "r", 300)}
    monkeypatch.setattr(D, "load_pairs", lambda dataset, split: list(splits[split]))
    monkeypatch.setattr(D, "JEV_VALID_SAMPLE", 600)
    monkeypatch.setattr(D, "BUDGETS", (100, 300, 1000))
    return splits


def test_no_record_of_a_test_pair_is_in_the_label_pool(fake):
    tests, pool = D.test_pairs("abt-buy"), D.label_pool("abt-buy")
    assert tests and len(pool) >= 1000
    assert not {r for p in tests for r in p["raw"]} & {r for p in pool for r in p["raw"]}
    assert all(D.on_test_side("abt-buy", r) for p in tests for r in p["raw"])
    assert not any(D.on_test_side("abt-buy", r) for p in pool for r in p["raw"])
    assert {p["id"] for p in pool} <= {p["id"] for p in fake["train"]}  # labels only ever come from the training split


def test_budgets_are_nested_and_hold_out_their_last_fifth(fake):
    pool = [p["id"] for p in D.label_pool("abt-buy")]
    for n in D.BUDGETS:
        train, held = D.budget("abt-buy", n)
        assert (len(train), len(held)) == (n * 4 // 5, n // 5)
        assert [p["id"] for p in train + held] == pool[:n]
    small_train, small_held = D.budget("abt-buy", 100)
    big_train, _ = D.budget("abt-buy", 300)
    assert [p["id"] for p in small_train + small_held] == [p["id"] for p in big_train[:100]]
    with pytest.raises(ValueError):
        D.budget("abt-buy", len(pool) + 1)


def test_the_validation_sample_is_drawn_as_jev_was_asked(fake):
    want = [p["id"] for p in random.Random(0).sample(fake["valid"], 600)]  # experiments/entity_match/run.py --sample 600
    assert [p["id"] for p in D.jev_valid("abt-buy")] == want
    tests = D.test_pairs("abt-buy")
    assert {p["id"] for p in tests} <= {p["id"] for p in fake["test"]} | set(want)


def test_a_training_record_is_the_evaluation_request_plus_a_label():
    pair = {"id": "train-1", "record_a": {"title": "x"}, "record_b": {"title": "y"}, "label": 1}
    record = D.labelled(pair, "dblp-acm")
    assert record["questions"]["same"].pop("label") is True and record == D.request(pair, "dblp-acm")
    assert D.request(pair, "dblp-acm") == P10.request(pair, "dblp-acm")  # phase 10's first-draft question, unchanged


def test_runs_are_named_ordered_and_asked_as_written():
    assert P.ARMS[0] == "released" and set(P.TRAINED) < set(P.ARMS) and len(P.ARMS) == 1 + 4 * (1 + len(D.BUDGETS) + len(D.REFERENCE_BUDGETS))
    for d in D.EM:
        mine = [a for a in P.ARMS if a != "released" and D.parse(a)[0] == d]
        assert mine[0] == D.arm(d, "loo", 0) and [D.parse(a)[1:] for a in mine[1:]] == [("loo", n) for n in D.BUDGETS] + [("rel", n) for n in D.REFERENCE_BUDGETS]
        assert P.sets(D.arm(d, "loo", 0)) == [f"side.{d}"]
        assert P.sets(D.arm(d, "loo", D.BUDGETS[0])) == [f"side.{d}", f"slice.{d}.{D.BUDGETS[0]}", "g1", "g2", "g3"]
    assert P.sets("released") == [f"side.{d}" for d in D.EM]
    assert set(D.REFERENCE_BUDGETS) <= set(D.BUDGETS)
    for bad in ("abt-buy.loo", "abt-buy.base.100", "beer.loo.100"):
        with pytest.raises(ValueError):
            D.parse(bad)


def test_n_zero_is_phase_10s_checkpoint_under_phase_10s_cache_name(monkeypatch, tmp_path):
    assert P.checkpoint("kev-4b", "abt-buy.loo.0") == P.P10_RUNS / "kev-4b" / "loo-abt-buy"
    assert P.checkpoint("kev-4b", "abt-buy.loo.100") == P.RUNS / "kev-4b" / "abt-buy.loo.100"
    monkeypatch.setattr(P, "weights_digest", lambda path: "0123456789abcdef")
    assert P.cache_id("kev-4b", "abt-buy.loo.0") == "p10/kev-4b/loo-abt-buy@0123456789abcdef"
    assert P.cache_id("kev-4b", "abt-buy.rel.300") == "p12/kev-4b/abt-buy.rel.300@0123456789abcdef"
    assert P.cache_id("kev-4b", "released") == "jaredpalmer/kev-4b@v1.0"
    # n = 0 counts only on the very weights phase 10 evaluated
    path = tmp_path / "loo-abt-buy"
    path.mkdir()
    for name in ("head.pt", "adapter_model.safetensors"):
        (path / name).write_bytes(b"w")
    monkeypatch.setattr(P, "checkpoint", lambda model, arm: path)
    assert "other weights" in P.not_trained("kev-4b", "abt-buy.loo.0")
    monkeypatch.setattr(P, "weights_digest", lambda p: P.P10_DIGESTS["abt-buy"])
    assert P.not_trained("kev-4b", "abt-buy.loo.0") == ""


def test_a_run_must_have_trained_on_every_record_for_every_epoch(monkeypatch, tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "abt-buy.100.jsonl").write_text("{}\n" * 80)
    monkeypatch.setattr(D, "OUT", tmp_path)
    assert P.training_records("abt-buy.loo.100") == P.training_records("abt-buy.rel.100") == 2 * 80 == D.EPOCHS * 80


def test_the_training_script_follows_the_authors_own_data_recipe():
    text = (ROOT / "experiments" / "database" / "p12_train.sh").read_text()
    assert re.search(rf"BATCH=1; ACCUM=8; CHECKPOINTING=1; WEIGHTS_DTYPE=fp32; EPOCHS={D.EPOCHS}; LR=2e-5", text)
    assert "p10_kev_train.py\" kev.train" in text and "--suite" not in text and "--replay" not in text and "--p_none_pair" not in text
    assert 'INIT="$DIR/p10/$MODEL/loo-$DATASET"' in text and 'INIT="jaredpalmer/$MODEL@$WEIGHTS_TAG"' in text
    assert '--init_from "$INIT"' in text and "--seed 1" in text and "--device mps" in text
    run = (ROOT / "experiments" / "database" / "p12_run.sh").read_text()
    assert "p12_train.sh" in run and "experiments/database/p11_" not in run and "pgrep -P \"$job\" -f 'kev\\.train'" in run


def test_n_star_is_the_smallest_settled_budget_that_meets_both_lines(monkeypatch):
    monkeypatch.setattr(D, "BUDGETS", (100, 300, 1000))
    assert R.n_star({0: True, 100: True, 300: True, 1000: True}) == (0, "0", True)
    assert R.n_star({0: False, 100: False, 300: True, 1000: False}) == (300, "300", False)
    assert R.n_star({0: False, 100: False, 300: False, 1000: False}) == (None, "more than 1,000", False)
    assert R.n_star({0: False, 300: True, 1000: True})[1] == "undetermined: runs missing"  # 100 is missing below the first budget met
    assert R.n_star({0: True, 300: True})[0] == 0  # a missing larger run does not undo a smaller one


def test_importing_phase_12_leaves_phase_10s_evaluation_untouched():
    import p10_eval as E
    assert E.NAMESPACE.startswith("p10") and E.ARMS[-1] == "released" and "p12" not in str(E.RUNS)
    assert E.sets("loo-abt-buy")[0] == "test.abt-buy" and E.rows is not P.rows and E.checkpoint is not P.checkpoint
