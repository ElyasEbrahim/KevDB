"""Easy mistakes in phase 10: is each training record identical to the request sent at evaluation, which datasets each arm trains on, which pairs are taken. Calls no model and reads no data file."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "database"))

import p10_data as D
from kevdb import Task
from p2_collect import sha as p7_sha

PAIR = {"id": "train-3", "record_a": {"title": "a", "price": "1.99"}, "record_b": {"title": "b"}, "label": 1}


@pytest.mark.parametrize("dataset", D.EM)
def test_training_record_is_the_evaluation_request_plus_a_label(dataset):
    question = D.KINDS[D.DATASETS[dataset]]
    task = Task(name="em", input="", questions={"same": question}, state_fields=["record_a", "record_b"])
    assert D.request(PAIR, dataset) == {"state": task.state_of(PAIR), "questions": {"same": question.payload()}}
    record = D.labelled(PAIR, dataset)
    assert record["questions"]["same"].pop("label") is True and record == D.request(PAIR, dataset)
    assert D.labelled(PAIR | {"label": 0}, dataset)["questions"]["same"]["label"] is False
    assert "label" not in question.payload()  # the question itself is unchanged


def test_each_leave_one_out_arm_trains_on_the_other_three():
    assert set(D.ARMS) == {f"loo-{d}" for d in D.EM} | {"all"} and D.ARMS["all"] == D.EM
    for held in D.EM:
        assert held not in D.ARMS[f"loo-{held}"] and set(D.ARMS[f"loo-{held}"]) | {held} == set(D.EM)


def test_training_pairs_are_a_fixed_hash_ordered_prefix(monkeypatch):
    pairs = [{"id": f"train-{i}", "record_a": {"t": str(i)}, "record_b": {"t": str(i)}, "label": i % 7 == 0} for i in range(50)]
    monkeypatch.setattr(D, "N_PER_DATASET", 10)
    monkeypatch.setattr(D, "load_pairs", lambda dataset, split: list(pairs))
    first = D.train_pairs("abt-buy")
    monkeypatch.setattr(D, "load_pairs", lambda dataset, split: list(reversed(pairs)))
    assert [p["id"] for p in D.train_pairs("abt-buy")] == [p["id"] for p in first] and len(first) == 10  # independent of the order in the file
    assert [p["id"] for p in D.train_pairs("dblp-acm")] != [p["id"] for p in first]  # each dataset has its own order
    assert [p["id"] for p in first] != [f"train-{i}" for i in range(10)]


def test_label_pool_is_ordered_like_stage_7():
    assert all(D.p7_key(d, f"valid-{i}") == p7_sha(f"valid-{i}", f"p7:{d}") for d in D.EM for i in (0, 17, 1915))


def test_made_up_pairs_are_stable_and_of_both_kinds():
    rows = D.made_pairs("x", 40, "product")
    assert rows == D.made_pairs("x", 40, "product") and rows != D.made_pairs("y", 40, "product")
    assert {r["label"] for r in rows} == {0, 1} and "authors" in D.made_pairs("x", 3, "publication")[0]["record_a"]
