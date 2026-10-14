"""Easy mistakes in phase 9: how requests are built, how pairs are cut into blocks, how the F1 difference is resampled. Calls no model and reads no data files or collected results."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "database"))

import p9_join as J


@pytest.fixture(autouse=True)
def toy(monkeypatch):
    monkeypatch.setattr(J, "_REVIEWS", J.made_up(30))


def test_pair_request_is_the_one_the_stage_1_sql_sent():
    import json

    import pyarrow as pa

    from kevdb.sql import _items

    state = json.dumps({"review_1": "a", "review_2": "b"})
    (sent,) = _items("noul", pa.chunked_array([[state]]), pa.chunked_array([[J.PREDICATES["same"]]]), None)
    assert sent == J.build("same", "pair", "a", ["b"])


def test_packed_request_carries_each_partner_in_its_own_question():
    state, questions = J.build("opposite", "Q", "a", ["b", "c"])
    assert state == {"review_1": "a"} and list(questions) == ["c1", "c2"]
    assert [q["instructions"] for q in questions.values()] == [{"question": J.PREDICATES["opposite"], "review_2": x} for x in ("b", "c")]


@pytest.mark.parametrize("order", J.ORDERS)
@pytest.mark.parametrize("k", [1, 7, 10, 127])
def test_blocks_cover_every_pair_exactly_once(order, k):
    got = [(a, b) for a, bs in J.blocks(order, k) for b in bs]
    assert sorted(got) == J.pairs() and len(set(got)) == len(got) == 30 * 29 // 2
    assert all(1 <= len(bs) <= k for _, bs in J.blocks(order, k))


def test_orders_differ_and_truth_follows_the_labels():
    a = J.pairs()[0][0]
    bs = [b for x, b in J.pairs() if x == a]
    assert J.arrange(a, bs, "natural") == bs and J.arrange(a, bs, "reversed") == bs[::-1] and sorted(J.arrange(a, bs, "random")) == bs
    assert J.arrange(a, bs, "random") not in (bs, bs[::-1])
    label = {r[0]: r[2] for r in J.reviews()}
    assert J.truth("same").tolist() == [label[x] == label[y] for x, y in J.pairs()] and (J.truth("same") ^ J.truth("opposite")).all()


def test_steps_have_their_own_files_and_kev_runs_only_what_the_protocol_says():
    for source in J.SOURCES:
        steps = J.steps(source)
        assert len({J.file(source, s, "run.json") for s in steps}) == len(steps)
    kev = J.steps("kev-4b")
    assert {s["pred"] for s in kev} == {"opposite"} and [s.get("timing") for s in kev if s["form"] == "pair"] == [True]
    rows, _ = J.requests(kev[0])
    assert {r["a"] for r in rows} <= set(J.timing_lefts()) and all(len(r["bs"]) == 1 for r in rows)


def test_f1_and_its_resampled_difference():
    y = J.truth("same")
    perfect, nothing = y.copy(), np.zeros_like(y)
    assert J.prf(perfect, y)[2] == 1.0 and J.prf(nothing, y)[2] == 0.0
    assert J.f1_difference(perfect, perfect, y) == (0.0, 0.0, 0.0)
    worse = y & (np.arange(len(y)) % 3 > 0)  # miss a third of the matches
    d, lo, hi = J.f1_difference(worse, perfect, y)
    assert np.isclose(d, (J.prf(worse, y)[2] - 1) * 100) and lo <= d <= hi < 0
    # Pairs without an answer do not count as matches
    assert J.decide(np.array([0.9, np.nan, 0.2])).tolist() == [True, False, False]
