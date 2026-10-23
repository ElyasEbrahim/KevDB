"""Easy mistakes in phase 11: what each arm trains on, how the product wordings are assigned, that no record of dblp-acm's test split is
trained on, that training records are the evaluation requests plus a label, and which request sets each arm is asked. Calls no model and
reads no data file."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "database"))

import p10_data as P10
import p11_data as D
import p11_eval as P

PAIR = {"id": "train-7", "record_a": {"title": "a", "venue": ""}, "record_b": {"title": "b"}, "label": 1, "raw": ("ra", "rb")}


def test_every_arm_holds_out_dblp_acm_and_keeps_the_controls_product_pairs():
    assert D.PRODUCTS == P10.ARMS["loo-dblp-acm"] and D.HELD_OUT not in D.PRODUCTS
    for arm, datasets in D.ARMS.items():
        assert datasets[: len(D.PRODUCTS)] == D.PRODUCTS and D.HELD_OUT not in datasets
    assert D.ARMS["words"] == D.PRODUCTS
    assert set(D.ARMS["domains"]) - set(D.PRODUCTS) == set(D.DOMAINS) and set(D.ARMS["pubs"]) - set(D.PRODUCTS) == {D.PUBS}


def test_the_added_publication_data_are_asked_dblp_acms_own_question_and_the_domains_their_own():
    assert D.question(D.PUBS) is P10.KINDS["publication"]
    payloads = {D.question(d).payload()["instructions"] for d in D.DOMAINS} | {P10.KINDS["product"].payload()["instructions"], P10.KINDS["publication"].payload()["instructions"]}
    assert len(payloads) == 5  # three new questions, none equal to the product or publication question


def test_product_wordings_keep_wording_zero_as_phase_10_and_spread_over_five():
    assert D.PRODUCT_WORDINGS[0] is P10.KINDS["product"] and len(D.PRODUCT_WORDINGS) == D.WORDINGS == 5
    assert len({q.payload()["instructions"] for q in D.PRODUCT_WORDINGS}) == 5
    counts = [0] * 5
    for i in range(5000):
        counts[D.wording("abt-buy", f"train-{i}")] += 1
    assert all(800 < c < 1200 for c in counts) and D.wording("abt-buy", "train-1") == D.wording("abt-buy", "train-1")


def test_a_training_record_is_the_evaluation_request_plus_a_label():
    q = D.question("beer")
    assert D.request(PAIR, q) == {"state": {"record_a": {"title": "a", "venue": ""}, "record_b": {"title": "b"}}, "questions": {"same": q.payload()}}
    record = D.labelled(PAIR, q)
    assert record["questions"]["same"].pop("label") is True and record == D.request(PAIR, q) and "label" not in q.payload()


def test_pairs_sharing_a_record_with_the_held_out_test_split_are_dropped(monkeypatch):
    pairs = [PAIR | {"id": f"train-{i}", "raw": (f"r{i}", f"s{i}")} for i in range(10)]
    monkeypatch.setattr(D, "load_pairs", lambda dataset, split: list(pairs))
    monkeypatch.setattr(D, "held_out_records", lambda: {"r3", "s7"})
    kept, dropped = D.added_pairs("beer")
    assert dropped == 2 and {p["id"] for p in kept} == {f"train-{i}" for i in range(10)} - {"train-3", "train-7"}
    monkeypatch.setattr(D, "N_PUBS", 4)
    kept, dropped = D.added_pairs(D.PUBS)
    assert len(kept) == 4 and dropped == 2 and not any({*p["raw"]} & {"r3", "s7"} for p in kept)


def test_each_arm_is_asked_the_held_out_test_split_its_own_pools_and_the_guards():
    for arm in D.ARMS:
        names = P.sets(arm)
        assert names[0] == "test.dblp-acm" and not any(n.startswith("test.") for n in names[1:])
        assert {n for n in names if n.startswith("pool.")} == {f"pool.{d}" for d in D.PRODUCTS}
        assert {n for n in names if n.startswith("pool11.")} == {f"pool11.{d}" for d in D.ARMS[arm][len(D.PRODUCTS):]}
        assert {"g1", "g2", "g3"} <= set(names)
    assert "e2e.dblp-scholar" not in P.sets("pubs")  # DBLP-GoogleScholar comes from the same source as phase 8's dblp-scholar table
    assert "e2e.dblp-scholar" in P.sets("words") and "e2e.dblp-scholar" in P.sets("domains")
    assert P.ARMS == ("pubs", "words", "domains")


def test_importing_phase_11_leaves_phase_10s_evaluation_untouched():
    import p10_eval as E
    assert E.NAMESPACE.startswith("p10") and E.ARMS[-1] == "released" and "p11" not in str(E.RUNS)
    assert E.sets("loo-abt-buy")[0] == "test.abt-buy"
