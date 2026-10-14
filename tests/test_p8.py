"""Easy mistakes in phase 8: how requests are built, how candidates are ordered, how end-to-end F1 is computed. Calls no model and reads no data files or collected results."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "database"))

import p8_join as J

A = {"name": "left"}
BS = [{"name": f"right {i}"} for i in range(3)]


@pytest.fixture
def toy(monkeypatch):
    """Two left records with 3 candidates each; L1 has one true match among its candidates, L2 has two true matches, one among the candidates and one that blocking missed."""
    left = {"L1": A, "L2": A}
    right = {f"R{i}": b for i, b in enumerate(BS)} | {"R9": {"name": "missed"}}
    truth = {("L1", "R0"), ("L2", "R1"), ("L2", "R9")}
    monkeypatch.setitem(J._DATA, "abt-buy", (left, right, truth, {"L1": ["R0", "R1", "R2"], "L2": ["R0", "R1", "R2"]}))


def test_pair_request_is_the_one_table_match_sent():
    task = J.tables.Task(name="t", input="", questions={"same": J.tables.KINDS["product"]}, state_fields=["record_a", "record_b"])
    state, questions = J.build("abt-buy", "pair", A, BS[:1])
    assert state == task.state_of({"record_a": A, "record_b": BS[0]}) and questions == {"same": J.tables.KINDS["product"].payload()}


def test_packed_requests_carry_every_candidate_once_and_in_order():
    q = J.tables.KINDS["product"]
    state, questions = J.build("abt-buy", "Q", A, BS)
    assert state == {"record_a": A} and list(questions) == ["c1", "c2", "c3"]
    assert [x["instructions"] for x in questions.values()] == [{"question": q.instructions, "record_b": b} for b in BS]
    assert all(x["type"] == "noul" and x["criteria"] == q.criteria for x in questions.values())

    state, questions = J.build("abt-buy", "S", A, BS)
    assert state == {"record_a": A, "candidates": {"c1": BS[0], "c2": BS[1], "c3": BS[2]}}
    assert ["`candidates.%s`" % n in x["instructions"] and "record_b" not in x["instructions"] for n, x in questions.items()] == [True] * 3

    state, questions = J.build("abt-buy", "C", A, BS)
    options = questions["match"]["criteria"]
    assert state == {"record_a": A} and list(options) == ["c1", "c2", "c3", "none"] and [options[f"c{i + 1}"] for i in range(3)] == BS


def test_orders_are_permutations_of_the_same_candidates():
    rids = [f"R{i}" for i in range(10)]
    shuffled = J.arrange("abt-buy", "L1", rids, "random")
    assert J.arrange("abt-buy", "L1", rids, "rank") == rids and J.arrange("abt-buy", "L1", rids, "reversed") == rids[::-1]
    assert sorted(shuffled) == sorted(rids) and shuffled == J.arrange("abt-buy", "L1", rids[::-1], "random") and shuffled not in (rids, rids[::-1])
    # When candidates are added, the relative order of the existing ones stays the same
    assert [r for r in J.arrange("abt-buy", "L1", rids, "random") if r in rids[:5]] == J.arrange("abt-buy", "L1", rids[:5], "random")


def test_every_step_has_its_own_files():
    for source in J.SOURCES:
        steps = J.steps(source)
        assert len({J.file(source, s, "run.json") for s in steps}) == len(steps) == len({J.file(source, s, "jsonl") for s in steps})


def test_judgments_and_end_to_end_f1(toy):
    noul = [{"lid": "L1", "rids": ["R1", "R0", "R2"], "p": [0.2, 0.9, 0.6]}, {"lid": "L2", "rids": ["R0", "R1", "R2"], "error": "x"}]
    judged = J.judgments(noul, "Q")
    assert judged["L1", "R0"] == (True, 0.9) and judged["L1", "R2"] == (True, 0.6) and judged["L2", "R1"] == (False, None)
    t = J.tally(judged, "abt-buy", ["L1", "L2"])
    assert t.tolist() == [[1, 1, 1], [0, 0, 2]]  # L2's request has no answer: both true matches count as missed, including the one blocking never offered
    precision, recall, f1 = J.prf(t.sum(0))
    assert (precision, recall) == (0.5, 1 / 3) and np.isclose(f1, 0.4)

    choice = [{"lid": "L1", "rids": ["R1", "R0", "R2"], "choice": "c2", "probs": {"c1": 0.1, "c2": 0.7, "c3": 0.1, "none": 0.1}},
              {"lid": "L2", "rids": ["R0", "R1", "R2"], "choice": "none", "probs": {"c1": 0.1, "c2": 0.3, "c3": 0.1, "none": 0.5}}]
    judged = J.judgments(choice, "C")
    assert [k for k, (yes, _) in judged.items() if yes] == [("L1", "R0")] and judged["L2", "R1"] == (False, 0.3)
    assert J.agreement(judged, J.judgments(noul, "Q")) == 5 / 6  # only (L1, R2) is judged differently


def test_f1_difference_is_zero_for_identical_judgments_and_signed():
    rng = np.random.default_rng(0)
    a = np.stack([rng.integers(0, 2, 200), rng.integers(0, 2, 200), np.ones(200)], axis=1).astype(float)
    assert J.f1_difference(a, a) == (0.0, 0.0, 0.0)
    worse = a.copy()
    worse[:, 0] = 0
    d, lo, hi = J.f1_difference(worse, a)
    assert d < 0 and lo <= d <= hi < 0


def test_sql_functions_send_the_requests_this_stage_collected(tmp_path):
    """Requests sent by jev_match / jev_pick must be identical to those built by build: only then can SQL replay this phase's cache, and only then do its conclusions apply to them."""
    import json

    import duckdb
    import httpx

    from kevdb import JevClient
    from kevdb import Backend, register

    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append((body["state"], body["questions"]))
        answers = {n: {"type": "noul", "noul": 0.5} if q["type"] == "noul" else {"type": "choice", "choice": "c1", "confidence": 1.0} for n, q in body["questions"].items()}
        return httpx.Response(200, json={"model": "m", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 0}})

    backend = Backend("jev", lambda: JevClient("k", provider="typesafe", transport=httpx.MockTransport(handler)), cache_path=tmp_path / "c.sqlite")
    con = duckdb.connect()
    register(con, "jev", backend)
    q = J.tables.KINDS["product"]
    params = {"a": json.dumps(A), "c": json.dumps(BS), "q": q.instructions, "crit": json.dumps(q.criteria), "pick": J.CHOICE["product"], "none": J.NONE["product"]}
    con.execute("SELECT jev_match($a, $c, $q, $crit), jev_pick($a, $c, $pick, $none)", params).fetchall()
    backend.close()
    assert sorted(sent, key=lambda x: len(x[1])) == sorted([J.build("abt-buy", "C", A, BS), J.build("abt-buy", "Q", A, BS)], key=lambda x: len(x[1]))
