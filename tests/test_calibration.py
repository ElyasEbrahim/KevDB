import json
import sys
from pathlib import Path

import duckdb
import httpx
import numpy as np
import pytest

from kevdb import JevClient
from kevdb.mock import mock_transport

from kevdb import Backend, calibration, register

QUESTION, CRITERIA = "Is it urgent?", '{"true": "It needs action today.", "false": "It can wait."}'


@pytest.fixture
def db(tmp_path):
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT i AS id, 'text ' || i AS txt, (i % 3 = 0)::INT AS label FROM range(400) r(i)")
    jev = Backend("jev", lambda: JevClient("k", provider="typesafe", transport=mock_transport()), cache_path=tmp_path / "c.sqlite")
    register(con, "jev", jev)
    calibration.install(con, "jev", jev.model)
    yield con, jev
    jev.close()


def test_fit_is_the_stage_6_recalibration():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "database"))
    from p6_report import platt

    rng = np.random.default_rng(0)
    p = rng.random(200)
    y = (rng.random(200) < p ** 2).astype(int)
    a, b = calibration.platt(p, y)
    assert np.allclose(calibration.apply(p, a, b), platt(p, y)(p), atol=1e-12)
    # Labels that are all of one class still give finite parameters
    assert all(np.isfinite(calibration.platt(p, np.zeros(200, int))))


def test_parameters_in_the_table_are_applied_in_sql(db):
    con, jev = db
    raw = con.sql("SELECT id, label, jev_noul(txt, 'Is it urgent?') AS p FROM t ORDER BY id").fetchnumpy()
    # Without parameters, p is returned unchanged
    same = con.sql("SELECT jev_noul_cal(txt, 'Is it urgent?', 't') AS p FROM t ORDER BY id").fetchnumpy()["p"]
    assert np.array_equal(same, raw["p"])

    a, b = calibration.fit(con, jev.model, QUESTION, "", "t", raw["p"][:200], raw["label"][:200])
    got = con.sql("SELECT jev_noul_cal(txt, 'Is it urgent?', 't') AS p FROM t ORDER BY id").fetchnumpy()["p"]
    assert np.allclose(got, calibration.apply(raw["p"], a, b), atol=1e-12) and not np.allclose(got, raw["p"])
    assert con.sql("SELECT labels, positives, question FROM semantic_calibration").fetchall() == [(200, int(raw["label"][:200].sum()), QUESTION)]

    # Other scopes, questions and models are unaffected; parameters with scope '' are this predicate's default and give way to a specific scope
    for sql in ("SELECT jev_noul_cal(txt, 'Is it urgent?', 'other') AS p FROM t ORDER BY id", "SELECT jev_noul_cal(txt, 'Is it long?', 't') AS p FROM t ORDER BY id"):
        untouched = con.sql(sql.replace("_cal", "").replace(", 'other')", ")").replace(", 't')", ")")).fetchnumpy()["p"]
        assert np.array_equal(con.sql(sql).fetchnumpy()["p"], untouched)
    calibration.store(con, jev.model, QUESTION, "", "", -2.0, 0.5, 0, 0)
    assert np.allclose(con.sql("SELECT jev_noul_cal(txt, 'Is it urgent?', 'other') AS p FROM t ORDER BY id").fetchnumpy()["p"], calibration.apply(raw["p"], -2.0, 0.5), atol=1e-12)
    assert np.allclose(con.sql("SELECT jev_noul_cal(txt, 'Is it urgent?', 't') AS p FROM t ORDER BY id").fetchnumpy()["p"], got, atol=1e-12)
    calibration.store(con, "another-model", QUESTION, "", "elsewhere", 5.0, 5.0, 0, 0)
    assert np.allclose(con.sql("SELECT jev_noul_cal(txt, 'Is it urgent?', 'elsewhere') AS p FROM t ORDER BY id").fetchnumpy()["p"], calibration.apply(raw["p"], -2.0, 0.5), atol=1e-12)

    # Change the parameters in the table and the next query uses them; each distinct input in a query is still asked only once
    calibration.store(con, jev.model, QUESTION, "", "t", 0.0, 1.0, 0, 0)
    jev.reset()
    assert np.allclose(con.sql("SELECT jev_noul_cal(txt, 'Is it urgent?', 't') AS p FROM t ORDER BY id").fetchnumpy()["p"], np.clip(raw["p"], 1e-3, 1 - 1e-3), atol=1e-12)
    assert jev.meter.calls == 400 and jev.meter.fresh == 0
    assert con.execute("SELECT jev_noul_cal(NULL, 'Is it urgent?', 't')").fetchone() == (None,)


def test_each_row_can_carry_its_own_predicate(db):
    """When the question is a column of the table, each predicate in one query gets its own parameters (how the 21 restaurant predicates are used).
    The caller's table deliberately has columns named like the parameter table's (question, scope, model, a, b, predicate): the parameter table must not capture them."""
    con, jev = db
    con.execute("""CREATE TABLE asks AS SELECT t.id, t.txt, q.question, 't' AS scope, 'not-the-model' AS model, 9.0 AS a, 9.0 AS b, 'x' AS predicate
                   FROM t, (VALUES ('Is it urgent?'), ('Is it long?'), ('Is it rude?')) q(question) WHERE t.id < 50""")
    calibration.store(con, jev.model, "Is it urgent?", "", "t", 1.0, 2.0, 0, 0)
    calibration.store(con, jev.model, "Is it long?", "", "", -1.0, 0.5, 0, 0)
    raw = con.sql("SELECT id, question, jev_noul(txt, question) AS p FROM asks ORDER BY id, question").fetchdf()
    rule = {"Is it urgent?": lambda p: calibration.apply(p, 1.0, 2.0), "Is it long?": lambda p: calibration.apply(p, -1.0, 0.5), "Is it rude?": lambda p: p}
    raw["want"] = [float(rule[q](np.array([p]))[0]) for q, p in zip(raw["question"], raw["p"])]
    kept = {q: int((rows["want"] >= 0.5).sum()) for q, rows in raw.groupby("question")}
    # This data catches the bug where every row gets the same parameters: that would give a different total
    assert int((calibration.apply(raw["p"].to_numpy(), 1.0, 2.0) >= 0.5).sum()) != sum(kept.values()) != int((raw["p"] >= 0.5).sum())
    for scope in ("'t'", "scope"):  # the scope can be a constant or a column
        call = f"jev_noul_cal(txt, question, {scope})"
        got = con.sql(f"SELECT id, question, {call} AS pc FROM asks ORDER BY id, question").fetchdf()
        assert np.allclose(got["pc"], raw["want"], atol=1e-12)
        # The same expression in WHERE, an aggregate FILTER, GROUP BY and a subquery must agree with the result above
        assert con.execute(f"SELECT count(*) FROM asks WHERE {call} >= 0.5").fetchone()[0] == sum(kept.values())
        assert con.execute(f"SELECT count(*) FILTER ({call} >= 0.5), avg({call}) FROM asks").fetchone() == pytest.approx((sum(kept.values()), raw["want"].mean()))
        assert dict(con.sql(f"SELECT question, count(*) FROM asks WHERE {call} >= 0.5 GROUP BY 1").fetchall()) == {q: n for q, n in kept.items() if n}
        assert con.execute(f"SELECT count(*) FROM (SELECT {call} AS pc FROM asks) WHERE pc >= 0.5").fetchone()[0] == sum(kept.values())
    # Storing raw probabilities in a table first and calibrating them with calibrated gives the same result
    con.execute("CREATE TABLE judged AS SELECT id, question, scope, model, jev_noul(txt, question) AS p FROM asks")
    assert con.execute("SELECT count(*) FROM judged WHERE calibrated(p, ?, predicate_key(question, ''), scope) >= 0.5", [jev.model]).fetchone()[0] == sum(kept.values())


def test_noul_with_criteria_sends_them_and_keys_its_own_parameters(tmp_path):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"model": "m", "answers": {"q": {"type": "noul", "noul": 0.25}}, "usage": {"input_tokens": 10, "output_tokens": 0}})

    con = duckdb.connect()
    b = Backend("jev", lambda: JevClient("k", provider="typesafe", transport=httpx.MockTransport(handler)), cache_path=tmp_path / "c.sqlite")
    register(con, "jev", b)
    calibration.install(con, "jev", b.model)
    assert con.execute("SELECT jev_noulc(json_object('a', 1), ?, ?)", [QUESTION, CRITERIA]).fetchone() == (0.25,)
    assert seen == [{"state": {"a": 1}, "model": b.client.model, "questions": {"q": {"type": "noul", "instructions": QUESTION, "criteria": json.loads(CRITERIA)}}}]
    assert con.execute("SELECT predicate_key(?, ?)", [QUESTION, CRITERIA]).fetchone() == (calibration.predicate_key(QUESTION, CRITERIA),)
    # With and without criteria are two different predicates, each with its own parameters
    calibration.store(con, b.model, QUESTION, CRITERIA, "t", 1.0, 1.0, 0, 0)
    with_criteria, without = con.execute("SELECT jev_noulc_cal(json_object('a', 1), ?, ?, 't'), jev_noul_cal(json_object('a', 1), ?, 't')", [QUESTION, CRITERIA, QUESTION]).fetchone()
    assert with_criteria == pytest.approx(float(calibration.apply(0.25, 1.0, 1.0))) and without == 0.25
    b.close()
