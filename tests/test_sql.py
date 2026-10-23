import json

import duckdb
import httpx
import pytest

from kevdb import JevClient
from kevdb.mock import mock_transport

from kevdb import Backend, register, register_cascade


@pytest.fixture
def db(tmp_path):
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT i AS id, CASE WHEN i % 4 = 0 THEN 'a' ELSE 'b' END AS grp, 'text ' || (i % 30) AS txt FROM range(200) r(i)")
    jev = Backend("jev", lambda: JevClient("k", provider="typesafe", transport=mock_transport()), cache_path=tmp_path / "c.sqlite")
    register(con, "jev", jev)
    yield con, jev, tmp_path
    jev.close()


def test_functions_return_typed_answers_and_dedupe(db):
    con, jev, _ = db
    p, label, level = con.execute("""
        SELECT jev_noul(txt, 'Is it urgent?'), jev_choice(json_object('review', txt), 'Which?', '{"x": "X", "y": "Y"}'),
               jev_score(txt, 'How much?', '["none", "some", "lots"]') FROM t WHERE id = 3""").fetchone()
    assert 0 <= p <= 1 and label in ("x", "y") and 0 <= level <= 2
    jev.reset()
    assert con.execute("SELECT count(*) FROM t WHERE jev_noul(txt, 'Is it urgent?') IS NOT NULL").fetchone() == (200,)
    assert jev.meter.calls == 30  # only 30 distinct texts among the 200 rows
    assert con.execute("SELECT jev_noul(NULL, 'Is it urgent?')").fetchone() == (None,)

    # Several questions packed together: one request per row, P(yes) for each question in order
    jev.reset()
    probs = con.sql("""SELECT jev_nouls(txt, '["Is it urgent?", "Is it long?", "Is it rude?"]') FROM t""").fetchall()
    assert jev.meter.calls == 30 and all(len(p) == 3 and all(0 <= x <= 1 for x in p) for (p,) in probs)
    assert con.execute("SELECT jev_nouls(NULL, '[]')").fetchone() == (None,)


def test_cheap_filter_runs_before_the_model(db):
    con, jev, _ = db
    jev.namespace = "cold:"
    con.sql("SELECT count(*) FROM t WHERE grp = 'a' AND id < 40 AND jev_noul(txt || id, 'Is it urgent?') >= 0.5").fetchall()
    assert jev.meter.calls == 10  # only the 10 rows that pass the ordinary condition were asked


def test_cascade_escalates_only_unsure_rows(db):
    con, jev, tmp_path = db
    asked = []

    def handler(request):
        body = json.loads(request.content)
        asked.append(body["state"])
        kind = body["questions"]["q"]["type"]
        answer = {"type": "noul", "noul": 1.0} if kind == "noul" else {"type": "choice", "choice": "y", "confidence": 1.0}
        return httpx.Response(200, json={"model": "m", "answers": {"q": answer}, "usage": {"input_tokens": 10, "output_tokens": 0}})

    strong = Backend("llm", lambda: JevClient("k", provider="typesafe", transport=httpx.MockTransport(handler)), cache_path=tmp_path / "s.sqlite")
    register_cascade(con, "cas", jev, strong, lo=0.1, hi=0.9, min_conf=0.9)
    rows = con.sql("SELECT txt, jev_noul(txt, 'Urgent?') AS p, cas_noul(txt, 'Urgent?') AS c FROM (SELECT DISTINCT txt FROM t)").fetchall()
    unsure = {txt for txt, p, _ in rows if 0.1 < p < 0.9}
    assert set(asked) == unsure and 0 < len(unsure) < 30
    assert all(c == (1.0 if txt in unsure else p) for txt, p, c in rows)

    asked.clear()
    opts = '{"x": "X", "y": "Y"}'
    labels = con.sql(f"SELECT txt, jev_choice(txt, 'Which?', '{opts}'), cas_choice(txt, 'Which?', '{opts}') FROM (SELECT DISTINCT txt FROM t)").fetchall()
    assert all(c == ("y" if txt in asked else j) for txt, j, c in labels) and 0 < len(asked) < 30
    strong.close()


def test_join_blocks_ask_once_per_left_record(db):
    con, jev, _ = db
    jev.reset()
    params = {"a": json.dumps({"name": "a"}), "c": json.dumps([{"name": f"b{i}"} for i in range(4)]), "crit": '{"true": "same", "false": "different"}'}
    probs, pick = con.execute("SELECT jev_match($a, $c, 'Same as `record_b`?', $crit), jev_pick($a, $c, 'Which one?', 'None of them.')", params).fetchone()
    assert jev.meter.calls == 2 and len(probs) == 4 and all(0 <= p <= 1 for p in probs)
    assert 0 <= pick["pos"] <= 4 and 0 <= pick["confidence"] <= 1
    # Rows without candidates send no request; rows with a NULL argument return NULL
    assert con.execute("SELECT jev_match($a, '[]', 'q', $crit), jev_pick($a, '[]', 'q', 'none'), jev_match(NULL, $c, 'q', $crit), jev_pick($a, NULL, 'q', 'none')", params).fetchone() == (
        [], {"pos": 0, "confidence": None}, None, None)
    assert jev.meter.calls == 2
