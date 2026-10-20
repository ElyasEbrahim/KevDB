"""Easy mistakes in phase 10's evaluation and analysis: which sets each arm is asked, that no label is ever sent, the replay of earlier answers,
the server's environment and identity checks, when a checkpoint counts, the temperature, the metrics and the lines. Calls no model and reads no data file."""

import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "database"))

import p10_data as D
import p10_eval as E
import p10_report as R
from kevdb.cache import cache_key

ROOT = Path(__file__).resolve().parents[1]


# ---------- Which sets each arm is asked ----------

def test_a_leave_one_out_arm_is_tested_on_its_held_out_dataset_and_never_sees_its_labels():
    for held in D.EM:
        names = E.sets(f"loo-{held}")
        assert [n for n in names if n.startswith("test.")] == [f"test.{held}"]
        assert sorted(n for n in names if n.startswith("pool.")) == sorted(f"pool.{d}" for d in D.EM if d != held)  # temperature labels of the training datasets only
        assert all(g in names for g in E.GUARDS)
    assert E.sets("loo-abt-buy")[-4:] == ["e2e.abt-buy", "order.random", "order.rank", "order.reversed"] and E.sets("loo-dblp-acm")[-1] == "e2e.dblp-scholar"
    assert not any(n.startswith(("e2e.", "order.")) for n in E.sets("loo-amazon-google") + E.sets("loo-walmart-amazon") + E.sets("all"))
    assert {n for n in E.sets("released")} >= {f"test.{d}" for d in D.EM} | {"e2e.abt-buy", "e2e.dblp-scholar", "order.rank"} | set(E.GUARDS)
    assert E.ARMS[-1] == "released" and E.TRAINED == E.ARMS[:-1]


def test_test_and_pool_requests_are_the_training_form_without_a_label(monkeypatch):
    pairs = [{"id": f"test-{i}", "record_a": {"title": f"a{i}"}, "record_b": {"title": f"b{i}", "price": ""}, "label": i % 2} for i in range(4)]
    monkeypatch.setattr(E.D, "load_pairs", lambda dataset, split: pairs)
    monkeypatch.setattr(E.D, "pool_pairs", lambda dataset: pairs[:2])
    monkeypatch.setattr(E, "_ROWS", {})
    for name in ("test.dblp-acm", "pool.abt-buy"):
        for row, pair in zip(E.rows(name), pairs):
            assert {"state": row["state"], "questions": row["questions"]} == D.request(pair, name.split(".")[1])
            assert row["label"] == pair["label"] and "label" not in json.dumps(row["questions"])


def test_guard_g3_sends_the_question_without_label_or_source(monkeypatch, tmp_path):
    record = {"state": "Is my card blocked?", "_meta": {"id": "banking77/test/1"}, "questions": {
        "intent": {"type": "choice", "instructions": "Which intent?", "criteria": {"card_blocked": None, "other": "Anything else"}, "label": "card_blocked", "src": "banking77"},
        "urgent": {"type": "noul", "instructions": "Is it urgent?", "label": False, "src": "banking77"},
        "tone": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "upset"], "label": 1, "src": "banking77"}}}
    (tmp_path / "development.jsonl").write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(E, "SUITE", tmp_path)
    monkeypatch.setattr(E.D, "REHEARSAL", False)
    row, = E._g3()
    assert row["questions"] == {"intent": {"type": "choice", "instructions": "Which intent?", "criteria": {"card_blocked": None, "other": "Anything else"}},
                                "urgent": {"type": "noul", "instructions": "Is it urgent?"}, "tone": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "upset"]}}
    assert row["labels"] == {"intent": "card_blocked", "urgent": False, "tone": 1} and row["state"] == record["state"] and row["id"] == "banking77/test/1"


# ---------- Answers ----------

def test_what_is_kept_of_an_answer():
    row = {"id": "x", "label": 1, "state": {}, "questions": {"same": {"type": "noul", "instructions": "?"}}}
    answer = {"answers": {"same": {"noul": 0.73}}, "usage": {"input_tokens": 12}, "_latency": 0.5, "_cached": False}
    assert E.record(row, answer) == {"id": "x", "label": 1, "p": 0.73, "tokens": 12, "latency": 0.5, "from": "fresh"}
    assert E.record(row, {"_error": "boom"}) == {"id": "x", "label": 1, "error": "boom"}
    assert E.record(row, answer | {"_cached": True, "_replayed_from": "p7"})["from"] == "p7"
    row = {"id": "r", "state": {}, "questions": {"c": {"type": "choice", "instructions": "?"}, "s": {"type": "score", "instructions": "?"}}}
    answer = {"answers": {"c": {"choice": "b", "probabilities": {"a": 0.2, "b": 0.8}, "confidence": 0.6},
                          "s": {"score": 1.4, "probabilities": {"0": 0.1, "1": 0.45, "2": 0.45}}}, "usage": {"input_tokens": 3}, "_latency": 0.1, "_cached": True}
    got = E.record(row, answer)["answers"]
    assert got["c"]["choice"] == "b" and got["s"]["level"] == 1  # most probable level; a tie goes to the lower level


# ---------- Replay of the released checkpoint's earlier answers ----------

def test_question_names_are_aliased_only_for_single_question_requests():
    q = {"type": "noul", "instructions": "?"}
    assert [f for f, _ in E.forms({"same": q})] == [{"same": q}, {"q": q}]
    assert E.forms({"q": q})[1] == ({"same": q}, {"same": "q"})
    assert len(E.forms({"a": q, "b": q})) == 1


def test_replay_copies_earlier_answers_under_this_phases_key_and_renames_them_back(tmp_path):
    ident, state, q = "jaredpalmer/kev-4b@v1.0", {"record_a": {"t": "x"}, "record_b": {"t": "y"}}, {"type": "noul", "instructions": "same?"}
    source = tmp_path / "p7.sqlite"
    con = sqlite3.connect(source)
    con.execute("CREATE TABLE answers (key TEXT PRIMARY KEY, response TEXT NOT NULL)")
    con.execute("INSERT INTO answers VALUES (?, ?)", (cache_key("p7:" + ident, state, {"q": q}), json.dumps({"answers": {"q": {"noul": 0.9}}, "usage": {"input_tokens": 5, "cost": 0}, "_latency": 1.0})))
    con.commit()
    con.close()
    target = tmp_path / "cache.sqlite"
    items = [(state, {"same": q}), ({"record_a": {"t": "other"}}, {"same": q})]
    assert E.replay("kev-4b", ident, items, target, [("p7", source, ("p7:",))]) == {"p7": 1}
    got = json.loads(sqlite3.connect(target).execute("SELECT response FROM answers WHERE key = ?", (cache_key(E.NAMESPACE + ident, state, {"same": q}),)).fetchone()[0])
    assert got["answers"] == {"same": {"noul": 0.9}} and got["_replayed_from"] == "p7"
    assert E.replay("kev-4b", ident, items, target, [("p7", source, ("p7:",))]) == {}  # already there: left alone
    assert sqlite3.connect(source).execute("SELECT count(*) FROM answers").fetchone()[0] == 1  # the source is only read


# ---------- The server ----------

def test_server_environment_is_the_one_open_models_sh_sets():
    text = (ROOT / "scripts" / "open_models.sh").read_text()
    exported = dict(re.findall(r"(\w+)=(\"?[^\s\"]*\"?)", " ".join(line.removeprefix("export ") for line in text.splitlines() if line.startswith("export "))))
    for name, value in exported.items():
        if name.startswith("LAYA") or name.startswith("TORCHINDUCTOR"):
            continue  # Laya's own caches
        assert E.SERVE_ENV[name] == value.strip('"').replace("$DIR", str(E.OPEN_MODELS_DIR)), name
    assert E.SERVE_ENV["HF_HUB_OFFLINE"] == "1" and "HF_HUB_OFFLINE=1" in text
    env = E.serve_env()
    assert not any(k in env for k in E.FACTORY)
    assert E.SANDBOX in text


def test_the_server_must_answer_as_the_checkpoint_asked_for():
    card = {"run": E.run_name("kev-4b", "loo-abt-buy"), "temperature": 1.0, "prefix_cache": {"size": 4}}
    E.check_card(card, "kev-4b", "loo-abt-buy")
    for bad in ({"run": "runs/smoke"}, {"temperature": 1.31}, {"prefix_cache": {"size": 600}}):
        with pytest.raises(E.ServerFailed):
            E.check_card(card | bad, "kev-4b", "loo-abt-buy")
    E.check_card({"run": "jaredpalmer/kev-4b@v1.0", "temperature": 1.31, "prefix_cache": {"size": 4}}, "kev-4b", "released")  # the released one keeps its own


def test_a_retrained_checkpoint_gets_its_own_cache_name(monkeypatch, tmp_path):
    monkeypatch.setattr(E, "RUNS", tmp_path)
    arm = tmp_path / "kev-0.8b" / "all"
    arm.mkdir(parents=True)
    (arm / "head.pt").write_bytes(b"head")
    (arm / "adapter_model.safetensors").write_bytes(b"weights 1")
    first = E.cache_id("kev-0.8b", "all")
    (arm / "adapter_model.safetensors").write_bytes(b"weights 2")
    assert first.startswith("p10/kev-0.8b/all@") and E.cache_id("kev-0.8b", "all") != first
    assert E.cache_id("kev-0.8b", "released") == "jaredpalmer/kev-0.8b@v1.0"


# ---------- When a checkpoint counts ----------

@pytest.fixture
def trained(monkeypatch, tmp_path):
    monkeypatch.setattr(E, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(E.D, "OUT", tmp_path / "out")
    monkeypatch.setattr(E, "REPLAY", 2)
    (tmp_path / "out" / "train").mkdir(parents=True)
    (tmp_path / "out" / "train" / "all.jsonl").write_text("{}\n{}\n{}\n")
    path = tmp_path / "runs" / "kev-4b" / "all"
    path.mkdir(parents=True)
    for name in ("head.pt", "adapter_model.safetensors"):
        (path / name).write_bytes(b"x")
    (path / "training_metrics.json").write_text(json.dumps({"requested_records": 5, "truncated_records": 0, "rejected_records": 0}))
    log = path.parent / "all.log"
    log.write_text(f"start\nep0 step 1/1 loss 0.1\nsaved {path}\n2026-10-15T00:00:00Z  end kev-4b all\n")
    return path, log


def test_a_complete_checkpoint_counts(trained):
    assert E.not_trained("kev-4b", "all") == ""


@pytest.mark.parametrize("damage", ["no end line", "dropped", "count", "truncated", "file"])
def test_an_incomplete_checkpoint_does_not_count(trained, damage):
    path, log = trained
    if damage == "no end line":
        log.write_text(log.read_text().replace("end kev-4b all", ""))
    elif damage == "dropped":
        log.write_text("dropped 3 of 5 records that exceed the training context\n" + log.read_text())
    elif damage == "count":
        (path / "training_metrics.json").write_text(json.dumps({"requested_records": 4, "truncated_records": 0, "rejected_records": 0}))
    elif damage == "truncated":
        (path / "training_metrics.json").write_text(json.dumps({"requested_records": 5, "truncated_records": 1, "rejected_records": 0}))
    else:
        (path / "head.pt").unlink()
    assert E.not_trained("kev-4b", "all")


def test_the_frozen_checksums_cover_every_built_file_and_input():
    built = {f"train/{a}.jsonl" for a in D.ARMS} | {f"pool/{d}.jsonl" for d in D.EM}
    assert set(E.BUILT) == built == set(E.BUILT_REHEARSAL)
    assert all(n == (6000 if "loo" in k else 8000 if "all" in k else 200) for k, (_, n) in E.BUILT.items())
    assert {f"experiments/entity_match/data/{d}/train.txt" for d in D.EM} <= set(E.INPUTS) and any("development" in k for k in E.INPUTS)


# ---------- Temperature, metrics, lines ----------

def test_temperature_recovers_a_known_one_and_never_changes_a_decision():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 3, 20000)
    y = (rng.random(z.size) < 1 / (1 + np.exp(-z / 2.5))).astype(int)
    p1 = 1 / (1 + np.exp(-z))
    T = R.fit_temperature(p1, y)
    assert abs(T - 2.5) < 0.15 and T in R.GRID
    pt = R.scale(p1, T)
    assert np.array_equal(pt >= 0.5, p1 >= 0.5) and np.isnan(R.scale(np.array([np.nan]), 2.0)[0])
    assert np.allclose(R.scale(np.array([0.2, 0.7]), 1.0), [0.2, 0.7])
    assert R.fit_temperature(np.array([0.9, 0.1, np.nan]), np.array([1, 0, 1])) == 0.25  # perfectly separated: the grid's end, not T -> 0; the unanswered row is left out
    assert R.fit_temperature(np.array([0.9, 0.1]), np.array([0, 1])) == 4.0
    assert len(R.GRID) == 121 and np.isclose(R.GRID[0], 0.25) and np.isclose(R.GRID[-1], 4) and np.allclose(np.diff(np.log(R.GRID)), np.log(16) / 120)  # Kev's TEMPERATURE_FIT


def test_an_unanswered_request_is_a_wrong_decision_with_probability_one_half():
    p, y = np.array([0.9, 0.2, np.nan, np.nan]), np.array([1, 0, 1, 0])
    s = R.stats(p, y)
    assert s["acc"] == 0.5 and np.isclose(s["brier"], (0.01 + 0.04 + 0.25 + 0.25) / 4)
    assert np.isclose(s["f1"], 2 * 1 / (2 * 1 + 1 + 1))  # tp 1, fp 1 (the unanswered non-match), fn 1 (the unanswered match)
    assert np.isclose(R.auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])), 0.75)


def test_bootstrap_draws_are_paired_and_vectorised_like_a_loop():
    rng = np.random.default_rng(3)
    y = (rng.random(300) < 0.3).astype(int)
    p = np.clip(y * 0.6 + rng.random(300) * 0.5, 0, 1)
    idx = R.draws(300)[:5]
    many = R.stats(p, y, idx)
    for i in range(5):
        one = R.stats(p[idx[i]], y[idx[i]])
        assert all(np.isclose(many[k][i], one[k]) for k in one)
    d = R.paired(p, p, y)
    assert all(v == (0.0, 0.0, 0.0) for v in d.values())


def test_line_t_uses_point_estimates_and_reports_intervals_that_cross_the_line():
    ok = {"acc": (-0.019, -0.03, -0.01), "auc": (-0.01, -0.02, 0.0), "brier": (0.019, 0.01, 0.03)}
    on_par, crit, separable = R.judge_t(ok, 0.0)
    assert on_par and all(crit.values()) and separable == {"acc": False, "auc": False, "brier": False}
    assert not R.judge_t(ok | {"auc": (-0.021, -0.03, -0.015)}, 0.0)[0]
    assert not R.judge_t(ok, 0.051)[0]  # rule 6


def test_guards_score_each_question_type_and_weight_g1_by_labelled_review():
    g3 = [{"labels": {"a": True, "b": "x", "c": 2}, "types": {"a": "noul", "b": "choice", "c": "score"},
           "answers": {"a": {"p": 0.7}, "b": {"choice": "y"}, "c": {"level": 2}}}, {"labels": {"a": False}, "types": {"a": "noul"}, "error": "e"}]
    c, w = R.guard_units("g3", g3)
    assert list(c) == [2, 0] and list(w) == [3, 1]
    c, w = R.guard_units("g1", [{"p": 0.8, "labels": [1, 1, 0]}, {"error": "e", "labels": [1]}, {"p": 0.1, "labels": []}])
    assert list(c) == [2, 0, 0] and list(w) == [3, 1, 0]


def test_end_to_end_keeps_true_matches_outside_the_candidates_and_at_most_one_per_record():
    rows = [{"lid": "L1", "rid": "R1"}, {"lid": "L1", "rid": "R2"}, {"lid": "L2", "rid": "R3"}]
    truth = {("L1", "R1"), ("L2", "R9")}
    p = np.array([0.7, 0.9, 0.6])
    assert R.e2e_tally(rows, p, truth, ["L1", "L2"]).tolist() == [[1, 1, 1], [0, 1, 1]]
    assert R.e2e_tally(rows, p, truth, ["L1", "L2"], one=True).tolist() == [[0, 1, 1], [0, 1, 1]]


def test_option_order_agreement_compares_the_chosen_record_not_the_position():
    truth = {("L1", "A")}
    rank = [{"lid": "L1", "rids": ["A", "B"], "answers": {"match": {"choice": "c1"}}}]
    reversed_ = [{"lid": "L1", "rids": ["B", "A"], "answers": {"match": {"choice": "c2"}}}]
    out = R.order_reading({"rank": rank, "reversed": reversed_}, truth)
    assert out["agreement"] == {"rank/reversed": 1.0} and out["rank"]["accuracy"] == 1.0 and out["reversed"]["chosen_first_3"] == 1.0
