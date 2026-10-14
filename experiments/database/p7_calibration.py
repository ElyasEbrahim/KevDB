"""Phase 7, part C: fit calibration parameters for semantic predicates on a few human labels, store them in the database and apply them in SQL. Criteria: experiments/database/protocol/p7.txt.

Everything goes through SQL: the model's judgements come from kev_noulc(...), the parameters live in the table semantic_calibration, the calibrated probabilities come from kev_noulc_cal(...)
(src/kevdb/calibration.py). The database file is out/database/p7/<model>/calibration.duckdb, which the analysis script reads directly.

    python experiments/database/p7_calibration.py run        # four record-matching datasets: label pool → fit and store → test split. The local model's server must be running
    python experiments/database/p7_calibration.py replay     # the movie and restaurant reviews collected in phase 6, replayed from the cache without asking the model: one parameter set per predicate
    python experiments/database/p7_calibration.py check      # are all tables there, how many rows have no answer? Counting only

run does three steps per dataset:
1. ask the model about the label pool (the fixed 1,000 validation pairs);
2. fit this question's parameters on this table from the human labels of the pool's first 200 pairs and store them (scope is the dataset name);
3. one query over the test split returns the raw probability, the one calibrated with this table's parameters and the one calibrated with walmart-amazon's (can a question's parameters transfer across tables?).
Fitting uses human labels only and involves no output of Jev.
"""

from __future__ import annotations

import json
import shutil
import sys
import time

import duckdb
import numpy as np

from kevdb import calibration, open_backend, register
from p7_data import EM, LABELS, MODEL, MOVIE_QUESTIONS, NAMESPACE, OUT as ROOT_OUT, REHEARSAL, em_question, em_rows, made_reviews, made_sentences
from paths import OUT as RESULTS

OUT = ROOT_OUT / "calibration"
DB = OUT / "calibration.duckdb"
CACHE = ROOT_OUT / "cache.sqlite"
TRANSFER = "walmart-amazon"  # the transferred parameters come from this table: phase 6 found the scale problem on it
# This model's judgements collected in phase 6 (only for models tested in phase 6)
STAGE6 = RESULTS / "p6" / f"cache.{MODEL}.sqlite"
# Replay part: the first REPLAY_FIT rows of each dataset (fixed hash order) are used for fitting, the rest for evaluation
REPLAY_FIT = 10 if REHEARSAL else 200


def table(dataset: str) -> str:
    return dataset.replace("-", "_")


def connect(backend) -> duckdb.DuckDBPyConnection:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB))
    con.execute("SET threads = 1")
    register(con, "kev", backend)
    calibration.install(con, "kev", backend.model)
    return con


def fetch(con, sql: str, params: list | None = None) -> dict[str, np.ndarray]:
    """Query result as one array per column; probabilities without an answer (NULL) become NaN."""
    df = con.execute(sql, params or []).fetchdf()
    return {c: df[c].to_numpy() for c in df.columns}


def judged(con, backend, sql: str, params: list | dict) -> dict:
    """Run a CREATE statement that asks the model. Failed requests are not cached, so re-running the same statement only re-asks the failures; at most 4 rounds."""
    backend.reset()
    started = time.perf_counter()
    for attempt in range(1, 5):
        con.execute(sql, params)
        if not backend.meter.errors:
            break
    m = backend.meter
    return {"requests": m.calls, "fresh": m.fresh, "errors": m.errors, "attempts": attempt, "seconds": round(time.perf_counter() - started, 1), "input_tokens": m.input_tokens}


def run() -> None:
    backend = open_backend(MODEL, concurrency=16, cache_path=CACHE, namespace=NAMESPACE)
    con = connect(backend)
    log = {"model": backend.model, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "datasets": {}}
    # walmart-amazon first: the other tables' test queries need its parameters
    for dataset in (TRANSFER,) + tuple(d for d in EM if d != TRANSFER):
        t, (instructions, criteria) = table(dataset), em_question(dataset)
        for part in ("pool", "test"):
            rows = em_rows(dataset, part)
            con.execute(f"CREATE OR REPLACE TABLE {part}_{t} (pos INTEGER, id VARCHAR, label INTEGER, state VARCHAR)")
            con.executemany(f"INSERT INTO {part}_{t} VALUES (?, ?, ?, ?)", [(i, r["id"], r["label"], r["state"]) for i, r in enumerate(rows)])
        steps = {"pool": judged(con, backend, f"CREATE OR REPLACE TABLE judged_pool_{t} AS SELECT pos, id, label, kev_noulc(state, ?, ?) AS p FROM pool_{t}", [instructions, criteria])}
        sample = fetch(con, f"SELECT p, label FROM judged_pool_{t} WHERE pos < {LABELS} ORDER BY pos")
        a, b = calibration.fit(con, backend.model, instructions, criteria, dataset, sample["p"], sample["label"])
        steps["fit"] = {"a": a, "b": b, "labels": int(len(sample["label"])), "positives": int(sample["label"].sum())}
        steps["test"] = judged(con, backend, f"""CREATE OR REPLACE TABLE judged_test_{t} AS SELECT pos, id, label,
            kev_noulc(state, $q, $c) AS p_raw, kev_noulc_cal(state, $q, $c, '{dataset}') AS p_cal, kev_noulc_cal(state, $q, $c, '{TRANSFER}') AS p_transfer
            FROM test_{t}""", {"q": instructions, "c": criteria})
        log["datasets"][dataset] = steps
        print(f"{dataset}: pool {steps['pool']['requests']} requests ({steps['pool']['errors']} failed), test {steps['test']['requests']} ({steps['test']['errors']} failed), {steps['pool']['seconds'] + steps['test']['seconds']:.0f} s", flush=True)
    log["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "run.json").write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n")
    con.close()
    backend.close()


def replay_rows() -> tuple[list[tuple], list[tuple]]:
    """(movie rows, restaurant rows), each row: (rank, id, text, predicate name, question, label). Movies have the single predicate "positive?", with one row per label of a text;
    restaurant reviews have 21 rows per sentence. rank is the text's / sentence's position in phase 6's fixed hash order; the REPLAY_FIT lowest ranks are used for fitting."""
    from p4_collect import PREDICATES

    positive = MOVIE_QUESTIONS["positive"]["instructions"]
    if REHEARSAL:
        movie = [(i, r["id"], r["state"]["review"], "positive", positive, r["label"]) for i, r in enumerate(made_reviews("replay", 40))]
        mams = [(i, r["id"], r["state"]["sentence"], name, q, int((i + j) % 3 == 0)) for i, r in enumerate(made_sentences("replay", 30)) for j, (name, q) in enumerate(PREDICATES.items())]
        return movie, mams
    from p2_collect import sha, texts
    from p3_aggregates import load
    from p4_collect import sentences
    from p6_collect import N_MAMS, N_MOVIE

    order = {sha(t): (i, t) for i, t in enumerate(texts(N_MOVIE, "open"))}
    labelled = load()
    movie = [(order[h][0], h, order[h][1], "positive", positive, int(y)) for h, y in zip(labelled["text"], labelled["y"]) if h in order]
    mams = [(i, r["sid"], r["text"], name, q, int(r["truth"][name])) for i, r in enumerate(sentences(N_MAMS)) for name, q in PREDICATES.items()]
    return movie, mams


def replay() -> None:
    """The Kev judgements collected in phase 6 are in its cache of the time (namespace p6:). Running the same states and questions through the SQL functions again hits the cache every time and sends no request."""
    OUT.mkdir(parents=True, exist_ok=True)
    cache, namespace = CACHE, NAMESPACE + "replay:"
    if not REHEARSAL:
        source, cache, namespace = STAGE6, OUT / "p6-cache.sqlite", "p6:"
        if not source.exists():
            print(f"no {source}: this model has no phase 6 judgements to replay, skipped")
            return
        shutil.copyfile(source, cache)  # replay on a copy, so phase 6's outputs stay untouched
    backend = open_backend(MODEL, concurrency=16, cache_path=cache, namespace=namespace)
    con = connect(backend)
    log, fits = {}, {}
    for scope, field, rows in zip(("movie", "mams"), ("review", "sentence"), replay_rows()):
        con.execute(f"CREATE OR REPLACE TABLE replay_{scope}_rows (pos INTEGER, id VARCHAR, text VARCHAR, predicate VARCHAR, question VARCHAR, label INTEGER)")
        con.executemany(f"INSERT INTO replay_{scope}_rows VALUES (?, ?, ?, ?, ?, ?)", rows)
        log[scope] = judged(con, backend, f"CREATE OR REPLACE TABLE replay_{scope}_raw AS SELECT pos, id, predicate, question, label, kev_noul(json_object('{field}', text), question) AS p FROM replay_{scope}_rows", [])
        if not REHEARSAL and (log[scope]["fresh"] or log[scope]["errors"]):
            sys.exit(f"a replay must not ask the model, but some requests missed the cache: {log}")
        # One parameter set per predicate (one for movies, 21 for restaurant reviews), each fitted on the human labels of the REPLAY_FIT lowest-ranked texts / sentences
        for (q,) in con.execute(f"SELECT DISTINCT question FROM replay_{scope}_raw ORDER BY 1").fetchall():
            sample = fetch(con, f"SELECT p, label FROM replay_{scope}_raw WHERE pos < {REPLAY_FIT} AND question = ? ORDER BY pos, id", [q])
            fits[f"{scope}:{calibration.predicate_key(q)}"] = calibration.fit(con, backend.model, q, "", scope, sample["p"], sample["label"])
        # Application: each predicate in one query gets its own parameters, which the database joins in by (model, predicate, scope)
        log[f"{scope}.apply"] = judged(con, backend, f"""CREATE OR REPLACE TABLE replay_{scope} AS SELECT pos, id, predicate, label,
            kev_noul(json_object('{field}', text), question) AS p_raw, kev_noul_cal(json_object('{field}', text), question, '{scope}') AS p_cal FROM replay_{scope}_rows""", [])
    log["predicates_fitted"] = len(fits)
    (OUT / "replay.json").write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n")
    print(f"replay: movies {log['movie']['requests']} requests, restaurants {log['mams']['requests']}, {sum(v['fresh'] for v in log.values() if isinstance(v, dict))} sent; parameters fitted for {len(fits)} predicates", flush=True)
    con.close()
    backend.close()


def check() -> None:
    if not DB.exists():
        sys.exit(f"missing: {DB}")
    con, missing = duckdb.connect(str(DB), read_only=True), []
    have = {name for (name,) in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    print(f"{'table':34s}{'rows':>8s}{'rows without answer':>20s}")
    wanted = [f"judged_{part}_{table(d)}" for d in EM for part in ("pool", "test")] + (["replay_movie", "replay_mams"] if REHEARSAL or STAGE6.exists() else [])
    for name in wanted:
        if name not in have:
            missing.append(name)
            continue
        column = "p" if "pool" in name else "p_raw"
        rows, empty = con.execute(f"SELECT count(*), count(*) FILTER ({column} IS NULL) FROM {name}").fetchone()
        print(f"{name:34s}{rows:>10d}{empty:>14d}")
    print(f"{'semantic_calibration':34s}{con.execute('SELECT count(*) FROM semantic_calibration').fetchone()[0]:>10d}")
    if missing:
        sys.exit("missing: " + "; ".join(missing))
    print("all tables complete")


if __name__ == "__main__":
    command = {"run": run, "replay": replay, "check": check}.get(sys.argv[1] if len(sys.argv) == 2 else "")
    if not command:
        sys.exit(__doc__)
    command()
