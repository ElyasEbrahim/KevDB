"""Phase 8 SQL demo: a semantic join over a whole table in one DuckDB statement, one request per left record (jev_pick / jev_match, see src/kevdb/sql.py).

    python experiments/database/p8_demo.py replay   # replays phase 8's cache at no cost: the matches from SQL must be identical to those of the collection script
    python experiments/database/p8_demo.py fresh    # asks the abt-buy multiple-choice join afresh to measure end-to-end time and cost (about $0.05)

What experiments/database/protocol/p8.txt fixed beforehand is the multiple choice on abt-buy (the form with the highest full-table F1 among those that held); replaying jev_match on both datasets was added afterwards.
Candidates are the top 10 from experiments/table_match/, in the same order inside the request as during collection (shuffled by hash).
"""

from __future__ import annotations

import json
import sys
import time

import duckdb
import pyarrow as pa
from kevdb.cli import load_dotenv

import p8_join as J
from kevdb import jev_backend, register

BLOCK = """
    WITH block AS (
        SELECT c.lid, any_value(l.record) AS a, list(c.rid ORDER BY c.pos) AS rids, '[' || string_agg(r.record, ',' ORDER BY c.pos) || ']' AS candidates
        FROM candidates c JOIN left_t l USING (lid) JOIN right_t r USING (rid)
        GROUP BY c.lid
    )"""
SQL = {
    # Multiple choice: a left record matches at most one record
    "C": BLOCK + """, picked AS (SELECT lid, rids, jev_pick(a, candidates, $question, $none) AS pick FROM block)
    SELECT lid, rids[pick.pos] AS rid FROM picked WHERE pick.pos > 0""",
    # One yes/no question per candidate: a left record can match several records
    "Q": BLOCK + """, judged AS (SELECT lid, unnest(rids) AS rid, unnest(jev_match(a, candidates, $question, $criteria)) AS p FROM block)
    SELECT lid, rid FROM judged WHERE p >= 0.5""",
}
SCORE = """
    SELECT count(*) FILTER (WHERE t.lid IS NOT NULL) AS hit, count(*) AS found, (SELECT count(*) FROM truth) AS wanted
    FROM matches m LEFT JOIN truth t USING (lid, rid)"""


def tables(con: duckdb.DuckDBPyConnection, dataset: str) -> None:
    left, right, truth, cands = J.load(dataset)
    rows = [(lid, rid, pos) for lid in left for pos, rid in enumerate(J.arrange(dataset, lid, cands[lid][:J.K], "random"))]
    dump = lambda r: json.dumps(r, ensure_ascii=False)
    for name, columns in {
        "left_t": {"lid": list(left), "record": [dump(r) for r in left.values()]},
        "right_t": {"rid": list(right), "record": [dump(r) for r in right.values()]},
        "candidates": {"lid": [r[0] for r in rows], "rid": [r[1] for r in rows], "pos": [r[2] for r in rows]},
        "truth": {"lid": [a for a, _ in truth], "rid": [b for _, b in truth]},
    }.items():
        con.register("incoming", pa.table(columns))
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM incoming")
        con.unregister("incoming")


def join(con: duckdb.DuckDBPyConnection, jev, dataset: str, form: str) -> dict:
    kind = J.tables.DATASETS[dataset]["kind"]
    q = J.tables.KINDS[kind]
    params = {"question": J.CHOICE[kind], "none": J.NONE[kind]} if form == "C" else {"question": q.instructions, "criteria": json.dumps(q.criteria)}
    jev.reset()
    started = time.perf_counter()
    con.execute("CREATE OR REPLACE TABLE matches AS " + SQL[form], params)
    seconds = time.perf_counter() - started
    hit, found, wanted = con.execute(SCORE).fetchone()
    precision, recall = hit / max(found, 1), hit / wanted
    m = jev.meter
    return {"dataset": dataset, "form": form, "matches": found, "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "seconds": round(seconds, 1), "requests": m.calls, "fresh": m.fresh, "errors": m.errors, "input_tokens": m.input_tokens, "cost_usd": round(m.cost, 4)}


def main(mode: str) -> None:
    load_dotenv()
    con = duckdb.connect()
    jev = jev_backend(concurrency=16, cache_path=J.OUT / "cache.jev.sqlite")
    register(con, "jev", jev)
    out = []
    for dataset, form in ([("abt-buy", "C"), ("abt-buy", "Q"), ("dblp-scholar", "Q")] if mode == "replay" else [("abt-buy", "C")]):
        tables(con, dataset)
        jev.namespace = J.NAMESPACE if mode == "replay" else f"p8.demo.{time.time():.0f}:"
        r = join(con, jev, dataset, form)
        got = set(con.execute("SELECT lid, rid FROM matches").fetchall())
        script = {pair for pair, (yes, _) in J.judgments(J.read("jev", dataset, form, "random", "full")[0], form).items() if yes}
        r |= {"mode": mode, "same_as_script": got == script, "only_in_sql": len(got - script), "only_in_script": len(script - got)}
        out.append(r)
        print(f"{mode:<7}{dataset:<13}{form}  {r['matches']:,} matches, precision {r['precision']:.1%}, recall {r['recall']:.1%}, F1 {r['f1']:.1%}; {r['seconds']} s, "
              f"{r['requests']:,} requests ({r['fresh']:,} fresh, {r['errors']} failed), {r['input_tokens']:,} input tokens, ${r['cost_usd']:.4f}; "
              + ("matches identical to the collection script" if r["same_as_script"] else f"differs from the collection script: {r['only_in_sql']} pairs only in SQL, {r['only_in_script']} only in the script"))
    jev.close()
    (J.OUT / f"demo.{mode}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] in (["replay"], ["fresh"]):
        main(sys.argv[1])
    else:
        sys.exit(__doc__)
