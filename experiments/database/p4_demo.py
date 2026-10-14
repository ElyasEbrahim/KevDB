"""Phase 4 SQL demo: one query with three semantic predicates, run cold in DuckDB in two ways.

    one by one   WHERE jev_noul(..) AND jev_noul(..) AND jev_noul(..)   one request per predicate; DuckDB short-circuits by itself
    packed       jev_nouls(.., [three questions])                         one request per row

Query: how many sentences criticise the staff and mention the food, but say nothing positive about the food?

    python experiments/database/p4_demo.py
"""

from __future__ import annotations

import json
import re
import time

import duckdb
import pyarrow as pa
from kevdb.cli import load_dotenv

from kevdb import jev_backend, register
from p4_collect import PREDICATES, sentences
from paths import OUT as RESULTS

OUT = RESULTS / "p4"
A, B, C = PREDICATES["negative:staff"], PREDICATES["mention:food"], PREDICATES["positive:food"]
S = "json_object('sentence', text)"
QUERIES = {
    "one by one": f"SELECT count(*) FROM mams WHERE jev_noul({S}, $a) >= 0.5 AND jev_noul({S}, $b) >= 0.5 AND jev_noul({S}, $c) < 0.5",
    "packed": f"SELECT count(*) FROM (SELECT jev_nouls({S}, $abc) AS p FROM mams) WHERE p[1] >= 0.5 AND p[2] >= 0.5 AND p[3] < 0.5",
}


def main() -> None:
    load_dotenv()
    rows = sentences()
    con = duckdb.connect()
    con.execute("SET threads = 1")
    con.register("mams_arrow", pa.table({"sid": [r["sid"] for r in rows], "text": [r["text"] for r in rows]}))
    con.execute("CREATE TABLE mams AS SELECT * FROM mams_arrow")
    truth = sum(r["truth"]["negative:staff"] and r["truth"]["mention:food"] and not r["truth"]["positive:food"] for r in rows)
    jev = jev_backend(concurrency=16)
    register(con, "jev", jev)
    results = {"rows": len(rows), "human_count": truth}
    for name, sql in QUERIES.items():
        jev.namespace = f"p4.demo.{name}.{time.time():.0f}:"
        jev.reset()
        used = set(re.findall(r"\$(\w+)", sql))
        params = {k: v for k, v in {"a": A, "b": B, "c": C, "abc": json.dumps([A, B, C])}.items() if k in used}
        started = time.perf_counter()
        (count,) = con.execute(sql, params).fetchone()
        seconds = time.perf_counter() - started
        m = jev.meter
        results[name] = {"count": count, "seconds": round(seconds, 1), "requests": m.calls, "errors": m.errors, "input_tokens": m.input_tokens, "cost_usd": round(m.cost, 4)}
        print(f"{name}: {count} sentences, {seconds:.1f} s, {m.calls} requests, {m.input_tokens:,} input tokens, ${m.cost:.4f}, {m.errors} failed")
    print(f"answer under the human labels: {truth} sentences (of {len(rows)})")
    (OUT / "demo.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n")
    jev.close()


if __name__ == "__main__":
    main()
