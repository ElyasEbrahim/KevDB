"""Phase 16 in DuckDB: Jev's answers come from the SQL function, replayed from phase 2's cache; kevdb.allocation picks the rows to
label and turns their labels into shares with 95% intervals. Calls no model and spends nothing: the backend can only read
the cache, so a request it would have to send fails instead of being sent. The dataset's human labels stand in for the
person who would label the chosen rows; they are read for the chosen rows only (and, at the end, to score the result).

    python experiments/database/p16_demo.py      out/database/p16/demo.txt
"""

from __future__ import annotations

import sys

import duckdb
import httpx
import numpy as np

from kevdb import allocation, register
from kevdb.backends import JEV_MODEL
from kevdb.client import JevClient
from kevdb.engine import Backend
from paths import DATA, OUT, ROOT

QUESTION = "Is the movie review in `review` positive?"  # phase 2's question, word for word, so that every answer is a cache hit
MOVIE = DATA / "sembench/files/movie/data/sf_16000"
lines: list[str] = []


def say(text: str = "") -> None:
    lines.append(text)
    print(text, flush=True)


def offline_jev() -> Backend:
    """Jev's backend with phase 2's cache namespace and no way out: a cache miss returns NULL and is counted as an error."""
    refuse = httpx.MockTransport(lambda request: httpx.Response(599, text="offline demo: not in the cache"))
    return Backend("jev", lambda: JevClient(api_key="offline", provider="typesafe", model=JEV_MODEL, transport=refuse),
                   cache_path=ROOT / ".jevcache.sqlite", namespace="p2:")


def tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""CREATE TABLE reviews AS SELECT DISTINCT id AS movie, reviewId, id || '|' || reviewId AS key, reviewText,
                    (scoreSentiment = 'POSITIVE')::INT AS human FROM read_csv('{MOVIE}/Reviews.csv')""")
    con.execute(f"""CREATE TABLE movies AS SELECT id AS movie, list(DISTINCT trim(g) ORDER BY trim(g)) AS genres,
                    any_value(CASE WHEN boxOffice LIKE '%M' THEN TRY_CAST(replace(replace(boxOffice, '$', ''), 'M', '') AS DOUBLE) * 1e6
                                   WHEN boxOffice LIKE '%K' THEN TRY_CAST(replace(replace(boxOffice, '$', ''), 'K', '') AS DOUBLE) * 1e3
                              END) AS box_office
                    FROM (SELECT id, boxOffice, unnest(string_split(genre, ',')) AS g FROM read_csv('{MOVIE}/Movies.csv')) GROUP BY id""")


def score(con, plan: allocation.Plan, truth_sql: str, cheap_sql: str, title: str) -> None:
    """Label the chosen rows, estimate, and compare with the truth and with counting Jev's answers."""
    chosen = plan.rows().column("key").to_pylist()
    con.execute("CREATE OR REPLACE TEMP TABLE chosen AS SELECT unnest($keys) AS key", {"keys": chosen})
    labels = dict(con.execute("SELECT key, human FROM reviews JOIN chosen USING (key)").fetchall())  # the "person" labels these
    est = plan.estimate(labels)
    truth, cheap = dict(con.execute(truth_sql).fetchall()), dict(con.execute(cheap_sql).fetchall())
    names = est.column("output").to_pylist()
    t = np.array([truth[n] for n in names])
    c = np.array([cheap[n] for n in names])
    e, lo, hi = (est.column(k).to_numpy() for k in ("estimate", "low", "high"))
    say(f"{title}: {len(chosen)} labels for {len(names)} output(s)")
    say(f"  mean |error|, points: labels + Jev {100 * np.abs(e - t).mean():5.1f}   Jev counted alone {100 * np.abs(c - t).mean():5.1f}")
    say(f"  outputs off by more than 10 points: {np.mean(np.abs(e - t) > 0.10):.0%} against {np.mean(np.abs(c - t) > 0.10):.0%}; "
        f"95% intervals covering the truth: {np.mean((lo <= t + 1e-12) & (t <= hi + 1e-12)):.0%}; median half-width "
        f"{100 * np.median((hi - lo) / 2):.1f} points")


def main() -> None:
    con = duckdb.connect()
    jev = offline_jev()
    register(con, "jev", jev)
    tables(con)
    con.execute("CREATE TABLE scored AS SELECT *, jev_noul(json_object('review', reviewText), $q) AS p FROM reviews", {"q": QUESTION})
    meter = jev.reset()
    if meter.errors or meter.fresh:
        sys.exit(f"{meter.errors} answers were not in the cache ({meter.fresh} sent): run phase 2 first (experiments/database/README.md)")
    rows = con.execute("SELECT count(*) FROM scored").fetchall()[0][0]
    say(f"Jev's answers on {rows:,} reviews from the cache: {meter.calls:,} distinct requests, none sent, $0")
    say()
    plan = allocation.plan(con, "scored", key="key", p="p", group="movie", budget=2000, seed=1)
    score(con, plan, "SELECT movie, avg(human) FROM reviews GROUP BY movie",
          "SELECT movie, avg((p >= 0.5)::INT) FROM scored GROUP BY movie", "Share of positive reviews per movie")
    con.execute("CREATE VIEW by_genre AS SELECT s.key, s.p, m.genres FROM scored s JOIN movies m USING (movie)")
    plan = allocation.plan(con, "by_genre", key="key", p="p", group="genres", budget=2000, seed=1)
    score(con, plan, "SELECT g, avg(human) FROM (SELECT unnest(m.genres) AS g, r.human FROM reviews r JOIN movies m USING (movie)) GROUP BY g",
          "SELECT g, avg((p >= 0.5)::INT) FROM (SELECT unnest(genres) AS g, p FROM by_genre) GROUP BY g",
          "Share per genre, a review counting in each genre of its movie")
    con.execute("""CREATE VIEW weighted AS SELECT s.key, s.p, m.box_office / count(*) OVER (PARTITION BY s.movie) AS w
                   FROM scored s JOIN movies m USING (movie) WHERE m.box_office IS NOT NULL""")
    plan = allocation.plan(con, "weighted", key="key", p="p", weight="w", budget=500, seed=1)
    score(con, plan, """SELECT 'all', sum(m.box_office * share) / sum(m.box_office) FROM (SELECT movie, avg(human) AS share FROM reviews
                        GROUP BY movie) JOIN movies m USING (movie) WHERE m.box_office IS NOT NULL""",
          "SELECT 'all', sum(w * (p >= 0.5)::INT) / sum(w) FROM weighted", "Box-office-weighted mean of the movie shares")
    jev.close()
    OUT.joinpath("p16").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("p16", "demo.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
