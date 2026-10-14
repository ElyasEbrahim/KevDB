"""Phase 1: run the 10 queries of SemBench's movie scenario as SQL in DuckDB, once per execution mode.

    jev      Jev only
    llm      LLM only (the route chosen in the qualification test)
    cascade  Jev first; rows it is unsure about go to the LLM: yes/no probability between 0.1 and 0.9, multiple-choice confidence below 0.9.
             The thresholds come from the earlier end-to-end matching experiment (experiments/table_match/) and were not tuned on this data. Score queries (Q9, Q10) are not cascaded.
    open     one open, Jev-compatible model only (added in phase 7; a local model needs its server started with scripts/open_models.sh).
             Questions and queries are identical to the jev mode; results go to the directory given by --out and do not enter the phase 1 report.

Every query runs cold in its own cache namespace, so time and cost are those of running that query alone; identical inputs within one query are asked once.
Scoring uses SemBench's own scoring code against the result of its reference SQL on the same data.

The questions are first drafts written from SemBench's natural-language queries and were not edited against the truth: Q1 asks for "clearly positive", Q2-Q4 for "positive".

Q5-Q7 pair reviews with each other. Besides asking the model about each pair literally (as every system in SemBench does),
a rewritten version Q5r-Q7r judges each review's sentiment once and then pairs them with an ordinary join.

    python experiments/database/p1_movie.py run jev --runs 3
    python experiments/database/p1_movie.py run llm --llm <route name>
    python experiments/database/p1_movie.py run cascade --llm <route name>
    python experiments/database/p1_movie.py run open --open kev-4b --out out/database/p7/sql --tag default --concurrency 16
    python experiments/database/p1_movie.py report
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import types
from pathlib import Path

import duckdb
import pandas as pd
from kevdb.cli import load_dotenv

from kevdb import Backend, jev_backend, llm_backend, open_backend, register, register_cascade
from paths import DATA, OUT as RESULTS

SEMBENCH = DATA / "sembench"
MOVIE = SEMBENCH / "files/movie"
SCALE = 2000
OUT = RESULTS / "p1"
TAKEN, ANT = "taken_3", "ant_man_and_the_wasp_quantumania"

PARAMS = {
    "clearly": "Is the movie review in `review` clearly positive?",
    "positive": "Is the movie review in `review` positive?",
    "same": "Do the two movie reviews `review_1` and `review_2` express the same sentiment, that is, both are positive or both are negative?",
    "opposite": "Do the two movie reviews `review_1` and `review_2` express opposite sentiments, that is, one is positive and the other is negative?",
    "sentiment": "What is the sentiment of the movie review in `review`?",
    "options": json.dumps({"POSITIVE": "The review is positive about the movie.", "NEGATIVE": "The review is negative about the movie."}),
    "liked": "How much did the reviewer like the movie, judging from the movie review in `review`?",
    # SemBench's scoring levels for the systems, from low to high
    "levels": json.dumps([
        "Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.",
        "Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.",
        "Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.",
        "Positive. Noticeably positive sentiment, indicating general satisfaction.",
        "Very positive. Strong positive sentiment, indicating high satisfaction.",
    ]),
}


def queries(p: str) -> dict[str, str]:
    """p is the function prefix: jev, llm, cas or open."""
    review = "json_object('review', reviewText)"
    pair = "json_object('review_1', a.reviewText, 'review_2', b.reviewText)"
    # This movie's 256 rows are really 128 reviews that each appear twice; deduplicate before pairing
    ant = f"WITH r AS MATERIALIZED (SELECT DISTINCT id, reviewId, reviewText FROM Reviews WHERE id = '{ANT}')"
    pairs = f"{ant} SELECT a.id, a.reviewId AS reviewId1, b.reviewId AS reviewId2 FROM r a JOIN r b ON a.reviewId < b.reviewId"
    judged = f"{ant}, s AS MATERIALIZED (SELECT *, {p}_noul({review}, $positive) >= 0.5 AS pos FROM r) " \
             "SELECT a.id, a.reviewId AS reviewId1, b.reviewId AS reviewId2 FROM s a JOIN s b ON a.reviewId < b.reviewId"
    q = {
        "Q1": f"SELECT reviewId FROM Reviews WHERE {p}_noul({review}, $clearly) >= 0.5 LIMIT 5",
        "Q2": f"SELECT reviewId FROM Reviews WHERE id = '{TAKEN}' AND {p}_noul({review}, $positive) >= 0.5 LIMIT 5",
        "Q3": f"SELECT count(*) AS positive_review_cnt FROM Reviews WHERE id = '{TAKEN}' AND {p}_noul({review}, $positive) >= 0.5",
        "Q4": f"SELECT avg(({p}_noul({review}, $positive) >= 0.5)::INT) AS positivity_ratio FROM Reviews WHERE id = '{TAKEN}'",
        "Q5": f"{pairs} WHERE {p}_noul({pair}, $same) >= 0.5 LIMIT 10",
        "Q6": f"{pairs} WHERE {p}_noul({pair}, $opposite) >= 0.5 LIMIT 10",
        "Q7": f"{pairs} WHERE {p}_noul({pair}, $opposite) >= 0.5",
        "Q8": f"SELECT sentiment AS scoreSentiment, count(*) AS count FROM (SELECT {p}_choice({review}, $sentiment, $options) AS sentiment "
              f"FROM Reviews WHERE id = '{TAKEN}') GROUP BY 1",
        "Q5r": f"{judged} WHERE a.pos = b.pos LIMIT 10",
        "Q6r": f"{judged} WHERE a.pos <> b.pos LIMIT 10",
        "Q7r": f"{judged} WHERE a.pos <> b.pos",
    }
    if p == "jev":
        # Variants added afterwards (after seeing Q2 at only 0.6): LIMIT queries take the k rows with the highest probability instead of the first k scanned.
        # Only backends that give probabilities can do this. The main results remain the version above, written to the original plan.
        top = "WHERE prob >= 0.5 ORDER BY prob DESC"
        scored = lambda instr: f"SELECT a.id, a.reviewId AS reviewId1, b.reviewId AS reviewId2, {p}_noul({pair}, ${instr}) AS prob FROM r a JOIN r b ON a.reviewId < b.reviewId"
        q["Q1p"] = f"SELECT reviewId FROM (SELECT DISTINCT reviewId, {p}_noul({review}, $clearly) AS prob FROM Reviews) {top}, reviewId LIMIT 5"
        q["Q2p"] = f"SELECT reviewId FROM (SELECT DISTINCT reviewId, {p}_noul({review}, $positive) AS prob FROM Reviews WHERE id = '{TAKEN}') {top}, reviewId LIMIT 5"
        q["Q5p"] = f"{ant} SELECT id, reviewId1, reviewId2 FROM ({scored('same')}) {top}, reviewId1, reviewId2 LIMIT 10"
        q["Q6p"] = f"{ant} SELECT id, reviewId1, reviewId2 FROM ({scored('opposite')}) {top}, reviewId1, reviewId2 LIMIT 10"
    if p != "cas":
        q["Q9"] = f"SELECT reviewId, {p}_score({review}, $liked, $levels) + 1 AS reviewScore FROM Reviews WHERE id = '{ANT}'"
        q["Q10"] = f"SELECT id AS movieId, avg({p}_score({review}, $liked, $levels) + 1) AS movieScore FROM Reviews GROUP BY id"
    return q


def connect(tables: Path | None = None) -> duckdb.DuckDBPyConnection:
    """tables: the directory holding Movies.csv and Reviews.csv; by default SemBench's 2,000-row version."""
    tables = tables or MOVIE / f"data/sf_{SCALE}"
    con = duckdb.connect()
    con.execute("SET threads = 1")  # fixed row order makes LIMIT queries reproducible; concurrency happens in the model backend
    for table in ("Movies", "Reviews"):
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM read_csv('{tables}/{table}.csv')")
    return con


def run_mode(mode: str, tag: str, backends: dict[str, Backend], only: list[str] | None = None,
             root: Path | None = None, tables: Path | None = None) -> None:
    """backends: {"jev"} for the jev mode, {"llm"} for llm, {"open"} for open, two for the cascade. Queries that already have results are skipped.
    root is the parent directory of the results (phase 1's by default); tables: see connect."""
    con = connect(tables)
    prefix = {"jev": "jev", "llm": "llm", "cascade": "cas", "open": "open"}[mode]
    if mode == "cascade":
        register_cascade(con, "cas", backends["jev"], backends["llm"])
    else:
        register(con, prefix, backends[mode])
    out = (root or OUT) / f"{mode}.{tag}"
    out.mkdir(parents=True, exist_ok=True)
    for name, sql in queries(prefix).items():
        if (only and name not in only) or (out / f"{name}.json").exists():
            continue
        for b in backends.values():
            b.namespace = f"p1.{mode}.{tag}.{name}:"
            b.reset()
        params = {k: v for k, v in PARAMS.items() if f"${k}" in sql}
        seconds, attempts = 0.0, 0
        while True:  # failed requests are not cached, so re-running the same query only re-asks the failures
            started = time.perf_counter()
            df = con.execute(sql, params).fetchdf()
            seconds += time.perf_counter() - started
            attempts += 1
            if not any(b.meter.errors for b in backends.values()) or attempts == 4:
                break
        df.to_csv(out / f"{name}.csv", index=False)
        record = {"query": name, "mode": mode, "tag": tag, "rows": len(df), "seconds": round(seconds, 2), "attempts": attempts,
                  "cost": sum(b.meter.cost for b in backends.values())}
        for role, b in backends.items():
            m = b.meter
            record[role] = {"model": b.model, "calls": m.calls, "fresh": m.fresh, "errors": m.errors, "input_tokens": m.input_tokens,
                            "output_tokens": m.output_tokens, "cost": m.cost, "latency_p50": m.latency(0.5)}
        (out / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
        calls = ", ".join(f"{role} {b.meter.calls} calls" for role, b in backends.items())
        print(f"{mode}.{tag} {name}: {len(df)} rows, {seconds:.1f} s, {calls}, ${record['cost']:.4f}", flush=True)
    con.close()


def verify(llm: str) -> None:
    """Replay every existing result and confirm none was computed with failed requests (rows that returned NULL).

    If everything hits the cache and matches what was stored, all is well. If requests have to be sent again, those rows had no answer at the time:
    they are re-asked, the complete result overwrites the old one, and the record notes how many rows were filled in (the time cannot be re-measured and is kept).
    """
    load_dotenv()
    caches = {"jev": ".jevcache.sqlite", "cascade": ".jevcache.sqlite", "llm": str(OUT / "llm-cache.sqlite")}
    for folder in sorted(p for p in OUT.iterdir() if p.is_dir()):
        mode, tag = folder.name.split(".")
        backends = {}
        if mode in ("jev", "cascade"):
            backends["jev"] = jev_backend(concurrency=16, cache_path=caches[mode])
        if mode in ("llm", "cascade"):
            backends["llm"] = llm_backend(llm, concurrency=32, cache_path=caches[mode])
        con = connect()
        prefix = {"jev": "jev", "llm": "llm", "cascade": "cas"}[mode]
        register_cascade(con, "cas", backends["jev"], backends["llm"]) if mode == "cascade" else register(con, prefix, backends[mode])
        for name, sql in queries(prefix).items():
            meta = folder / f"{name}.json"
            if not meta.exists():
                continue
            for b in backends.values():
                b.namespace = f"p1.{mode}.{tag}.{name}:"
                b.reset()
            params = {k: v for k, v in PARAMS.items() if f"${k}" in sql}
            for _ in range(4):
                df = con.execute(sql, params).fetchdf()
                if not any(b.meter.errors for b in backends.values()):
                    break
            fresh, errors = sum(b.meter.fresh for b in backends.values()), sum(b.meter.errors for b in backends.values())
            same = df.to_csv(index=False) == (folder / f"{name}.csv").read_text()
            if fresh or errors or not same:
                print(f"{folder.name} {name}: re-asked {fresh} rows, {errors} still failing, result {'unchanged' if same else 'changed'}", flush=True)
                df.to_csv(folder / f"{name}.csv", index=False)
                record = json.loads(meta.read_text())
                record.update(repaired=fresh, still_failed=errors, result_changed=not same)
                meta.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
        con.close()
        for b in backends.values():
            b.close()
        print(f"{folder.name}: checked", flush=True)


def evaluator():
    """SemBench's movie scorer. The cdlib it imports is only for clustering metrics, which the movie scenario never uses, so a stub stands in for it."""
    stub = types.ModuleType("cdlib")
    stub.NodeClustering, stub.evaluation = object, types.SimpleNamespace()
    sys.modules.setdefault("cdlib", stub)
    for path in (SEMBENCH / "src", SEMBENCH / "src/scenario/movie/evaluation"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from evaluate import MovieEvaluator

    return MovieEvaluator("movie", SCALE)


def ground_truth(ev, number: int) -> pd.DataFrame:
    """Same as the scorer itself, except that nothing is written into its directory."""
    con = duckdb.connect()
    con.register("Movies", ev.movies_df)
    con.register("Reviews", ev.reviews_df)
    return con.execute((MOVIE / f"query/gold_sql/Q{number}.sql").read_text()).fetchdf()


def quality(ev, name: str, csv: Path) -> tuple[str, float]:
    """Return (metric name, value). Higher F1 and correlation are better; lower relative error is better."""
    number = int(name.strip("Qrp"))
    try:
        system = pd.read_csv(csv)
    except pd.errors.EmptyDataError:
        system = pd.DataFrame()
    result = dataclasses.asdict(ev._evaluate_single_query(number, system, ground_truth(ev, number)))
    for key in ("f1_score", "relative_error", "spearman_correlation"):
        if result.get(key) is not None:
            return key, float(result[key])
    raise ValueError(result)


def unified(metric: str, value: float) -> float:
    """SemBench's quality aggregation: F1 and correlation as they are, relative error turned into 1 / (1 + error). Higher is better."""
    return 1 / (1 + value) if metric == "relative_error" else value


def report() -> None:
    ev = evaluator()
    rows = []
    for folder in sorted(OUT.iterdir()):
        if not folder.is_dir():
            continue
        for meta in sorted(folder.glob("Q*.json")):
            record = json.loads(meta.read_text())
            metric, value = quality(ev, record["query"], meta.with_suffix(".csv"))
            rows.append({**record, "metric": metric, "raw": value, "quality": unified(metric, value)})
    (OUT / "scores.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n")
    published = {}
    for path in sorted((MOVIE / "metrics/across_system_2.5flash").glob("*.json")):
        for name, m in json.loads(path.read_text()).items():
            found = next(((k, m[k]) for k in ("f1_score", "relative_error", "spearman_correlation") if isinstance(m.get(k), (int, float))), None)
            if found:
                published[path.stem, name] = {"raw": found[1], "quality": unified(*found), "seconds": m["execution_time"], "cost": m["money_cost"]}
    systems = sorted({system for system, _ in published})
    modes = ["jev", "llm", "cascade"]
    names = sorted({r["query"] for r in rows}, key=lambda s: (int(s.strip("Qrp")), s))
    mid = lambda xs: sorted(xs)[len(xs) // 2]

    def cell(name: str, mode: str) -> dict | None:
        """For a mode run several times, take the median and give the minimum and maximum of the raw metric."""
        runs = [r for r in rows if r["query"] == name and r["mode"] == mode]
        if not runs:
            return None
        llm = [r["llm"]["calls"] for r in runs if "llm" in r]
        return {"raw": mid([r["raw"] for r in runs]), "lo": min(r["raw"] for r in runs), "hi": max(r["raw"] for r in runs),
                "quality": mid([r["quality"] for r in runs]), "seconds": mid([r["seconds"] for r in runs]), "cost": mid([r["cost"] for r in runs]),
                "runs": len(runs), "jev_calls": mid([r["jev"]["calls"] for r in runs]) if "jev" in runs[0] else 0, "llm_calls": mid(llm) if llm else 0}

    print("Raw metrics: F1 (Q1, Q2, Q5-Q7) and correlation (Q9, Q10), higher is better; relative error (Q3, Q4, Q8), lower is better.")
    print(f"{'':5s}" + "".join(f"{m:>26s}" for m in modes) + "".join(f"{s:>12s}" for s in systems))
    for name in names:
        line = f"{name:5s}"
        for mode in modes:
            c = cell(name, mode)
            line += f"{'—':>26s}" if c is None else f"{c['raw']:>12.3f}" + (f" ({c['lo']:.3f}~{c['hi']:.3f})" if c["runs"] > 1 else " " * 14)
        line += "".join(f"{published[s, name]['raw']:>12.3f}" if (s, name) in published else f"{'—':>12s}" for s in systems)
        print(line)
    for title, field, fmt in (("time (seconds)", "seconds", "{:>12.1f}"), ("cost (US$)", "cost", "{:>12.4f}")):
        print(f"\n{title}")
        for name in names:
            line = f"{name:5s}"
            for mode in modes:
                c = cell(name, mode)
                line += f"{'—':>12s}" if c is None else fmt.format(c[field])
            line += "".join(fmt.format(published[s, name][field]) if (s, name) in published else f"{'—':>12s}" for s in systems)
            print(line)
    print("\ncalls (Jev / LLM)")
    for name in names:
        print(f"{name:5s}" + "".join(f"{'—':>16s}" if (c := cell(name, m)) is None else f"{c['jev_calls']:>9d}/{c['llm_calls']:<6d}" for m in modes))
    for label, subset in (("Q1~Q10", [f"Q{i}" for i in range(1, 11)]), ("Q1~Q8", [f"Q{i}" for i in range(1, 9)])):
        print(f"\n{label} summary: mean quality (higher is better) / total time / total cost")
        for mode in modes:
            cells = [cell(n, mode) for n in subset]
            if all(cells):
                print(f"  {mode:12s}{sum(c['quality'] for c in cells) / len(cells):.3f}  {sum(c['seconds'] for c in cells):9.1f}s  ${sum(c['cost'] for c in cells):.4f}")
        for s in systems:
            cells = [published.get((s, n)) for n in subset]
            if all(cells):
                print(f"  {'[published] ' + s:24s}{sum(c['quality'] for c in cells) / len(cells):.3f}  {sum(c['seconds'] for c in cells):9.1f}s  ${sum(c['cost'] for c in cells):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("mode", choices=["jev", "llm", "cascade", "open"])
    run.add_argument("--llm", help="OpenRouter route name, see kevdb/routes.py")
    run.add_argument("--open", help="open mode: which open model, see kevdb/open_models.py")
    run.add_argument("--out", type=Path, help="open mode: parent directory of the results (required, so they cannot mix with phase 1's)")
    run.add_argument("--tag", help="open mode: label of this run (default r1)")
    run.add_argument("--tables", type=Path, help="open mode: directory with Movies.csv and Reviews.csv (made-up tables in rehearsals)")
    run.add_argument("--runs", type=int, default=1, help="how many cold repeats (labels r1, r2, ...)")
    run.add_argument("--only", nargs="*", help="run only these queries")
    run.add_argument("--concurrency", type=int, default=32, help="concurrency for LLMs and open models; Jev is fixed at 16")
    run.add_argument("--cache", default=".jevcache.sqlite", help="cache file. Two processes running at once use one each")
    sub.add_parser("report")
    sub.add_parser("verify").add_argument("--llm", required=True)
    args = parser.parse_args()
    if args.cmd == "report":
        return report()
    if args.cmd == "verify":
        return verify(args.llm)
    load_dotenv()
    backends = {}
    if args.mode in ("jev", "cascade"):
        backends["jev"] = jev_backend(concurrency=16, cache_path=args.cache)
    if args.mode in ("llm", "cascade"):
        backends["llm"] = llm_backend(args.llm, concurrency=args.concurrency, cache_path=args.cache)
    if args.mode == "open":
        if not args.open or not args.out:
            parser.error("the open mode needs both --open and --out")
        backends["open"] = open_backend(args.open, concurrency=args.concurrency, cache_path=args.cache)
        run_mode("open", args.tag or "r1", backends, args.only, args.out, args.tables)
    for i in range(1, args.runs + 1 if args.mode != "open" else 1):
        run_mode(args.mode, f"r{i}", backends, args.only)
    print("new spending this run: " + ", ".join(f"{role} ${b.spent:.4f}" for role, b in backends.items()))
    for b in backends.values():
        b.close()


if __name__ == "__main__":
    main()
