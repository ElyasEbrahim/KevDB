"""Exploratory study: can Jev tell, from table names, column names and sample values alone, that "knowing column A nearly determines column B",
and how much more accurate do the optimizer's row estimates get when joint statistics (CREATE STATISTICS) are built on its judgement?

    python experiments/query_plan/corr.py truth      # compute each column pair's real dependency from the data (no model)
    python experiments/query_plan/corr.py judge [--llm MODEL] [--names-only]
    python experiments/query_plan/corr.py eval       # the model's judgements vs the truth
    python experiments/query_plan/corr.py impact     # the optimizer's row-estimate errors before and after building statistics
    python experiments/query_plan/corr.py cleanup    # drop everything this experiment created in the database

The truth is Goodman–Kruskal's λ(B|A): how much knowing A reduces the error of predicting B, relative to always guessing B's mode.
0 means A does not help predict B at all; 1 means A fully determines B.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from itertools import combinations
from pathlib import Path

import psycopg

from kevdb import JevClient, Question, Task, run_task
from kevdb.cache import Cache
from kevdb.cli import load_dotenv
from kevdb.llm import LLMClient

DSN = "postgresql://job:job@127.0.0.1:5434/imdb"
OUT = Path(__file__).resolve().parents[2] / "out" / "query_plan"
SAMPLE_ROWS = 1_000_000
# Drop primary keys, hash columns and all-empty columns; small dictionary tables with a single usable column take no part
TABLES = {
    "cast_info": ["person_id", "movie_id", "person_role_id", "note", "nr_order", "role_id"],
    "movie_info": ["movie_id", "info_type_id", "info", "note"],
    "movie_keyword": ["movie_id", "keyword_id"],
    "name": ["name", "imdb_index", "gender", "name_pcode_cf", "name_pcode_nf", "surname_pcode"],
    "char_name": ["name", "name_pcode_nf", "surname_pcode"],
    "person_info": ["person_id", "info_type_id", "info", "note"],
    "movie_companies": ["movie_id", "company_id", "company_type_id", "note"],
    "title": ["title", "imdb_index", "kind_id", "production_year", "phonetic_code", "episode_of_id", "season_nr", "episode_nr", "series_years"],
    "movie_info_idx": ["movie_id", "info_type_id", "info"],
    "aka_name": ["person_id", "name", "imdb_index", "name_pcode_cf", "name_pcode_nf", "surname_pcode"],
    "aka_title": ["movie_id", "title", "imdb_index", "kind_id", "production_year", "phonetic_code", "episode_of_id", "season_nr", "episode_nr", "note"],
    "company_name": ["name", "country_code", "name_pcode_nf", "name_pcode_sf"],
    "complete_cast": ["movie_id", "subject_id", "status_id"],
    "keyword": ["keyword", "phonetic_code"],
    "movie_link": ["movie_id", "linked_movie_id", "link_type_id"],
}
# When fewer rows than this share repeated A values in the sample, A is near-unique: it "determines" B only because each value occurs once.
# The database already knows that from its single-column statistics, so it does not count as a semantic dependency
MIN_COVERAGE = 0.5

QUESTION = Question(
    "noul",
    "In the database table `table`, if you know a row's value in the column `determinant`, can you usually tell that row's value in the column `dependent`?",
    {
        "true": "Yes. The dependent column is largely determined by, or very strongly associated with, the determinant column.",
        "false": "No. Knowing the determinant column tells you little about the dependent column.",
    },
)


def connect() -> psycopg.Connection:
    return psycopg.connect(DSN, autocommit=True)


def read(name: str) -> list[dict]:
    return [json.loads(line) for line in (OUT / name).read_text().splitlines()]


def cmd_truth(_: argparse.Namespace) -> None:
    rows = []
    with connect() as conn, conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS corr")
        for table, columns in TABLES.items():
            cur.execute("SELECT reltuples::bigint FROM pg_class WHERE relname = %s", (table,))
            total = cur.fetchone()[0]
            pct = min(100.0, 100.0 * SAMPLE_ROWS / max(total, 1))
            cur.execute(f"DROP TABLE IF EXISTS corr.{table}")
            cur.execute(f"CREATE TABLE corr.{table} AS SELECT {', '.join(columns)} FROM {table} TABLESAMPLE BERNOULLI ({pct}) REPEATABLE (1)")
            cur.execute(f"SELECT count(*) FROM corr.{table}")
            n = cur.fetchone()[0]
            examples = {}
            for c in columns:
                cur.execute(f"SELECT left({c}::text, 40) FROM corr.{table} WHERE {c} IS NOT NULL GROUP BY 1 ORDER BY count(*) DESC LIMIT 5")
                examples[c] = [r[0] for r in cur.fetchall()]
            for a, b in combinations(columns, 2):
                for det, dep in ((a, b), (b, a)):
                    cur.execute(f"""
                        WITH g AS (SELECT {det} AS a, {dep} AS b, count(*) AS c FROM corr.{table} GROUP BY 1, 2),
                             ga AS (SELECT a, sum(c) AS n, max(c) AS m FROM g GROUP BY a),
                             gb AS (SELECT b, sum(c) AS n FROM g GROUP BY b)
                        SELECT (SELECT coalesce(sum(m) FILTER (WHERE n >= 2), 0) FROM ga),
                               (SELECT coalesce(sum(n) FILTER (WHERE n >= 2), 0) FROM ga),
                               (SELECT max(n) FROM gb)""")
                    hit, covered, modal = (float(x) for x in cur.fetchone())
                    coverage, base = covered / n, modal / n
                    predictable = hit / covered if covered else 0.0
                    lam = max(0.0, (predictable - base) / (1 - base)) if base < 1 else 0.0
                    rows.append({"id": f"{table}.{det}->{dep}", "table": table, "determinant": det, "dependent": dep,
                                 "sample_rows": n, "table_rows": total, "coverage": round(coverage, 4), "base": round(base, 4),
                                 "predictable": round(predictable, 4), "lambda": round(lam, 4),
                                 "det_examples": examples[det], "dep_examples": examples[dep]})
            print(f"{table}: {n:,} sampled rows, {len(columns)} columns", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "corr_truth.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    usable = [r for r in rows if r["coverage"] >= MIN_COVERAGE]
    print(f"\n{len(rows)} directed column pairs in total; {len(usable)} where A is not near-unique, of which {sum(r['lambda'] >= 0.5 for r in usable)} with λ≥0.5 and {sum(r['lambda'] < 0.1 for r in usable)} with λ<0.1")


async def cmd_judge(args: argparse.Namespace) -> None:
    load_dotenv()
    truth = read("corr_truth.jsonl")
    variant = "names" if args.names_only else "examples"
    items = []
    for r in truth:
        det, dep = {"name": r["determinant"]}, {"name": r["dependent"]}
        if not args.names_only:
            det["most_common_values"], dep["most_common_values"] = r["det_examples"], r["dep_examples"]
        items.append({"id": r["id"], "table": r["table"], "determinant": det, "dependent": dep})
    task = Task(name="qo_corr", input="", questions={"depends": QUESTION}, state_fields=["table", "determinant", "dependent"],
                provider="typesafe", model="jev-1.13.0", concurrency=16)
    if args.llm:
        client, label = LLMClient(provider="opencode", model=args.llm, timeout=120.0, reasoning="none"), args.llm
    else:
        client, label = JevClient(provider=task.provider, model=task.model), task.model
    async with client:
        results, stats = await run_task(task, items, client, Cache())
    path = OUT / f"corr_pred.{label}.{variant}.jsonl"
    path.write_text("".join(json.dumps({"id": i["id"], "p": r.get("depends_p"), "error": r.get("_error")}) + "\n" for i, r in zip(items, results)))
    print(f"{label} / {variant}: {len(items)} questions, {stats.errors} failed, {stats.seconds:.1f}s, {stats.input_tokens:,} input tokens, about ${stats.cost_usd:.4f} → {path.name}")


def auc(scored: list[tuple[float, bool]]) -> float | None:
    pos, neg = [s for s, y in scored if y], [s for s, y in scored if not y]
    if not pos or not neg:
        return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=v.__getitem__)
        out, i = [0.0] * len(v), 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


def ndistinct() -> dict[str, float]:
    """The database's existing single-column statistics: distinct values per column. A reference that needs no model."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT s.tablename || '.' || s.attname,
                              CASE WHEN s.n_distinct < 0 THEN -s.n_distinct * c.reltuples ELSE s.n_distinct END
                       FROM pg_stats s JOIN pg_class c ON c.relname = s.tablename WHERE s.schemaname = 'public'""")
        return {k: float(v) for k, v in cur.fetchall()}


def cmd_eval(_: argparse.Namespace) -> None:
    truth = {r["id"]: r for r in read("corr_truth.jsonl")}
    usable = [r for r in truth.values() if r["coverage"] >= MIN_COVERAGE]
    print(f"{len(truth)} directed column pairs; {len(usable)} where A is not near-unique (evaluated on these only)")
    print(f"of which {sum(r['lambda'] >= 0.5 for r in usable)} with λ≥0.5 (strong dependency) and {sum(r['lambda'] < 0.1 for r in usable)} with λ<0.1 (essentially unrelated)\n")
    nd = ndistinct()
    scorers: dict[str, dict[str, float]] = {
        "reference: distinct counts only (A many, B few)": {r["id"]: (nd.get(f"{r['table']}.{r['determinant']}", 1) + 1) / (nd.get(f"{r['table']}.{r['dependent']}", 1) + 1) for r in usable},
    }
    for path in sorted(OUT.glob("corr_pred.*.jsonl")):
        preds = {r["id"]: r["p"] for r in read(path.name) if r["p"] is not None}
        scorers[path.name.removeprefix("corr_pred.").removesuffix(".jsonl")] = preds
    print(f"{'source':<34}{'AUC (λ≥0.5)':>12}{'AUC (λ≥0.3)':>12}{'rank corr':>10}{'flagged':>8}{'of which λ≥0.5':>15}")
    for name, score in scorers.items():
        rows = [r for r in usable if r["id"] in score]
        s = [score[r["id"]] for r in rows]
        a5 = auc([(score[r["id"]], r["lambda"] >= 0.5) for r in rows])
        a3 = auc([(score[r["id"]], r["lambda"] >= 0.3) for r in rows])
        flagged = [r for r in rows if score[r["id"]] >= 0.5] if "reference" not in name else []
        tail = f"{len(flagged):>10}{sum(r['lambda'] >= 0.5 for r in flagged):>12}" if flagged else f"{'-':>10}{'-':>12}"
        print(f"{name:<36}{a5:>12.2f}{a3:>13.2f}{spearman(s, [r['lambda'] for r in rows]):>10.2f}{tail}")


def unordered(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    """Merge both directions into one column pair: keep it if A is not near-unique in at least one direction, and take the larger usable dependency."""
    pairs: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        a, b = sorted((r["determinant"], r["dependent"]))
        key = (r["table"], a, b)
        entry = pairs.setdefault(key, {"lambda": 0.0, "usable": False, "table_rows": r["table_rows"], "sample_rows": r["sample_rows"]})
        if r["coverage"] >= MIN_COVERAGE:
            entry["usable"] = True
            entry["lambda"] = max(entry["lambda"], r["lambda"])
    return {k: v for k, v in pairs.items() if v["usable"]}


def cmd_impact(args: argparse.Namespace) -> None:
    truth = read("corr_truth.jsonl")
    pairs = unordered(truth)
    jev = {r["id"]: r["p"] for r in read("corr_pred.jev-1.13.0.examples.jsonl")}
    jev_score = {k: max(jev.get(f"{k[0]}.{k[1]}->{k[2]}", 0), jev.get(f"{k[0]}.{k[2]}->{k[1]}", 0)) for k in pairs}
    rng = random.Random(0)
    with connect() as conn, conn.cursor() as cur:
        drop_stats(cur)
        # Workload: for each column pair, the most common value combinations in the sample, asking "how many rows have A = x and B = y"
        workload = []
        for (table, a, b), info in pairs.items():
            cur.execute(f"SELECT {a}, {b}, count(*) FROM corr.{table} WHERE {a} IS NOT NULL AND {b} IS NOT NULL GROUP BY 1, 2 HAVING count(*) >= 200 ORDER BY 3 DESC LIMIT 20")
            combos = cur.fetchall()
            # Some column pairs have no sufficiently common value combinations; skip them
            for va, vb, c in (combos[:1] + rng.sample(combos[1:], min(2, max(len(combos) - 1, 0)))):
                workload.append({"pair": (table, a, b), "sql": f"SELECT * FROM {table} WHERE {a} = %s AND {b} = %s", "params": (va, vb),
                                 "actual": c * info["table_rows"] / info["sample_rows"]})
        covered = {w["pair"] for w in workload}
        print(f"{len(pairs)} usable column pairs, {len(covered)} with sufficiently common value combinations, {len(workload)} queries in total\n")

        def measure(label: str, chosen: list[tuple[str, str, str]]) -> dict:
            drop_stats(cur)
            for i, (table, a, b) in enumerate(chosen):
                cur.execute(f"CREATE STATISTICS corr_s{i} ON {a}, {b} FROM {table}")
            started = time.perf_counter()
            for table in sorted({p[0] for p in pairs}):
                cur.execute(f"ANALYZE {table}")
            analyze_s = time.perf_counter() - started
            errors = []
            for w in workload:
                cur.execute("EXPLAIN (FORMAT JSON) " + w["sql"], w["params"])
                est = max(cur.fetchone()[0][0]["Plan"]["Plan Rows"], 1)
                errors.append(max(est / w["actual"], w["actual"] / est))
            errors.sort()
            result = {"label": label, "stats": len(chosen), "analyze_s": round(analyze_s, 1), "median": errors[len(errors) // 2],
                      "p90": errors[int(len(errors) * 0.9)], "max": errors[-1], "within2": sum(e <= 2 for e in errors) / len(errors)}
            print(f"{label:<30}{len(chosen):>6}{result['median']:>10.2f}{result['p90']:>10.1f}{result['max']:>12.0f}{result['within2']:>10.0%}{analyze_s:>10.1f}s", flush=True)
            return result

        keys = list(pairs)
        by_jev = sorted(keys, key=lambda k: -jev_score[k])
        by_truth = sorted(keys, key=lambda k: -pairs[k]["lambda"])
        n = sum(jev_score[k] >= 0.5 for k in keys)
        print(f"{'joint statistics built on':<28}{'count':>6}{'median err':>11}{'p90 err':>9}{'max err':>9}{'err ≤2x':>9}{'ANALYZE':>10}")
        results = [measure("none (database default)", [])]
        results.append(measure(f"pairs Jev flags as dependent (p≥0.5)", by_jev[:n]))
        results.append(measure(f"the same number of truly most dependent pairs", by_truth[:n]))
        for seed in (1, 2, 3):
            results.append(measure(f"the same number at random (draw {seed})", random.Random(seed).sample(keys, n)))
        results.append(measure("all column pairs", keys))
        drop_stats(cur)
        for table in sorted({p[0] for p in pairs}):
            cur.execute(f"ANALYZE {table}")
    (OUT / "corr_impact.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))


def drop_stats(cur: psycopg.Cursor) -> None:
    cur.execute("SELECT stxname FROM pg_statistic_ext WHERE stxname LIKE 'corr_s%'")
    for (name,) in cur.fetchall():
        cur.execute(f"DROP STATISTICS {name}")


def cmd_cleanup(_: argparse.Namespace) -> None:
    with connect() as conn, conn.cursor() as cur:
        drop_stats(cur)
        cur.execute("DROP SCHEMA IF EXISTS corr CASCADE")
        for table in TABLES:
            cur.execute(f"ANALYZE {table}")
    print("dropped the statistics and sample tables this experiment created and gathered the default statistics again")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("truth")
    j = sub.add_parser("judge"); j.add_argument("--llm"); j.add_argument("--names-only", action="store_true")
    sub.add_parser("eval"); sub.add_parser("impact"); sub.add_parser("cleanup")
    args = parser.parse_args()
    if args.cmd == "judge":
        asyncio.run(cmd_judge(args))
    else:
        {"truth": cmd_truth, "eval": cmd_eval, "impact": cmd_impact, "cleanup": cmd_cleanup}[args.cmd](args)


if __name__ == "__main__":
    main()
