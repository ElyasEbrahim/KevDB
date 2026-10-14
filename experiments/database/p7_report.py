"""Phase 7 analysis: does a local decision model inside DuckDB offer anything a cloud API cannot? Reads the collected results only and calls no model.

Questions, metrics and decision rules are in experiments/database/protocol/p7.txt, all written before collection; the constants below are those criteria.
Numbers for Jev, the cheap LLM and the published systems come from the outputs of earlier phases and serve as comparison columns only.

    python experiments/database/p7_report.py | tee out/database/p7/kev-4b/report.txt

In a rehearsal (P7_REHEARSAL=1) it reads results on made-up rows, only to walk every step; the numbers mean nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import p1_movie as P1
import p6_report as R
import p7_cache as CACHE
import p7_calibration as CAL
from kevdb import JEV_MODEL, calibration
from p7_data import BUDGET_STATES, EM, FRESH, LABELS, MAIN_SETS, MEMORY_BUDGET, MODEL, OUT, REHEARSAL, REPS, SHARED, SWEEP_SETS, WHOLE_TABLES, cache_set, em_question, made_tables
from paths import OUT as RESULTS, ROOT

# ---------- Criteria fixed beforehand (protocol/p7.txt) ----------
PER_REVIEW = ["Q1", "Q2", "Q3", "Q4", "Q5r", "Q6r", "Q7r", "Q8", "Q9", "Q10"]  # the 10 queries whose requests each contain a single review (joins in their rewritten form)
LONG_BAR, SHORT_BAR, AGREE_BAR = 3.0, 2.0, 0.995
SPREAD_FLAG = 0.15  # flag when the passes of one arm differ in time by more than this share
# The cache's memory allowance MEMORY_BUDGET and the table sizes WHOLE_TABLES live in p7_data.py: the budget was set per model by the capacity probe before collection (use as much memory as possible)
BOOT, SEED, DRAWS = 2000, 20261011, 200
SIZES = (25, 50, 100, 200, 500)

out: dict = {"model": MODEL, "rehearsal": REHEARSAL}
mid = lambda xs: sorted(xs)[len(xs) // 2]


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# ---------- A: semantic-SQL benchmark ----------

def sql_runs() -> tuple[dict, dict, dict]:
    """(the local model's runs, phase 1's reference, the published systems). Each cell: raw metric, quality, seconds, requests."""
    ev = P1.evaluator()
    if REHEARSAL:
        ev.movies_df, ev.reviews_df = (pd.read_csv(made_tables() / f"{t}.csv") for t in ("Movies", "Reviews"))
    runs: dict = {}
    for folder in sorted((OUT / "sql").glob("open.*")):
        if not folder.is_dir():
            continue
        cells = {}
        for meta in sorted(folder.glob("Q*.json")):
            r = json.loads(meta.read_text())
            metric, value = P1.quality(ev, r["query"], meta.with_suffix(".csv"))
            cells[r["query"]] = {"metric": metric, "raw": value, "quality": P1.unified(metric, value), "seconds": r["seconds"], "calls": r["open"]["calls"],
                                 "fresh": r["open"].get("fresh"), "errors": r["open"]["errors"], "attempts": r["attempts"], "csv": meta.with_suffix(".csv").read_text()}
        runs[folder.name.removeprefix("open.")] = cells
    reference: dict = {}
    for r in json.loads((RESULTS / "p1" / "scores.json").read_text()):
        reference.setdefault(r["mode"], {}).setdefault(r["query"], []).append(r)
    reference = {mode: {q: {"raw": mid([r["raw"] for r in rs]), "quality": mid([r["quality"] for r in rs]), "seconds": mid([r["seconds"] for r in rs]),
                            "cost": mid([r["cost"] for r in rs])} for q, rs in cells.items()} for mode, cells in reference.items()}
    published: dict = {}
    for path in sorted((P1.MOVIE / "metrics/across_system_2.5flash").glob("*.json")):
        for name, m in json.loads(path.read_text()).items():
            found = next(((k, m[k]) for k in ("f1_score", "relative_error", "spearman_correlation") if isinstance(m.get(k), (int, float))), None)
            if found:
                published.setdefault(path.stem, {})[name] = {"raw": found[1], "quality": P1.unified(*found), "seconds": m["execution_time"], "cost": m["money_cost"]}
    return runs, reference, published


def total(cells: dict, names: list[str]) -> dict | None:
    if not all(n in cells for n in names):
        return None
    return {"quality": sum(cells[n]["quality"] for n in names) / len(names), "seconds": sum(cells[n]["seconds"] for n in names), "cost": sum(cells[n].get("cost", 0.0) for n in names)}


def section_sql() -> None:
    runs, ref, published = sql_runs()
    ten = [f"Q{i}" for i in range(1, 11)]
    print("=" * 100 + "\nA  Semantic-SQL benchmark: SemBench's movie scenario, local model at its server's factory settings, every query cold\n" + "=" * 100)
    main = runs.get("default", {})
    if not all(q in main for q in ten):
        print("  no complete results yet")
        return
    print(f"{'query':6s}{'metric':>22s}{MODEL:>10s}{'Jev':>8s}{'LLM':>8s}   |{'s':>4s}{MODEL:>9s}{'Jev':>8s}{'LLM':>8s}   |{'requests':>9s}{'no ans':>7s}{'rounds':>7s}")
    for q in sorted(main, key=lambda s: (int(s.strip("Qrp")), s)):
        c, j, l = main[q], ref["jev"].get(q), ref["llm"].get(q)
        show = lambda cell, key, fmt: format(cell[key], fmt) if cell else f"{'—':>8s}"
        print(f"{q:6s}{c['metric']:>24s}{c['raw']:>10.3f}{show(j, 'raw', '8.3f')}{show(l, 'raw', '8.3f')}   |{'':5s}{c['seconds']:>9.1f}{show(j, 'seconds', '8.1f')}{show(l, 'seconds', '8.1f')}"
              f"   |{c['calls']:>9d}{c['errors']:>8d}{c['attempts']:>6d}")
    kev, jev, llm = total(main, ten), total(ref["jev"], ten), total(ref["llm"], ten)
    print("\nQ1-Q10 summary: mean quality (higher is better) / total time / total cost")
    print(f"  {MODEL:24s}{kev['quality']:.3f}  {kev['seconds']:9.1f}s  $0 (local; electricity and hardware not counted)")
    for label, t in (("Jev (official API)", jev), ("LLM (GPT 6 Luna, low reasoning)", llm)):
        print(f"  {label:24s}{t['quality']:.3f}  {t['seconds']:9.1f}s  ${t['cost']:.4f}")
    for system, cells in published.items():
        if t := total(cells, ten):
            print(f"  {'[published] ' + system:24s}{t['quality']:.3f}  {t['seconds']:9.1f}s  ${t['cost']:.4f}")
    unanswered, warm = sum(main[q]["errors"] for q in ten), [q for q in main if main[q]["fresh"] is not None and main[q]["fresh"] != main[q]["calls"]]
    passed = kev["quality"] >= llm["quality"] and unanswered == 0
    wonder = kev["quality"] >= jev["quality"] and kev["seconds"] <= llm["seconds"] and unanswered == 0
    print(f"\n  qualifying bar (mean quality not below the cheap LLM's {llm['quality']:.3f}, and no unanswered rows): {'pass' if passed else 'fail'} ({kev['quality']:.4f}, {unanswered} unanswered rows)")
    print(f"  wonder bar (mean quality not below Jev's {jev['quality']:.3f}, and total time not above the cheap LLM's {llm['seconds']:.0f} s): {'pass' if wonder else 'fail'}"
          f" (quality difference {kev['quality'] - jev['quality']:+.4f}, time {kev['seconds'] / llm['seconds']:.2f}x the LLM's and {kev['seconds'] / jev['seconds']:.1f}x Jev's)")
    if warm:
        print(f"  note: not every request of these queries was new this time (resumed after an interruption), so their times are not cold: {warm}")
    out["sql"] = {"queries": {q: {k: v for k, v in c.items() if k != "csv"} for q, c in main.items()}, "total": kev, "jev": jev, "llm": llm, "passed": bool(passed), "wonder": bool(wonder), "unanswered": unanswered}

    print("\n  The 10 per-review queries (joins rewritten): the local model under two cache settings, against Jev and the LLM")
    print(f"  {'':26s}{'mean quality':>13s}{'total time':>11s}")
    rows = {"factory setting (4 prefixes)": total(main, PER_REVIEW), f"cache at budget ({BUDGET_STATES} prefixes)": total(runs.get("budget", {}), PER_REVIEW),
            "Jev (official API)": total(ref["jev"], PER_REVIEW), "LLM": total(ref["llm"], PER_REVIEW)}
    for label, t in rows.items():
        print(f"  {label:26s}" + (f"{t['quality']:>10.3f}{t['seconds']:>11.1f}s" if t else f"{'no results':>10s}"))
    if budget := runs.get("budget"):
        if all(q in budget for q in PER_REVIEW):
            print(f"  {'query':6s}{'factory (s)':>16s}{'at budget (s)':>16s}{'ratio':>10s}{'same result':>12s}")
            for q in PER_REVIEW:
                same = budget[q]["csv"] == main[q]["csv"]
                print(f"  {q:6s}{main[q]['seconds']:>16.1f}{budget[q]['seconds']:>16.1f}{main[q]['seconds'] / budget[q]['seconds']:>10.2f}{'yes' if same else 'no':>10s}")
            cards = {when: json.loads(p.read_text()) for when in ("before", "after") if (p := OUT / "sql" / f"open.budget.server.{when}.json").exists()}
            if len(cards) == 2:
                hits, misses = (cards["after"]["prefix_cache"][k] - cards["before"]["prefix_cache"][k] for k in ("hits", "misses"))
                print(f"  server counters: {hits} hits, {misses} misses (hit rate {hits / max(hits + misses, 1):.1%}), {cards['after']['prefix_cache']['cached_states']} prefixes cached at the end")
                out["sql"]["budget_hits"] = {"hits": hits, "misses": misses}
            out["sql"]["per_review"] = {k: v for k, v in zip(("default", "budget", "jev", "llm"), rows.values())}
            out["sql"]["budget_same_results"] = {q: budget[q]["csv"] == main[q]["csv"] for q in PER_REVIEW}


# ---------- B: prefix cache ----------

def cache_record(name: str, arm: str, rep: int) -> dict | None:
    path = CACHE.OUT / f"{CACHE.stem(name, arm, rep)}.json"
    return json.loads(path.read_text()) if path.exists() else None


def decided(answer: dict) -> tuple[object, list[float]]:
    """(judgement, probability vector). Yes/no: probability at least 0.5; multiple choice: the chosen option; score: the most probable level."""
    if answer["type"] == "noul":
        return answer["noul"] >= 0.5, [answer["noul"]]
    probs = list(answer["probabilities"].values())
    return (answer["choice"] if answer["type"] == "choice" else int(np.argmax(probs))), probs


def compare(a: list[dict], b: list[dict]) -> dict:
    """Compare two runs' answers row by row for the same pass: share of identical judgements and probability differences. A row missing on either side counts as different."""
    A, B = {r["id"]: r["answers"] for r in a}, {r["id"]: r["answers"] for r in b}
    same, gaps = 0, []
    for key, x in A.items():
        y = B.get(key)
        if not x or not y:
            continue
        (dx, px), (dy, py) = decided(x["q"]), decided(y["q"])
        same += dx == dy
        gaps.append(max(abs(u - v) for u, v in zip(px, py)))
    return {"n": len(A), "same": same, "compared": len(gaps), "mean_gap": float(np.mean(gaps)) if gaps else None, "max_gap": float(np.max(gaps)) if gaps else None}


def pooled(parts: list[dict]) -> dict:
    """Several comparisons combined. The identical-judgement share uses all answers as denominator (missing ones count as different); probability differences use rows answered on both sides only."""
    n, same, compared = sum(p["n"] for p in parts), sum(p["same"] for p in parts), sum(p["compared"] for p in parts)
    gaps = [p["mean_gap"] * p["compared"] for p in parts if p["compared"]]
    return {"n": n, "agree": same / n if n else None, "mean_gap": sum(gaps) / compared if compared else None, "max_gap": max((p["max_gap"] for p in parts if p["compared"]), default=None)}


def answers_of(name: str, arm: str, rep: int, which: object) -> list[dict]:
    return read(CACHE.OUT / f"{CACHE.stem(name, arm, rep)}.{which}.jsonl")


def long_sanity(got: list[dict], docs: dict) -> dict:
    """Long-document answers against each document's share of positive reviews: the three questions are "mostly positive?", multiple choice and score. got holds {document: answer} for each question."""
    ids = [i for i in docs if all(i in g for g in got)]
    share = np.array([docs[i] for i in ids])
    p, choice, score = np.array([got[0][i]["noul"] for i in ids]), np.array([got[1][i]["choice"] == "POSITIVE" for i in ids]), np.array([got[2][i]["score"] for i in ids])
    return {"n": len(ids), "noul_accuracy": float(((p >= 0.5) == (share >= 0.5)).mean()), "noul_auc": float(R.auc(p, (share >= 0.5).astype(int))),
            "choice_accuracy": float((choice == (share >= 0.5)).mean()), "score_spearman": float(spearmanr(score, share).statistic)}


def jev_long(result: dict, docs: dict) -> None:
    """Reference column: the official Jev API answering the same three questions on the same long documents (experiments/database/p7_jev_long.py). No threshold."""
    folder = SHARED / "jev"
    if not (folder / "runs.jsonl").exists():
        print("  (no Jev reference on the long documents)")
        return
    passes = [read(folder / f"long.{i}.jsonl") for i in (1, 2, 3)]
    answered = [[r for r in rows if r["answers"]] for rows in passes]
    seconds = [sum(r["latency"] for r in rows) for rows in answered]
    latency = [mid([r["latency"] for r in rows]) for rows in answered]
    cost, tokens = [sum(r["cost"] for r in rows) for rows in answered], [sum(r["tokens"] for r in rows) / max(len(rows), 1) for rows in answered]
    sanity = long_sanity([{r["id"]: r["answers"]["q"] for r in rows} for rows in answered], docs)
    kev = result["sets"]["long"]
    print(f"\nreference: the official Jev API on the same {len(passes[0])} long documents (concurrency 1, every question re-sends the whole document; {sum(len(a) - len(b) for a, b in zip(passes, answered))} unanswered)")
    print(f"  {'question':8s}{'Jev per doc':>12s}{'local off':>12s}{'local hit':>12s}   |{'pass total (s)':>14s}{'Jev':>8s}{'off':>8s}{'table':>8s}   |{'Jev tokens per doc':>19s}{'Jev cost this pass':>19s}")
    for j, q in enumerate(kev["questions"]):
        print(f"  {j + 1:<10d}{latency[j] * 1000:>13.0f} ms{kev['latency_p50']['off'][j] * 1000:>11.0f} ms{kev['latency_p50']['table'][j] * 1000:>13.0f} ms   |{'':14s}{seconds[j]:>8.1f}"
              f"{kev['median_seconds']['off'][j]:>11.1f}{kev['median_seconds']['table'][j]:>12.1f}   |{tokens[j]:>22.0f}{cost[j]:>18.4f} US$")
    print(f"  three questions in total: Jev {sum(seconds):.1f} s, ${sum(cost):.4f}; local, cache off {sum(kev['median_seconds']['off']):.1f} s; local, whole-table cache {sum(kev['median_seconds']['table']):.1f} s. "
          f"Each extra question from the 2nd on: Jev {np.mean(seconds[1:]):.1f} s, ${np.mean(cost[1:]):.4f}; local with cache hits {np.mean(kev['median_seconds']['table'][1:]):.1f} s")
    print(f"  Jev's answers on these documents (same reference readings): \"mostly positive?\" accuracy {sanity['noul_accuracy']:.1%}, AUC {sanity['noul_auc']:.3f}; multiple-choice accuracy {sanity['choice_accuracy']:.1%}; "
          f"rank correlation of score with share positive {sanity['score_spearman']:.3f}")
    result["jev_long"] = {"seconds": seconds, "latency_p50": latency, "cost": cost, "tokens_per_doc": tokens, "sanity": sanity}


def section_cache() -> None:
    print("\n" + "=" * 100 + "\nB  Prefix cache: how much cheaper is asking the same rows another question?\n" + "=" * 100)
    result: dict = {"sets": {}}
    memory_rows = []
    speed = R.jev_speed()
    jev_rate = {"movie": speed["movie"], "mams": speed["mams"]}
    for name in MAIN_SETS + SWEEP_SETS:
        reps = range(1, REPS + 1) if name in MAIN_SETS else (1,)
        recs = {arm: [r for rep in reps if (r := cache_record(name, arm, rep))] for arm in ("off", "table")}
        default = cache_record(name, "default", 1) if name in MAIN_SETS else None
        if len(recs["off"]) != len(reps) or len(recs["table"]) != len(reps):
            print(f"\n{name}: results incomplete (off {len(recs['off'])} passes, table {len(recs['table'])} passes, {len(reps)} each expected)")
            continue
        questions, rows = [p["question"] for p in recs["off"][0]["passes"]], recs["off"][0]["rows"]
        tokens = sum(b["tokens"] for b in recs["table"][0]["passes"][0]["blocks"]) / rows
        sec = {arm: [[r["passes"][j]["seconds"] for r in recs[arm]] for j in range(len(questions))] for arm in recs}
        med = {arm: [mid(xs) for xs in sec[arm]] for arm in sec}
        later = sum(med["off"][1:]) / sum(med["table"][1:])
        per_rep = [sum(recs["off"][i]["passes"][j]["seconds"] for j in range(1, len(questions))) / sum(recs["table"][i]["passes"][j]["seconds"] for j in range(1, len(questions))) for i in range(len(reps))]
        spread = max(max(xs) / mid(xs) - 1 for arm in sec for xs in sec[arm]), max(1 - min(xs) / mid(xs) for arm in sec for xs in sec[arm])
        print(f"\n{name}: {rows} rows, {tokens:.0f} tokens per request on average (question 1); off and table {len(reps)} passes each, medians")
        print(f"  {'question':8s}{'name':16s}{'off (s)':>12s}{'table (s)':>14s}{'ratio':>8s}{'factory (s)':>14s}   hits / misses (table)   median per row: off → on")
        for j, q in enumerate(questions):
            t = recs["table"][0]["passes"][j]
            lat = f"{mid([r['passes'][j]['latency_p50'] for r in recs['off']]) * 1000:.0f} → {mid([r['passes'][j]['latency_p50'] for r in recs['table']]) * 1000:.0f} ms"
            d = f"{default['passes'][j]['seconds']:>16.1f}" if default else f"{'':>16s}"
            print(f"  {j + 1:<10d}{q:16s}{med['off'][j]:>14.1f}{med['table'][j]:>16.1f}{med['off'][j] / med['table'][j]:>10.2f}{d}   {t['hits']:>6d} / {t['misses']:<6d}              {lat}")
        overhead = med["table"][0] / med["off"][0] - 1
        print(f"  from question 2 on: off {sum(med['off'][1:]):.1f} s, whole-table cache {sum(med['table'][1:]):.1f} s, {later:.2f}x faster (per pass: {', '.join(f'{x:.2f}' for x in per_rep)})")
        print(f"  question 1 (the pass that builds the cache) is {abs(overhead):.1%} {'slower' if overhead >= 0 else 'faster'} than off; passes of one arm differ by at most +{spread[0]:.1%} / -{spread[1]:.1%}"
              + (" (above 15%: timing noise on this dataset is high)" if max(spread) > SPREAD_FLAG else ""))
        # Mechanism check: the speed-up must match the hits the server itself reports
        ok_table = all(p["hits"] == (rows if j else 0) and p["misses"] == (0 if j else rows) for r in recs["table"] for j, p in enumerate(r["passes"]))
        ok_off = all(p["hits"] == 0 and p["misses"] == 0 for r in recs["off"] for p in r["passes"])
        print(f"  mechanism check: whole-table cache misses everything on question 1 and hits everything afterwards: {'yes' if ok_table else 'no'}; no hits with the cache off: {'yes' if ok_off else 'no'}"
              + (f"; factory setting hits {sum(p['hits'] for p in default['passes'][1:])} times from question 2 on" if default else ""))
        # How many prefixes the whole-table passes actually held and whether the cache was cleared for lack of memory: direct evidence, to read alongside the memory model's "fits or not" below
        held = [r["passes"][-1]["blocks"][-1] for r in recs["table"]]
        oom = sum(p["oom_retries"] for r in recs["table"] for p in r["passes"])
        print(f"  at the end of each whole-table pass: {', '.join(str(b['cached_states']) for b in held)} prefixes cached (including {recs['table'][0]['warmup']['cached_states']} from warm-up), "
              f"server at {', '.join(f'{b['footprint'] / 2**30:.1f}' for b in held)} GB, {oom} cache clears and retries for lack of memory"
              + (f". This dataset is the whole table ({rows} rows)" if WHOLE_TABLES.get(name) == rows else (f". The whole table has {WHOLE_TABLES[name]} rows; this is the first {rows} that fit the budget" if name in WHOLE_TABLES else "")))
        failed = sum(p["errors"] for arm in recs for r in recs[arm] for p in r["passes"])
        packed = mid([r["packed"]["seconds"] for r in recs["off"]])
        print(f"  reference: {len(questions)} questions packed into one request (cache off) {packed:.1f} s; asked one by one {len(questions)} times, cache off {sum(med['off']):.1f} s, whole-table cache {sum(med['table']):.1f} s. {failed} failed requests")
        if name in jev_rate:
            k = len(questions) - 1
            print(f"  reference: from question 2 on, {rows * k / sum(med['off'][1:]):.1f} rows per second (cache off) → {rows * k / sum(med['table'][1:]):.1f} (whole-table cache); the official Jev API did {jev_rate[name]:.0f} requests per second cold on similar data (concurrency 16, phase 6's reference)")
        # Did the answers change?
        hit = pooled([compare(answers_of(name, "off", rep, j), answers_of(name, "table", rep, j)) for rep in reps for j in range(2, len(questions) + 1)])
        first = pooled([compare(answers_of(name, "off", rep, 1), answers_of(name, "table", rep, 1)) for rep in reps])
        again = {arm: pooled([compare(answers_of(name, arm, 1, j), answers_of(name, arm, rep, j)) for rep in reps if rep > 1 for j in range(1, len(questions) + 1)]) for arm in recs} if len(reps) > 1 else {}
        print(f"  answers: questions served from the cache (from question 2 on) agree with cache off on {hit['agree']:.2%} ({hit['n']} answers), mean probability difference {hit['mean_gap']:.4f}, max {hit['max_gap']:.4f}; question 1 {first['agree']:.2%}")
        if again:
            print("  the same arm run again, identical judgements: " + ", ".join(f"{'cache off' if arm == 'off' else 'whole-table cache'} {again[arm]['agree']:.2%} (max probability difference {again[arm]['max_gap']:.4f})" for arm in again))
        result["sets"][name] = {"rows": rows, "tokens": tokens, "questions": questions, "median_seconds": med, "seconds": sec, "later_speedup": later, "later_speedup_per_rep": per_rep,
                                "latency_p50": {arm: [mid([r["passes"][j]["latency_p50"] for r in recs[arm]]) for j in range(len(questions))] for arm in recs},
                                "build_overhead": overhead, "spread": list(spread), "mechanism_ok": bool(ok_table and ok_off), "whole_table": WHOLE_TABLES.get(name) == rows,
                                "table_runs": {"cached_states": [b["cached_states"] for b in held], "footprint": [b["footprint"] for b in held], "oom_retries": oom}, "agreement": hit, "agreement_first": first, "repeat": again, "packed_seconds": packed,
                                "default_seconds": [p["seconds"] for p in default["passes"]] if default else None, "failed": failed}
        # Memory: in the first whole-table pass, after each block, how many prefixes and tokens were added and how much more the process used
        t, states, toks = recs["table"][0], 0, 0
        for b in t["passes"][0]["blocks"]:
            states, toks = states + b["rows"], toks + b["tokens"]
            memory_rows.append((name, states, toks, b["footprint"] - t["warmup"]["footprint"]))
        result["sets"][name]["footprint"] = {"loaded": t["footprint_loaded"], "after_warmup": t["warmup"]["footprint"], "end": t["passes"][-1]["blocks"][-1]["footprint"],
                                            "off_end": recs["off"][0]["passes"][-1]["blocks"][-1]["footprint"]}

    if memory_rows:
        X, y = np.array([[s, t] for _, s, t, _ in memory_rows], float), np.array([m for *_, m in memory_rows], float)
        (per_state, per_token), *_ = np.linalg.lstsq(X, y, rcond=None)
        fit = X @ np.array([per_state, per_token])
        r2 = 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"\nmemory model ({len(y)} readings of the first whole-table pass, least squares without intercept): {per_state / 2**20:.1f} MB per cached prefix plus {per_token / 2**10:.1f} KB per token; "
              f"R² = {r2:.4f}, max deviation {np.abs(y - fit).max() / 2**30:.2f} GB")
        print(f"  {'dataset':12s}{'rows':>6s}{'tok/row':>10s}{'cache measured':>15s}{'model':>10s}")
        for name, s in result["sets"].items():
            measured = [m for n, st, _, m in memory_rows if n == name][-1]
            print(f"  {name:12s}{s['rows']:>8d}{s['tokens']:>12.0f}{measured / 2**30:>14.2f} GB{(per_state * s['rows'] + per_token * s['tokens'] * s['rows']) / 2**30:>9.2f} GB")
        print(f"  what fits in the {MEMORY_BUDGET / 2**30:.0f} GB budget:")
        capacity = {}
        for name, n_table in list(WHOLE_TABLES.items()) + [("long", result["sets"].get("long", {}).get("rows"))]:
            if name not in result["sets"] or not n_table:
                continue
            tok = result["sets"][name]["tokens"]
            need, fits = (per_state + per_token * tok) * n_table, int(MEMORY_BUDGET // (per_state + per_token * tok))
            capacity[name] = {"whole_table_rows": n_table, "bytes_needed": float(need), "rows_that_fit": fits, "fits": bool(need <= MEMORY_BUDGET)}
            print(f"    {name:8s}the whole table, {n_table:>5d} prefixes, needs {need / 2**30:>6.1f} GB: {'fits' if need <= MEMORY_BUDGET else 'does not fit'}; at most {fits} prefixes within the budget")
        result["memory"] = {"per_state_bytes": float(per_state), "per_token_bytes": float(per_token), "r2": float(r2), "capacity": capacity}

    if result["sets"]:
        print("\nspeed-up by text length (from question 2 on, cache off ÷ whole-table cache):")
        print(f"  {'dataset':12s}{'tok/row':>10s}{'off s/row':>12s}{'table s/row':>14s}{'ratio':>8s}")
        for name, s in sorted(result["sets"].items(), key=lambda kv: kv[1]["tokens"]):
            k = len(s["questions"]) - 1
            print(f"  {name:12s}{s['tokens']:>12.0f}{sum(s['median_seconds']['off'][1:]) / k / s['rows']:>14.3f}{sum(s['median_seconds']['table'][1:]) / k / s['rows']:>16.3f}{s['later_speedup']:>10.2f}")

    # Are the long-document answers meaningful (reference reading)? The cache-off answers against each document's share of positive reviews
    if "long" in result["sets"]:
        docs = {d["id"]: d["share"] for d in cache_set("long")[0]}
        sanity = long_sanity([{r["id"]: r["answers"]["q"] for r in answers_of("long", "off", 1, j) if r["answers"]} for j in (1, 2, 3)], docs)
        tokens = [r["tokens"] for j in (1, 2, 3) for r in answers_of("long", "off", 1, j) if r["tokens"]]
        print(f"\nare the long-document answers meaningful (reference reading, {sanity['n']} documents, half mostly positive, half not): \"mostly positive?\" accuracy {sanity['noul_accuracy']:.1%}, AUC {sanity['noul_auc']:.3f}; "
              f"multiple-choice accuracy {sanity['choice_accuracy']:.1%}; rank correlation of score with share positive {sanity['score_spearman']:.3f}. Longest request {max(tokens)} tokens (Kev-4B was validated up to 8,192)")
        result["long_sanity"] = sanity | {"max_request_tokens": max(tokens)}
        jev_long(result, docs)

    if all(n in result["sets"] for n in MAIN_SETS) and "memory" in result:
        s, cap = result["sets"], result["memory"]["capacity"]
        agree = {n: s[n]["agreement"]["agree"] >= AGREE_BAR for n in MAIN_SETS}
        long_ok = s["long"]["later_speedup"] >= LONG_BAR and agree["long"] and cap["long"]["fits"]
        short_ok = all(s[n]["later_speedup"] >= SHORT_BAR and agree[n] and cap[n]["fits"] for n in ("movie", "mams"))
        print("\nverdict (fixed beforehand):")
        print(f"  long documents: at least {LONG_BAR:.0f}x faster from question 2 on ({s['long']['later_speedup']:.1f}x), identical judgements at least {AGREE_BAR:.1%} ({s['long']['agreement']['agree']:.2%}), "
              f"the documents' cache within the {MEMORY_BUDGET / 2**30:.0f} GB budget ({cap['long']['bytes_needed'] / 2**30:.1f} GB): {'wonder' if long_ok else 'does not hold'}")
        print("  whole short-text tables: " + "; ".join(f"{n} {s[n]['later_speedup']:.2f}x faster (bar {SHORT_BAR:.0f}x), identical judgements {s[n]['agreement']['agree']:.2%}, "
              f"the whole table needs {cap[n]['bytes_needed'] / 2**30:.0f} GB ({'fits' if cap[n]['fits'] else 'does not fit'})" for n in ("movie", "mams")) + f": {'wonder' if short_ok else 'does not hold'}")
        result["verdict"] = {"long": bool(long_ok), "short": bool(short_ok)}
    out["cache"] = result


# ---------- C: calibration parameters in the database ----------

def scores(p: np.ndarray, y: np.ndarray) -> dict:
    """Rows without an answer (NaN) count as wrong for accuracy and as 0.5 for the other metrics, as in phase 6."""
    q, yes = np.where(np.isnan(p), 0.5, p), y == 1
    decide = (q >= 0.5) & ~np.isnan(p)
    hit, tp = (decide == yes) & ~np.isnan(p), (decide & yes).sum()
    precision, recall = tp / max(decide.sum(), 1), tp / max(yes.sum(), 1)
    return {"acc": float(hit.mean()), "auc": float(R.auc(q, y)), "brier": float(((q - y) ** 2).mean()), "ece_mass": float(R.ece_mass(q, y)), "missing_share": float(np.isnan(p).mean()),
            "precision": float(precision), "recall": float(recall), "f1": float(2 * precision * recall / max(precision + recall, 1e-12))}


def matches(candidate: dict, jev: dict) -> bool:
    """Phase 6's rules A-C, written as in p6_report.py: the difference does not cross the bar, and at most 5% of rows are unanswered."""
    gaps = {k: candidate[k] - jev[k] for k in R.BARS}
    return candidate["missing_share"] <= R.MISSING_BAR and all(gaps[k] >= bar if sign > 0 else gaps[k] <= bar for k, (_, bar, sign) in R.BARS.items())


def fake_jev(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """In a rehearsal there are no Jev judgements to compare with, so a placeholder column is made up."""
    return np.clip(np.where(y == 1, 0.9, 0.05) + rng.normal(0, 0.15, len(y)), 0.01, 0.99)


def section_calibration() -> None:
    print("\n" + "=" * 100 + "\nC  Calibration parameters in the database: two numbers per predicate, fitted on 200 human labels, stored in a table and applied in SQL\n" + "=" * 100)
    if not CAL.DB.exists():
        print("  no results yet")
        return
    con, rng, result = duckdb.connect(str(CAL.DB), read_only=True), np.random.default_rng(SEED), {"datasets": {}}
    have = {name for (name,) in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    params = {(scope, predicate): (a, b, labels, positives) for scope, predicate, a, b, labels, positives in
              con.execute("SELECT scope, predicate, a, b, labels, positives FROM semantic_calibration").fetchall()}
    print(f"the parameter table semantic_calibration has {len(params)} rows (model × predicate × scope). Decision rule as in phase 6: accuracy no more than 2 points below Jev, AUC no more than 0.02 below, Brier no more than 0.02 above; all three mean on par with Jev.")
    for dataset in EM:
        t = CAL.table(dataset)
        if f"judged_test_{t}" not in have:
            print(f"\n{dataset}: no results yet")
            continue
        test, pool = CAL.fetch(con, f"SELECT id, label, p_raw, p_cal, p_transfer FROM judged_test_{t} ORDER BY pos"), CAL.fetch(con, f"SELECT pos, label, p FROM judged_pool_{t} ORDER BY pos")
        y, raw = test["label"].astype(int), test["p_raw"].astype(float)
        key = calibration.predicate_key(*em_question(dataset))
        a, b, n_labels, n_pos = params[dataset, key]
        if REHEARSAL:
            jev = fake_jev(y, rng)
        else:
            got = {r["id"]: r["p"] for r in read(ROOT / f"out/entity_match/{dataset}.test.{JEV_MODEL}.jsonl")}
            jev = np.array([np.nan if got.get(i) is None else got[i] for i in test["id"]], float)
        P = {"jev": jev, "raw": raw, "cal": test["p_cal"].astype(float), "transfer": test["p_transfer"].astype(float)}
        S = {k: scores(v, y) for k, v in P.items()}
        sql_gap = float(np.nanmax(np.abs(P["cal"] - calibration.apply(raw, a, b))))
        index = rng.integers(0, len(y), (BOOT, len(y)))
        boot = {k: [scores(v[ix], y[ix]) for ix in index] for k, v in P.items()}
        ci = lambda k, ref, stat: np.percentile([x[stat] - z[stat] for x, z in zip(boot[k], boot[ref])], [2.5, 97.5])
        verdict = {k: matches(S[k], S["jev"]) for k in ("raw", "cal", "transfer")}
        print(f"\n{dataset}: test set {len(y)} pairs ({y.sum()} matches). Parameters a = {a:.3f}, b = {b:.3f}, from {n_labels} labels ({n_pos} matches); {int(np.isnan(raw).sum())} unanswered rows")
        print(f"  {'source':30s}{'accuracy':>9s}{'AUC':>8s}{'Brier':>8s}{'ECE':>8s}{'prec':>8s}{'recall':>8s}{'F1':>8s}   on par with Jev")
        names = {"jev": "Jev (cached, reference)", "raw": f"{MODEL} raw", "cal": f"{MODEL} + this table's parameters", "transfer": f"{MODEL} + {CAL.TRANSFER}'s parameters"}
        for k, label in names.items():
            s = S[k]
            note = ""
            if k != "jev":
                close = [name for stat, (name, bar, _) in R.BARS.items() if (lambda lo, hi: lo <= bar <= hi)(*ci(k, "jev", stat))]
                note = ("yes" if verdict[k] else "no") + (f" (not separable from the bar: {', '.join(close)})" if close else "")
            print(f"  {label:30s}{s['acc']:>9.1%}{s['auc']:>9.3f}{s['brier']:>9.3f}{s['ece_mass']:>9.3f}{s['precision']:>9.1%}{s['recall']:>9.1%}{s['f1']:>9.3f}   {note}")
        gain = {stat: (S["cal"][stat] - S["raw"][stat], *ci("cal", "raw", stat)) for stat in ("acc", "brier", "f1")}
        print(f"  change from calibration (this table's parameters − raw, 95% interval): accuracy {gain['acc'][0]:+.1%} ({gain['acc'][1]:+.1%} to {gain['acc'][2]:+.1%}), "
              f"Brier {gain['brier'][0]:+.3f} ({gain['brier'][1]:+.3f} to {gain['brier'][2]:+.3f}), F1 {gain['f1'][0]:+.3f} ({gain['f1'][1]:+.3f} to {gain['f1'][2]:+.3f})")
        print(f"  the calibrated probabilities computed in SQL differ from those computed in Python with the same parameters by at most {sql_gap:.1e}")
        if dataset != CAL.TRANSFER and (CAL.TRANSFER, key) not in params:
            print(f"  this question has no parameters on {CAL.TRANSFER} (another question is asked there), so the transfer column returns p unchanged by design: identical to the raw probabilities {np.array_equal(P['transfer'], raw, equal_nan=True)}")
        # How many labels are enough, and does another set of labels change things? Redraw from the 1,000-pair label pool
        ok, sizes = ~np.isnan(pool["p"].astype(float)), {}
        pool_p, pool_y = pool["p"].astype(float)[ok], pool["label"].astype(int)[ok]
        for n in [k for k in SIZES if k <= len(pool_y)]:
            draws = []
            for _ in range(DRAWS):
                pick = rng.choice(len(pool_y), n, replace=False)
                draws.append(scores(calibration.apply(raw, *calibration.platt(pool_p[pick], pool_y[pick])), y))
            sizes[n] = {"acc": float(np.mean([d["acc"] for d in draws])), "brier": float(np.mean([d["brier"] for d in draws])), "f1": float(np.mean([d["f1"] for d in draws])),
                        "match_share": float(np.mean([matches(d, S["jev"]) for d in draws]))}
        print("  number of labels (200 random draws each, from the label pool): " + "; ".join(f"{n} labels → accuracy {v['acc']:.1%}, Brier {v['brier']:.3f}, on par with Jev in {v['match_share']:.0%}" for n, v in sizes.items()))
        result["datasets"][dataset] = {"n": int(len(y)), "positives": int(y.sum()), "a": a, "b": b, "scores": S, "verdict": verdict, "gain": {k: list(map(float, v)) for k, v in gain.items()},
                                       "sql_gap": sql_gap, "sizes": sizes, "missing": int(np.isnan(raw).sum())}
        if dataset == CAL.TRANSFER and not REHEARSAL and (path := RESULTS / "p6" / f"em.{MODEL}.jsonl").exists():
            old = {r["id"]: r["p"] for r in read(path)}
            before = np.array([np.nan if old.get(i) is None else old[i] for i in test["id"]], float)
            both = ~np.isnan(before) & ~np.isnan(raw)
            print(f"  compared with the same judgements collected in phase 6 (another run, another collection path): identical judgements {((before >= 0.5) == (raw >= 0.5))[both].mean():.2%}, max probability difference {np.abs(before - raw)[both].max():.4f}")

    fresh = [d for d in FRESH if d in result["datasets"]]
    if len(fresh) == len(FRESH):
        v = {d: result["datasets"][d]["verdict"] for d in fresh}
        holds = all(v[d]["cal"] for d in fresh) and not all(v[d]["raw"] for d in fresh)
        print("\nverdict (fixed beforehand, on the three datasets Kev had never answered):")
        for d in fresh:
            print(f"  {d:16s}raw {'on par with Jev' if v[d]['raw'] else 'behind Jev'} → with this table's parameters from {LABELS} labels {'on par with Jev' if v[d]['cal'] else 'behind Jev'}")
        reason = "after calibration all three are on par with Jev, and at least one was behind raw" if holds else ("already on par with Jev raw, calibration not needed" if all(v[d]["raw"] for d in fresh) else "some dataset is still behind Jev after calibration")
        print(f"  calibration parameters in the database: {'wonder' if holds else 'does not hold'} ({reason})")
        result["verdict"] = {"holds": bool(holds), "reason": reason}

    if "replay_movie" in have and "replay_mams" in have:
        print(f"\nreplay (data already seen in phase 6, not a new test): one parameter set per predicate, fitted on the labels of the first {CAL.REPLAY_FIT} rows in fixed order, evaluated on the rest")
        result["replay"] = {}
        for scope in ("movie", "mams"):
            df = con.execute(f"SELECT predicate, label, p_raw, p_cal FROM replay_{scope} WHERE pos >= {CAL.REPLAY_FIT}").fetchdf()
            # Does each predicate use its own parameters in SQL? Recompute in Python from the stored parameters to compare
            stored = {name: params[scope, calibration.predicate_key(q)][:2] for name, q in con.execute(f"SELECT DISTINCT predicate, question FROM replay_{scope}_raw").fetchall()}
            sql_gap = max(float(np.nanmax(np.abs(g["p_cal"].to_numpy(float) - calibration.apply(g["p_raw"].to_numpy(float), *stored[name])))) for name, g in df.groupby("predicate"))
            fit = con.execute(f"SELECT label, p AS p_raw FROM replay_{scope}_raw WHERE pos < {CAL.REPLAY_FIT}").fetchdf()
            a, b = calibration.platt(fit["p_raw"].to_numpy(float), fit["label"].to_numpy())
            df["p_pooled"] = calibration.apply(df["p_raw"].to_numpy(float), a, b)
            per = {col: np.mean([[scores(g[col].to_numpy(float), g["label"].to_numpy())[k] for k in ("acc", "brier", "ece_mass")] for _, g in df.groupby("predicate")], axis=0) for col in ("p_raw", "p_cal", "p_pooled")}
            n_pred = df["predicate"].nunique()
            print(f"  {scope:6s}{n_pred:>3d} predicates, {len(df) // n_pred} rows evaluated: " + "; ".join(f"{label} accuracy {v[0]:.1%}, Brier {v[1]:.3f}, ECE {v[2]:.3f}" for label, v in
                  zip(("raw", "per-predicate parameters", "shared parameters for all predicates"), per.values()) if not (n_pred == 1 and label.startswith("shared"))))
            print(f"        the calibrated probabilities computed in SQL differ from those computed in Python with each predicate's own parameters by at most {sql_gap:.1e}")
            result["replay"][scope] = {k: list(map(float, v)) for k, v in per.items()} | {"sql_gap": sql_gap}
    out["calibration"] = result


if __name__ == "__main__":
    if REHEARSAL:
        print("[REHEARSAL] the numbers below come from made-up rows and only show that the steps run; they mean nothing.\n")
    section_sql()
    section_cache()
    section_calibration()
    jev_runs = SHARED / "jev" / "runs.jsonl"
    spent = sum(r["spent"] for r in read(jev_runs)) if jev_runs.exists() else 0.0
    print(f"\ncost: local models are free. Jev's long-document reference ${spent:.4f} (official API); no other paid API was called in this phase." if jev_runs.exists()
          else "\ncost: local models are free; no paid API was called in this phase.")
    out["cost_usd"] = spent
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
