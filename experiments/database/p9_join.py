"""Phase 9: does phase 8's packed form carry over to joins other than record matching? Questions and reading rules: experiments/database/protocol/p9.txt.
The whole run is one command, experiments/database/p9_run.sh, which calls the subcommands below.

The data are the 128 reviews of the one movie used by SemBench's three pairwise movie queries (Q5, Q6, Q7), 8,128 pairs, with phase 1's questions verbatim:
same       do the two reviews have the same sentiment? (Q5)
opposite   do the two reviews have opposite sentiment? (Q6, Q7)
The truth comes from each review's human label (positive / negative), as in SemBench's reference SQL.

Two question forms:
pair   pairwise: both reviews in the state, one yes/no question. Exactly phase 1's approach
Q      packed: only the first review in the state, k yes/no questions, each with the other review in its instructions. The form that held for record matching in phase 8

Each review is paired with every later review (higher reviewId, at most 127) and cut into blocks of k; k is 10, 40 or 127 (127 means one request per review).
Order of the partners inside a request: random (shuffled by hash, used for the primary analysis), natural (by reviewId), reversed.

    python experiments/database/p9_join.py plan            # no model call: requests per step
    python experiments/database/p9_join.py probe jev       # is a request with 127 questions (made-up reviews) accepted, and how slow is it?
    python experiments/database/p9_join.py run jev         # official Jev API, paid
    python experiments/database/p9_join.py run kev-4b      # local Kev-4B; start its server first
    python experiments/database/p9_join.py check
    python experiments/database/p9_join.py report          # analysis, no model call

With the environment variable P9_REHEARSAL=1 this is a rehearsal: the same steps on 24 made-up reviews, results under out/database/p9_rehearsal.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from kevdb.cache import cache_key
from paths import OUT as RESULTS, ROOT

REHEARSAL = os.environ.get("P9_REHEARSAL") == "1"
OUT = RESULTS / ("p9_rehearsal" if REHEARSAL else "p9")
NAMESPACE = "p9.rehearsal:" if REHEARSAL else "p9:"
PREDICATES = {
    "same": "Do the two movie reviews `review_1` and `review_2` express the same sentiment, that is, both are positive or both are negative?",
    "opposite": "Do the two movie reviews `review_1` and `review_2` express opposite sentiments, that is, one is positive and the other is negative?",
}
POSITIVE = "Is the movie review in `review` positive?"
KS = (5, 10, 23) if REHEARSAL else (10, 40, 127)
ORDERS = ("random", "natural", "reversed")
SOURCES = {"jev": 16, "kev-4b": 16}
# The local model runs one question, one k and two orders only (its phase 7 pairwise judgements are complete only for opposite)
KEV = {"predicates": ("opposite",), "ks": (10,), "orders": ("random", "natural")}
# Subset on which the local model's pairwise time is measured: this many reviews and all their later partners
TIMING_LEFTS = 24 if REHEARSAL else 16
BUDGET_USD = 2.0
BOOT, SEED = 2000, 9
MOVIE = "ant_man_and_the_wasp_quantumania"
# Caches of the earlier phases that asked the same pairwise requests: (file, namespace + model)
EARLIER = {
    ("jev", "same"): (".jevcache.sqlite", "p1.jev.r1.Q5p:jev-1.13.0"),
    ("jev", "opposite"): (".jevcache.sqlite", "p1.jev.r1.Q7:jev-1.13.0"),
    ("kev-4b", "opposite"): ("out/database/p7/kev-4b/cache.sqlite", "p1.open.default.Q7:jaredpalmer/kev-4b@v1.0"),
}
REWRITE = (".jevcache.sqlite", "p1.jev.r1.Q7r:jev-1.13.0")


def sha(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


# ---------- Data ----------

def made_up(n: int = 24) -> list[tuple[int, str, str]]:
    """Made-up rehearsal reviews, unrelated to any dataset."""
    rng = random.Random("p9.rehearsal")
    rows = []
    for i in range(n):
        good = rng.random() < 0.5
        text = (f"{rng.choice(['A dazzling', 'A heartfelt', 'A nimble'] if good else ['A sluggish', 'A muddled', 'A joyless'])} "
                f"{rng.choice(['space opera', 'family drama', 'heist caper', 'biopic'])} "
                f"{rng.choice(['that earns its final scene', 'anchored by two fine leads'] if good else ['that never finds its footing', 'padded with a needless subplot'])}. "
                f"{rng.choice(['I would gladly see it again.', 'It won me over.'] if good else ['I checked my watch twice.', 'It left me cold.'])} (screening {i})")
        rows.append((i + 1, text, "POSITIVE" if good else "NEGATIVE"))
    return rows


_REVIEWS: list | None = None


def reviews() -> list[tuple[int, str, str]]:
    """(reviewId, text, human label), ordered by reviewId."""
    global _REVIEWS
    if _REVIEWS is None:
        if REHEARSAL:
            _REVIEWS = made_up()
        else:
            import duckdb

            path = ROOT / "experiments/database/data/sembench/files/movie/data/sf_2000/Reviews.csv"
            _REVIEWS = duckdb.sql(f"SELECT DISTINCT reviewId, reviewText, scoreSentiment FROM read_csv('{path}') WHERE id = '{MOVIE}' ORDER BY reviewId").fetchall()
    return _REVIEWS


def pairs() -> list[tuple[int, int]]:
    """Every pair, the first with the smaller reviewId: as in phase 1's SQL (a.reviewId < b.reviewId)."""
    ids = [r[0] for r in reviews()]
    return [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]


def truth(pred: str) -> np.ndarray:
    label = {r[0]: r[2] for r in reviews()}
    same = np.array([label[a] == label[b] for a, b in pairs()])
    return same if pred == "same" else ~same


def arrange(a: int, bs: list[int], order: str) -> list[int]:
    if order == "natural":
        return sorted(bs)
    if order == "reversed":
        return sorted(bs, reverse=True)
    return sorted(bs, key=lambda b: sha("p9:order", str(a), str(b)))


def blocks(order: str, k: int, lefts: list[int] | None = None) -> list[tuple[int, list[int]]]:
    """(first review, its partners in this block). Every pair falls into exactly one block."""
    ids = [r[0] for r in reviews()]
    out = []
    for a in ids if lefts is None else lefts:
        bs = arrange(a, [b for b in ids if b > a], order)
        out += [(a, bs[i:i + k]) for i in range(0, len(bs), k)]
    return out


def timing_lefts() -> list[int]:
    return sorted((r[0] for r in reviews()), key=lambda rid: sha("p9", str(rid)))[:TIMING_LEFTS]


# ---------- Requests ----------

def build(pred: str, form: str, a: str, bs: list[str]) -> tuple:
    """One request: (state, questions). pair has a single b, and its request is identical to the one phase 1's SQL function sent."""
    if form == "pair":
        return {"review_1": a, "review_2": bs[0]}, {"q": {"type": "noul", "instructions": PREDICATES[pred]}}
    return {"review_1": a}, {f"c{i + 1}": {"type": "noul", "instructions": {"question": PREDICATES[pred], "review_2": b}} for i, b in enumerate(bs)}


def requests(step: dict) -> tuple[list[dict], list[tuple]]:
    text = {r[0]: r[1] for r in reviews()}
    lefts = timing_lefts() if step.get("timing") else None
    groups = blocks(step["order"], step["k"], lefts)
    return [{"a": a, "bs": bs} for a, bs in groups], [build(step["pred"], step["form"], text[a], [text[b] for b in bs]) for a, bs in groups]


def steps(source: str) -> list[dict]:
    if source == "jev":
        # Pairwise is asked fresh today in full: it is both the quality reference and the denominator of the cost bar's time and tokens
        plan = [{"pred": p, "form": "pair", "order": "natural", "k": 1} for p in PREDICATES]
        return plan + [{"pred": p, "form": "Q", "order": o, "k": k} for p in PREDICATES for o in ORDERS for k in KS]
    plan = [{"pred": p, "form": "pair", "order": "natural", "k": 1, "timing": True} for p in KEV["predicates"]]
    return plan + [{"pred": p, "form": "Q", "order": o, "k": k} for p in KEV["predicates"] for o in KEV["orders"] for k in (KS[1:2] if REHEARSAL else KEV["ks"])]


def file(source: str, step: dict, ext: str) -> Path:
    return OUT / source / f"{step['pred']}.{step['form']}.{step['order']}.k{step['k']}.{ext}"


# ---------- Collection ----------

def collect(backend, source: str, step: dict) -> dict:
    done = json.loads(file(source, step, "run.json").read_text()) if file(source, step, "run.json").exists() else None
    if done and not done["errors"]:
        return done
    rows, items = requests(step)
    # Each step has its own namespace: reviews with only a few partners left produce identical requests for different k and orders,
    # so with a shared cache a later step would hit an earlier step's answers, its time would not be comparable, and the reordering test would compare an answer with itself
    backend.namespace = NAMESPACE + f"{step['pred']}.{step['form']}.{step['order']}.k{step['k']}:"
    backend.reset()
    spent = backend.spent
    started = time.perf_counter()
    for _ in range(4):
        answers = backend.ask_many(items)
        if not backend.meter.errors:
            break
    seconds = time.perf_counter() - started
    m = backend.meter
    (OUT / source).mkdir(parents=True, exist_ok=True)
    with file(source, step, "jsonl").open("w") as f:
        for row, (_, questions), a in zip(rows, items, answers):
            out = dict(row)
            if "_error" in a:
                out["error"] = a["_error"]
            else:
                out |= {"p": [a["answers"][name]["noul"] for name in questions], "tokens": a["usage"]["input_tokens"], "latency": round(a["_latency"], 4), "cached": a["_cached"]}
            f.write(json.dumps(out) + "\n")
    summary = {**{k: step[k] for k in ("pred", "form", "order", "k")}, "source": source, "model": backend.model, "pairs": sum(len(r["bs"]) for r in rows),
               "requests": m.calls, "fresh": m.fresh, "errors": m.errors, "seconds": round(seconds, 2), "concurrency": SOURCES[source], "input_tokens": m.input_tokens,
               "cost_usd": round(m.cost, 5), "spent_usd": round(backend.spent - spent, 5), "latency_p50": m.latency(0.5), "latency_p95": m.latency(0.95)}
    file(source, step, "run.json").write_text(json.dumps(summary, ensure_ascii=False) + "\n")
    print(summary, flush=True)
    return summary


def spent_so_far() -> float:
    return sum(json.loads(p.read_text())["spent_usd"] for p in (OUT / "jev").glob("*.run.json")) if (OUT / "jev").exists() else 0.0


def backend_for(source: str, concurrency: int | None = None):
    from kevdb.cli import load_dotenv
    from kevdb import jev_backend, open_backend

    load_dotenv()
    OUT.mkdir(parents=True, exist_ok=True)
    kw = {"concurrency": concurrency or SOURCES[source], "cache_path": OUT / f"cache.{source}.sqlite"}
    return jev_backend(**kw) if source == "jev" else open_backend(source, **kw)


def run(source: str) -> None:
    backend = backend_for(source)
    for step in steps(source):
        if source == "jev" and spent_so_far() > BUDGET_USD:
            raise SystemExit(f"Jev has spent ${spent_so_far():.2f}, over the ${BUDGET_USD:.2f} fixed beforehand; stopping collection")
        collect(backend, source, step)
    backend.close()


def probe(source: str) -> None:
    """Put 127 questions in one request (made-up reviews) to see whether it is accepted, how slow it is and how many tokens it bills. Writes no result file."""
    rows = made_up(128)
    backend = backend_for(source, concurrency=1)
    backend.namespace = f"p9.probe.{time.time():.0f}:"
    for k in (10, 40, 127):
        a = backend.ask_many([build("opposite", "Q", rows[0][1], [r[1] for r in rows[1:k + 1]])])[0]
        print(f"{source} k={k}: " + (f"failed {a['_error'][:200]}" if "_error" in a else f"{len(a['answers'])} answers, {a['_latency']:.2f} s, {a['usage']['input_tokens']:,} input tokens"), flush=True)
    print(f"cost ${backend.spent:.4f}")
    backend.close()


def check() -> int:
    bad = 0
    for source in SOURCES:
        for step in steps(source):
            path = file(source, step, "run.json")
            if not path.exists():
                print(f"missing: {path.relative_to(OUT)}")
                bad += 1
            elif (r := json.loads(path.read_text()))["errors"]:
                print(f"{r['errors']} requests without an answer: {path.relative_to(OUT)}")
                bad += 1
    print(f"inventory: {sum(len(steps(s)) for s in SOURCES) - bad} steps complete, {bad} missing or incomplete. Jev has spent ${spent_so_far():.3f}")
    return bad


def plan() -> None:
    for source in SOURCES:
        print(f"== {source}")
        total = 0
        for step in steps(source):
            rows, _ = requests(step)
            total += len(rows)
            print(f"   {step['pred']:<10}{step['form']:<6}{step['order']:<9}k={step['k']:<4} requests {len(rows):>6,}  pairs {sum(len(r['bs']) for r in rows):>6,}")
        print(f"   {total:,} requests in total")


# ---------- Analysis ----------

def read(source: str, step: dict) -> tuple[np.ndarray, dict] | None:
    """This step's probability for each pair (in pairs() order; nan if not asked or unanswered) and its summary."""
    if not file(source, step, "run.json").exists():
        return None
    index = {pair: i for i, pair in enumerate(pairs())}
    p = np.full(len(index), np.nan)
    for line in file(source, step, "jsonl").read_text().splitlines():
        r = json.loads(line)
        if "error" not in r:
            for b, x in zip(r["bs"], r["p"]):
                p[index[r["a"], b]] = x
    return p, json.loads(file(source, step, "run.json").read_text())


def earlier(source: str, pred: str) -> np.ndarray | None:
    """Probabilities earlier phases gave for the same pairwise requests (reads their caches only, never writes)."""
    if REHEARSAL or (source, pred) not in EARLIER:
        return None
    path, prefix = EARLIER[source, pred]
    text = {r[0]: r[1] for r in reviews()}
    db = sqlite3.connect(f"file:{ROOT / path}?mode=ro", uri=True)
    out = []
    for a, b in pairs():
        row = db.execute("SELECT response FROM answers WHERE key = ?", (cache_key(prefix, *build(pred, "pair", text[a], [text[b]])),)).fetchone()
        out.append(json.loads(row[0])["answers"]["q"]["noul"] if row else np.nan)
    return np.array(out)


def rewritten(pred: str) -> np.ndarray | None:
    """Phase 1's rewrite: judge "positive?" once per review, then pair the judgements."""
    if REHEARSAL:
        return None
    db = sqlite3.connect(f"file:{ROOT / REWRITE[0]}?mode=ro", uri=True)
    positive = {}
    for rid, text, _ in reviews():
        row = db.execute("SELECT response FROM answers WHERE key = ?", (cache_key(REWRITE[1], {"review": text}, {"q": {"type": "noul", "instructions": POSITIVE}}),)).fetchone()
        if row is None:
            return None
        positive[rid] = json.loads(row[0])["answers"]["q"]["noul"] >= 0.5
    same = np.array([positive[a] == positive[b] for a, b in pairs()])
    return same if pred == "same" else ~same


def decide(p: np.ndarray) -> np.ndarray:
    return np.nan_to_num(p, nan=0.0) >= 0.5  # pairs without an answer do not count as matches


def prf(yes: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> tuple:
    """Precision, recall, F1. w gives each pair a weight (for resampling) and may hold a batch of weights (one set per row)."""
    w = np.ones(len(y)) if w is None else w
    tp, fp, fn = w @ (yes & y), w @ (yes & ~y), w @ (~yes & y)
    return tp / np.maximum(tp + fp, 1e-12), tp / np.maximum(tp + fn, 1e-12), 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)


def f1_difference(a: np.ndarray, b: np.ndarray, y: np.ndarray, keep: np.ndarray | None = None) -> tuple[float, float, float]:
    """F1(a) - F1(b) (points) with a 95% interval. Pairs are not independent (one review appears in 127 pairs), so the resampling unit is the review:
    each draw takes n reviews with replacement, and a pair's weight is the product of how often each of its two reviews was drawn."""
    ids = [r[0] for r in reviews()]
    at = {rid: i for i, rid in enumerate(ids)}
    ia, ib = np.array([at[x] for x, _ in pairs()]), np.array([at[x] for _, x in pairs()])
    counts = np.random.default_rng(SEED).multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=BOOT).astype(np.float32)
    w = counts[:, ia] * counts[:, ib]
    if keep is not None:
        a, b, y, w = a[keep], b[keep], y[keep], w[:, keep]
    diff = (prf(a, y, w)[2] - prf(b, y, w)[2]) * 100
    return float((prf(a, y)[2] - prf(b, y)[2]) * 100), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def quality(p: np.ndarray, y: np.ndarray) -> dict:
    """Brier, equal-mass calibration error, wrong judgements, and the share of them in the least certain 30% of pairs."""
    ok = ~np.isnan(p)
    p, y = p[ok], y[ok].astype(float)
    bins = np.array_split(np.argsort(p, kind="stable"), 10)
    wrong = (p >= 0.5) != (y == 1)
    unsure = np.argsort(np.abs(p - 0.5), kind="stable")[: int(round(0.3 * len(p)))]
    return {"brier": float(np.mean((p - y) ** 2)), "ece": float(sum(len(b) * abs(p[b].mean() - y[b].mean()) for b in bins) / len(p)),
            "errors": int(wrong.sum()), "errors_in_unsure_30": float(wrong[unsure].sum() / max(wrong.sum(), 1))}


def sembench_q7(yes: np.ndarray) -> float | None:
    """The F1 SemBench's own scoring code gives Q7 (all pairs with opposite sentiment)."""
    if REHEARSAL:
        return None
    import pandas as pd

    import p1_movie

    ev = p1_movie.evaluator()
    system = pd.DataFrame([{"id": MOVIE, "reviewId1": a, "reviewId2": b} for (a, b), hit in zip(pairs(), yes) if hit])
    return float(ev._evaluate_single_query(7, system, p1_movie.ground_truth(ev, 7)).f1_score)


def report() -> None:
    results: dict = {"lines": {}, "verdict": {}}
    say = print
    n = len(reviews())
    say("Phase 9: packing for joins other than record matching" + (" (rehearsal, made-up reviews)" if REHEARSAL else ""))
    say(f"{n} reviews, {len(pairs()):,} pairs; by the truth {int(truth('same').sum()):,} pairs have the same sentiment and {int(truth('opposite').sum()):,} the opposite.")
    say("Bars: quality = packed F1 no more than 1.0 point below pairwise; cost = input tokens and time both at most 50% of pairwise; "
        "order = pairwise agreement between the orders ≥ 99.0% and F1 within 1.0 point. A value of k holds only if all three pass on every question that backend ran.")
    for source in SOURCES:
        say(f"\n==================== {source} ====================")
        preds = [p for p in PREDICATES if file(source, {"pred": p, "form": "pair", "order": "natural", "k": 1}, "run.json").exists()]
        ks = sorted({s["k"] for s in steps(source) if s["form"] == "Q"})
        passed: dict[int, list[bool]] = {k: [] for k in ks}
        for pred in preds:
            y = truth(pred)
            pair_p, pair_run = read(source, {"pred": pred, "form": "pair", "order": "natural", "k": 1})
            before = earlier(source, pred)
            asked = ~np.isnan(pair_p)
            # Quality reference: today's answers where pairwise was asked fresh in full; earlier phases' answers where only part was asked fresh (the local model's timing subset)
            ref = pair_p if asked.all() else before
            say(f"\n-- {pred}: {PREDICATES[pred]}")
            if before is not None:
                same = float(np.mean(decide(pair_p[asked]) == decide(before[asked])))
                say(f"   control: today's fresh pairwise answers ({int(asked.sum()):,} pairs) agree with the earlier phases' cached ones on {same:.2%}")
                results.setdefault("control", {})[f"{source}|{pred}"] = same
            if ref is None:
                say("   no pairwise reference, skipped")
                continue
            p0, r0, f0 = prf(decide(ref), y)
            say(f"   {'form':<10}{'prec':>9}{'recall':>9}{'F1':>8}{'F1 diff vs pairwise (95% CI)':>30}{'agree w/ pairwise':>18}{'mean p diff':>12}{'requests':>9}"
                f"{'input tok':>11}{'time':>7}{'latency p50 / p95':>18}   F1 per order / lowest pairwise agreement   quality / cost / order")
            per_pair = lambda run: (run["input_tokens"] / run["pairs"], run["seconds"] / run["pairs"])
            say(f"   {'pairwise':<10}{p0:>9.1%}{r0:>9.1%}{f0:>8.1%}{'':>44}{pair_run['requests']:>11,}{pair_run['input_tokens']:>12,}{pair_run['seconds']:>8.0f}s"
                f"{pair_run['latency_p50']:>9.2f} / {pair_run['latency_p95']:.2f}s" + ("" if asked.all() else f"   (only {int(asked.sum()):,} pairs asked fresh, to measure time)"))
            q0 = quality(ref, y)
            for k in ks:
                got = read(source, {"pred": pred, "form": "Q", "order": "random", "k": k})
                if got is None:
                    continue
                p, run_ = got
                yes = decide(p)
                pr, rc, f = prf(yes, y)
                d, lo, hi = f1_difference(yes, decide(ref), y)
                clean = run_["fresh"] == run_["requests"] and pair_run["fresh"] == pair_run["requests"]
                tokens = per_pair(run_)[0] / per_pair(pair_run)[0]
                seconds = per_pair(run_)[1] / per_pair(pair_run)[1] if clean else float("nan")
                by_order = {o: decide(x[0]) for o in ORDERS if (x := read(source, {"pred": pred, "form": "Q", "order": o, "k": k}))}
                f1s = {o: float(prf(v, y)[2]) for o, v in by_order.items()}
                agree = {f"{a}/{b}": float(np.mean(by_order[a] == by_order[b])) for i, a in enumerate(by_order) for b in list(by_order)[i + 1:]}
                l1, l2 = d >= -1.0, bool(tokens <= 0.5 and seconds <= 0.5)
                l3 = len(by_order) > 1 and min(agree.values()) >= 0.99 and (max(f1s.values()) - min(f1s.values())) * 100 <= 1.0
                passed[k].append(l1 and l2 and l3)
                gap = float(np.nanmean(np.abs(p - ref)))
                results["lines"][f"{source}|{pred}|{k}"] = {
                    "precision": float(pr), "recall": float(rc), "f1": float(f), "pair_f1": float(f0), "f1_diff": d, "f1_diff_ci": [lo, hi],
                    "agreement_with_pair": float(np.mean(yes == decide(ref))), "mean_abs_gap": gap, "requests": run_["requests"], "tokens_ratio": tokens,
                    "seconds_ratio": seconds, "timing_all_fresh": clean, "seconds": run_["seconds"], "latency_p50": run_["latency_p50"], "latency_p95": run_["latency_p95"],
                    "f1_by_order": f1s, "agreement_between_orders": agree, "quality": quality(p, y), "pair_quality": q0,
                    "L1": bool(l1), "L2": l2, "L3": bool(l3), "L1_separable": not lo <= -1.0 <= hi}
                orders = "  ".join(f"{o} {v:.1%}" for o, v in f1s.items())
                say(f"   {'Q, k=' + str(k):<10}{pr:>9.1%}{rc:>9.1%}{f:>8.1%}{d:>+14.1f} ({lo:+.1f} to {hi:+.1f}){np.mean(yes == decide(ref)):>14.2%}{gap:>12.3f}"
                    f"{run_['requests']:>11,}{tokens:>12.0%}{seconds:>8.0%}{run_['latency_p50']:>9.2f} / {run_['latency_p95']:.2f}s   {orders} / {min(agree.values(), default=float('nan')):.2%}   "
                    + " / ".join("pass" if x else "fail" for x in (l1, l2, l3)) + ("" if clean else " (time includes cache hits, not comparable)") + ("" if not lo <= -1.0 <= hi else "; the quality bar is inside the interval"))
            say("   pair level: Brier / calibration error / wrong judgements / share of them in the least certain 30%")
            say(f"   {'pairwise':<10}{q0['brier']:>8.4f}{q0['ece']:>9.4f}{q0['errors']:>7,}{q0['errors_in_unsure_30']:>9.0%}")
            for k in ks:
                x = results["lines"].get(f"{source}|{pred}|{k}", {}).get("quality")
                if x:
                    say(f"   {'Q, k=' + str(k):<10}{x['brier']:>8.4f}{x['ece']:>9.4f}{x['errors']:>7,}{x['errors_in_unsure_30']:>9.0%}")
            alt = rewritten(pred) if source == "jev" else None
            if alt is not None:
                pr, rc, f = prf(alt, y)
                d, lo, hi = f1_difference(alt, decide(ref), y)
                results.setdefault("rewrite", {})[pred] = {"precision": float(pr), "recall": float(rc), "f1": float(f), "f1_diff_vs_pair": d, "ci": [lo, hi]}
                say(f"   reference, phase 1's rewrite (judge each review's sentiment once, then an ordinary join, {n} requests): {pr:.1%} / {rc:.1%} / {f:.1%}, vs pairwise {d:+.1f} ({lo:+.1f} to {hi:+.1f})")
            if pred == "opposite":
                scores = {"pairwise": sembench_q7(decide(ref))} | {f"Q, k={k}": sembench_q7(decide(x[0])) for k in ks if (x := read(source, {"pred": pred, "form": "Q", "order": "random", "k": k}))}
                if None not in scores.values():
                    results.setdefault("sembench_q7", {})[source] = scores
                    say("   F1 for Q7 from SemBench's own scoring code: " + ", ".join(f"{name} {v:.3f}" for name, v in scores.items()))
        if len(preds) == 2 and source == "jev":
            say("\n-- The two questions are opposites: share of pairs where both get the same answer (both yes or both no)")
            for label, step in [("pairwise", {"form": "pair", "order": "natural", "k": 1})] + [(f"Q, k={k}", {"form": "Q", "order": "random", "k": k}) for k in ks]:
                a, b = (read(source, {"pred": p, **step}) for p in PREDICATES)
                if a and b:
                    clash = float(np.mean(decide(a[0]) == decide(b[0])))
                    results.setdefault("contradiction", {})[label] = clash
                    say(f"   {label:<10}{clash:>7.1%}")
        results["verdict"][source] = {str(k): bool(v and all(v)) for k, v in passed.items()}
        say("\n   verdict: " + "; ".join(f"k={k} {'holds' if v and all(v) else 'does not hold' if v else 'no results'}" for k, v in passed.items()))
    spent = spent_so_far()
    say(f"\ncost: Jev ${spent:.3f}; local models are free.")
    results["spent_usd"] = spent
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"] and sys.argv[2:] and all(s in SOURCES for s in sys.argv[2:]):
        for s in sys.argv[2:]:
            run(s)
    elif sys.argv[1:2] == ["probe"] and sys.argv[2:] and all(s in SOURCES for s in sys.argv[2:]):
        for s in sys.argv[2:]:
            probe(s)
    elif sys.argv[1:] == ["plan"]:
        plan()
    elif sys.argv[1:] == ["check"]:
        sys.exit(1 if check() else 0)
    elif sys.argv[1:] == ["report"]:
        report()
    else:
        sys.exit(__doc__)
