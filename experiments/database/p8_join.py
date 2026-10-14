"""Phase 8: how to ask a semantic join. A left record has k candidates: ask pair by pair (k requests), or put them into one request?

Questions and reading rules: experiments/database/protocol/p8.txt. The whole run is one command, experiments/database/p8_run.sh, which calls the subcommands below.

The data are the two datasets of experiments/table_match/ (two raw tables with the complete human mapping); the candidates are the top 10 found by full-text search back then, with the same question:
abt-buy        1,081 × 1,092 products, nearly one-to-one
dblp-scholar   2,616 × 64,263 publications, a left record often has several matches

Four question forms. The last three use one request per left record:
pair   pairwise: both records in the state, one yes/no question. Exactly what experiments/table_match/ did
Q      candidates in the questions: only the left record in the state, k yes/no questions, each with one candidate in its instructions
S      candidates in the state: the left record and k candidates in the state, k yes/no questions each pointing at one candidate
C      multiple choice: only the left record in the state, one multiple-choice question with the k candidates plus "none of them" as options. Picks at most one per left record, so it only runs on abt-buy

Order of the candidates inside a request: random (shuffled by hash, used for the primary analysis), rank (search rank), reversed (rank reversed).

    python experiments/database/p8_join.py plan            # no model call: requests per step
    python experiments/database/p8_join.py run jev         # official Jev API, paid (estimate in plan)
    python experiments/database/p8_join.py run kev-4b      # local Kev-4B; start its server first (scripts/open_models.sh serve kev-4b)
    python experiments/database/p8_join.py check           # are the collected files complete?
    python experiments/database/p8_join.py report          # analysis, no model call

With the environment variable P8_REHEARSAL=1 this is a rehearsal: the same steps with every row made up, results under out/database/p8_rehearsal.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from paths import OUT as RESULTS, ROOT

sys.path.insert(0, str(ROOT / "experiments" / "table_match"))
import pipeline as tables  # noqa: E402  data, candidates and question all come from experiments/table_match/

REHEARSAL = os.environ.get("P8_REHEARSAL") == "1"
OUT = RESULTS / ("p8_rehearsal" if REHEARSAL else "p8")
NAMESPACE = "p8.rehearsal:" if REHEARSAL else "p8:"
DATASETS = ("abt-buy", "dblp-scholar")
SINGLE_MATCH = ("abt-buy",)  # multiple choice only counts as a way to run a join on data where a left record has at most one or two matches
K = 10
SUBSET = 12 if REHEARSAL else 400
ORDERS = ("random", "rank", "reversed")
PACKED = ("Q", "S", "C")
# Effect of the number of candidates on latency and judgements: the first SCALE_ROWS rows of the abt-buy subset, concurrency 1
SCALE_ROWS = 6 if REHEARSAL else 80
SCALE_KS = (5, 10, 12) if REHEARSAL else (5, 10, 20, 40)
SOURCES = {"jev": 16, "kev-4b": 16}  # concurrency: 16 measured fastest on the official Jev API; Kev's server is serial, 16 as in phase 6
# The local model runs on the subset only; candidate orders as in p8.txt
KEV_ORDERS = {"Q": ("random", "rank"), "S": ("random", "rank"), "C": ("random", "rank", "reversed")}
BUDGET_USD = 3.0
BOOT, SEED = 2000, 8

CHOICE = {
    "product": "Which of the candidate records in the options describes the same product as `record_a`, i.e. the exact same item that a "
               "shopper could buy from either listing? Same brand and same model or edition counts as the same product; differences in "
               "wording, missing fields, or price do not matter.",
    "publication": "Which of the candidate records in the options refers to the same publication as `record_a`? The same paper has the same "
                   "title and authors, allowing for abbreviations, typos, and differently formatted venue names.",
}
NONE = {
    "product": "None of the candidates is the same product. Each is a different product, including a different model, size, capacity, "
               "color, edition, or version of a similar item.",
    "publication": "None of the candidates is the same publication. Each is a different paper, including different papers by the same "
                   "authors or with similar titles.",
}


def sha(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


# ---------- Data ----------

def made_up(dataset: str) -> tuple[dict, dict, set, dict]:
    """Made-up rehearsal data: the same field names as the real data, with content built below and unrelated to any dataset."""
    rng = random.Random("p8.rehearsal:" + dataset)
    left, right, truth, cands = {}, {}, set(), {}
    for i in range(12):
        lid, depth = f"L{i}", 12
        hits = set() if i % 5 == 4 else {rng.randrange(depth)} | ({rng.randrange(depth)} if dataset == "dblp-scholar" and i % 3 == 0 else set())
        if dataset == "abt-buy":
            brand, item, n = rng.choice(["Norvale", "Kestrel", "Ombra", "Tavish"]), rng.choice(["wireless optical mouse", "USB-C docking station", "portable SSD 1TB", "mechanical keyboard"]), rng.randrange(100, 900)
            left[lid] = {"name": f"{brand} {item} - {brand[:2].upper()}{n}", "description": f"{brand} {item} model {brand[:2].upper()}{n}, black, retail pack with cable and quick-start guide", "price": f"${rng.randrange(20, 300)}.00"}
            for j in range(depth):
                m = n if j in hits else n + j + 1
                right[f"R{i}-{j}"] = {"name": f"{brand} {brand[:2].upper()}-{m} {item} (black)", "manufacturer": brand, "price": f"${rng.randrange(20, 300)}.99"}
        else:
            topic, year = rng.choice(["Adaptive Indexing", "Approximate Joins", "Stream Windows", "Learned Sketches"]), rng.randrange(1994, 2004)
            authors = f"{rng.choice(['A. Norvale', 'B. Kestrel', 'C. Ombra'])}, {rng.choice(['D. Tavish', 'E. Lindqvist'])}"
            left[lid] = {"title": f"{topic} for Column Stores: Method {i}", "authors": authors, "venue": "VLDB", "year": str(year)}
            for j in range(depth):
                title = f"{topic} for column stores: method {i}" if j in hits else f"{topic} for Row Stores: Method {i * 20 + j}"
                right[f"R{i}-{j}"] = {"title": title, "authors": authors.replace(". ", " "), "venue": "Very Large Data Bases", "year": str(year if j in hits else year + 1)}
        cands[lid] = [f"R{i}-{j}" for j in range(depth)]
        truth |= {(lid, f"R{i}-{j}") for j in hits}
    return left, right, truth, cands


_DATA: dict[str, tuple] = {}


def load(dataset: str) -> tuple[dict, dict, set, dict]:
    """(left table, right table, true matches, each left record's candidates by rank). Candidates are stored to the deepest level of SCALE_KS; the first K are those experiments/table_match/ found."""
    if dataset not in _DATA:
        if REHEARSAL:
            _DATA[dataset] = made_up(dataset)
        else:
            spec = tables.DATASETS[dataset]
            left, right, truth = tables.load_table(spec["left"]), tables.load_table(spec["right"]), tables.load_truth(spec["truth"])
            cands = tables.load_candidates(dataset, K)
            if dataset == SINGLE_MATCH[0]:
                deep = tables.block(left, right, max(SCALE_KS), spec.get("query"))
                for lid in subset(dataset, left)[:SCALE_ROWS]:
                    if deep[lid][:K] != cands[lid]:
                        raise SystemExit(f"{dataset} {lid}: the top {K} candidates found again differ from those stored by experiments/table_match/")
                    cands[lid] = deep[lid]
            _DATA[dataset] = left, right, truth, cands
    return _DATA[dataset]


def subset(dataset: str, left: dict) -> list[str]:
    return sorted(left, key=lambda lid: sha("p8", dataset, lid))[:SUBSET]


def arrange(dataset: str, lid: str, rids: list[str], order: str) -> list[str]:
    if order == "rank":
        return list(rids)
    if order == "reversed":
        return list(reversed(rids))
    return sorted(rids, key=lambda rid: sha("p8:order", dataset, lid, rid))


# ---------- Requests ----------

def build(dataset: str, form: str, a: dict, bs: list[dict]) -> tuple:
    """One request: (state, questions). bs holds the candidates in their order inside the request; pair has just one."""
    kind = tables.DATASETS[dataset]["kind"]
    q = tables.KINDS[kind]
    names = [f"c{i + 1}" for i in range(len(bs))]
    if form == "pair":
        return {"record_a": a, "record_b": bs[0]}, {"same": q.payload()}
    if form == "Q":
        return {"record_a": a}, {n: {"type": "noul", "instructions": {"question": q.instructions, "record_b": b}, "criteria": q.criteria} for n, b in zip(names, bs)}
    if form == "S":
        return ({"record_a": a, "candidates": dict(zip(names, bs))},
                {n: {"type": "noul", "instructions": q.instructions.replace("`record_b`", f"`candidates.{n}`"), "criteria": q.criteria} for n in names})
    return {"record_a": a}, {"match": {"type": "choice", "instructions": CHOICE[kind], "criteria": {**dict(zip(names, bs)), "none": NONE[kind]}}}


def requests(dataset: str, form: str, order: str, lids: list[str], k: int) -> tuple[list[dict], list[tuple]]:
    left, right, _, cands = load(dataset)
    rows, items = [], []
    for lid in lids:
        rids = arrange(dataset, lid, cands[lid][:k], order)
        for group in ([[r] for r in rids] if form == "pair" else [rids] if rids else []):
            rows.append({"lid": lid, "rids": group})
            items.append(build(dataset, form, left[lid], [right[r] for r in group]))
    return rows, items


def file(source: str, step: dict, ext: str) -> Path:
    """Result file of one step: ext is jsonl (one line per request) or run.json (the step's summary)."""
    return OUT / source / f"{step['dataset']}.{step['form']}.{step['order']}.{step['scope']}.k{step['k']}.{ext}"


def steps(source: str) -> list[dict]:
    """Every step of one source, in execution order. scope: sub (subset), full (whole table), scale (number of candidates)."""
    plan = []
    for dataset in DATASETS:
        forms = [f for f in PACKED if f != "C" or dataset in SINGLE_MATCH]
        # On the subset, pairwise first and then the three packed forms, all asked fresh: the cost bar uses this stretch's time and tokens
        plan.append({"dataset": dataset, "form": "pair", "order": "rank", "scope": "sub", "k": K, "fresh": True})
        plan += [{"dataset": dataset, "form": f, "order": "random", "scope": "sub", "k": K} for f in forms]
        for order in ORDERS[1:]:
            plan += [{"dataset": dataset, "form": f, "order": order, "scope": "sub", "k": K} for f in forms if source == "jev" or order in KEV_ORDERS[f]]
        if source == "jev":
            plan += [{"dataset": dataset, "form": f, "order": "random", "scope": "full", "k": K} for f in forms]
    if source == "jev":
        dataset = SINGLE_MATCH[0]
        plan.append({"dataset": dataset, "form": "pair", "order": "random", "scope": "scale", "k": 1})
        plan += [{"dataset": dataset, "form": f, "order": "random", "scope": "scale", "k": k} for k in SCALE_KS for f in PACKED]
    return plan


def lids_of(step: dict) -> list[str]:
    left = load(step["dataset"])[0]
    return {"sub": subset(step["dataset"], left), "full": list(left), "scale": subset(step["dataset"], left)[:SCALE_ROWS]}[step["scope"]]


def namespace(source: str, step: dict) -> str:
    # The pairwise requests are identical to those experiments/table_match/ sent to Jev back then, so a separate namespace ensures they are asked fresh today; the candidate-count stretch is asked fresh too
    if step["scope"] == "scale":
        return NAMESPACE + "scale:"
    return NAMESPACE + ("fresh:" if step.get("fresh") else "")


# ---------- Collection ----------

def collect(backend, source: str, step: dict) -> dict:
    done = json.loads(file(source, step, "run.json").read_text()) if file(source, step, "run.json").exists() else None
    if done and not done["errors"]:
        return done
    rows, items = requests(step["dataset"], step["form"], step["order"], lids_of(step), step["k"])
    backend.namespace = namespace(source, step)
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
                got = a["answers"]
                if step["form"] == "C":
                    out |= {"choice": got["match"]["choice"], "probs": got["match"]["probabilities"], "confidence": got["match"].get("confidence")}
                else:
                    out["p"] = [got[name]["noul"] for name in questions]
                out |= {"tokens": a["usage"]["input_tokens"], "latency": round(a["_latency"], 4), "cached": a["_cached"]}
            f.write(json.dumps(out) + "\n")
    summary = {**{k: step[k] for k in ("dataset", "form", "order", "scope", "k")}, "source": source, "model": backend.model,
               "left_records": len({r["lid"] for r in rows}), "pairs": sum(len(r["rids"]) for r in rows), "requests": m.calls, "fresh": m.fresh,
               "errors": m.errors, "seconds": round(seconds, 2), "concurrency": backend.concurrency, "input_tokens": m.input_tokens,
               "cost_usd": round(m.cost, 5), "spent_usd": round(backend.spent - spent, 5), "latency_p50": m.latency(0.5)}
    file(source, step, "run.json").write_text(json.dumps(summary, ensure_ascii=False) + "\n")
    print(summary, flush=True)
    return summary


def spent_so_far() -> float:
    return sum(json.loads(p.read_text())["spent_usd"] for p in (OUT / "jev").glob("*.run.json")) if (OUT / "jev").exists() else 0.0


def run(source: str) -> None:
    from kevdb.cli import load_dotenv
    from kevdb import jev_backend, open_backend

    load_dotenv()
    cache = OUT / f"cache.{source}.sqlite"
    OUT.mkdir(parents=True, exist_ok=True)
    make = jev_backend if source == "jev" else lambda **kw: open_backend(source, **kw)
    # The candidate-count stretch measures single-request latency at concurrency 1, so the two backends run one after the other
    for scale in (False, True):
        todo = [s for s in steps(source) if (s["scope"] == "scale") == scale]
        if not todo:
            continue
        concurrency = 1 if scale else SOURCES[source]
        backend = make(concurrency=concurrency, cache_path=cache)
        backend.concurrency = concurrency
        for step in todo:
            if source == "jev" and spent_so_far() > BUDGET_USD:
                raise SystemExit(f"Jev has spent ${spent_so_far():.2f}, over the ${BUDGET_USD:.2f} fixed beforehand; stopping collection")
            collect(backend, source, step)
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
        total = 0
        print(f"== {source}")
        for step in steps(source):
            rows, _ = requests(step["dataset"], step["form"], step["order"], lids_of(step), step["k"])
            total += len(rows)
            print(f"   {step['dataset']:<13}{step['form']:<5}{step['order']:<9}{step['scope']:<6}k={step['k']:<3} requests {len(rows):>6,}  candidate pairs {sum(len(r['rids']) for r in rows):>6,}")
        print(f"   {total:,} requests in total")


# ---------- Analysis ----------

def read(source: str, dataset: str, form: str, order: str, scope: str, k: int = K) -> tuple[list[dict], dict] | None:
    step = {"dataset": dataset, "form": form, "order": order, "scope": scope, "k": k}
    if not file(source, step, "run.json").exists():
        return None
    return [json.loads(line) for line in file(source, step, "jsonl").read_text().splitlines()], json.loads(file(source, step, "run.json").read_text())


def judgments(rows: list[dict], form: str) -> dict[tuple[str, str], tuple[bool, float | None]]:
    """Each candidate pair -> (judged a match?, probability). Pairs in a request without an answer are never judged a match and have probability None."""
    out = {}
    for r in rows:
        for i, rid in enumerate(r["rids"]):
            if "error" in r:
                out[r["lid"], rid] = (False, None)
            elif form == "C":
                out[r["lid"], rid] = (r["choice"] == f"c{i + 1}", r["probs"].get(f"c{i + 1}"))
            else:
                out[r["lid"], rid] = (r["p"][i] >= 0.5, r["p"][i])
    return out


def reference(source: str, dataset: str, scope: str) -> dict | None:
    """Pairwise judgements. Jev's whole table uses what experiments/table_match/ collected on 5 October (same model version, same requests); the subset uses today's fresh answers."""
    if source == "jev" and scope == "full" and not REHEARSAL:
        _, _, _, cands = load(dataset)
        wanted = {f"{lid}|{rid}" for lid, rids in cands.items() for rid in rids[:K]}
        rows = map(json.loads, (ROOT / "out" / "table_match" / f"{dataset}.jev-1.13.0.jsonl").read_text().splitlines())
        return {tuple(r["id"].split("|")): (r["p"] >= 0.5, r["p"]) for r in rows if r["id"] in wanted}
    got = read(source, dataset, "pair", "rank", "sub")
    return judgments(got[0], "pair") if got else None


def tally(judged: dict, dataset: str, lids: list[str]) -> np.ndarray:
    """One row per left record: correct matches, wrong matches, number of true matches (true matches outside the candidates count too, so recall is end to end)."""
    truth = load(dataset)[2]
    index = {lid: i for i, lid in enumerate(lids)}
    out = np.zeros((len(lids), 3))
    for lid, _ in truth:
        if lid in index:
            out[index[lid], 2] += 1
    for (lid, rid), (yes, _) in judged.items():
        if yes and lid in index:
            out[index[lid], 0 if (lid, rid) in truth else 1] += 1
    return out


def prf(t: np.ndarray) -> tuple[float, float, float]:
    """The last axis of t is (correct, wrong, true matches); the leading axes are already summed."""
    tp, fp, n = t[..., 0], t[..., 1], t[..., 2]
    precision, recall = tp / np.maximum(tp + fp, 1), tp / np.maximum(n, 1)
    return precision, recall, 2 * precision * recall / np.maximum(precision + recall, 1e-12)


def f1_difference(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """F1(a) - F1(b) (points) with a 95% interval from resampling left records."""
    draws = np.random.default_rng(SEED).integers(0, len(a), (BOOT, len(a)))
    diff = (prf(a[draws].sum(1))[2] - prf(b[draws].sum(1))[2]) * 100
    return float((prf(a.sum(0))[2] - prf(b.sum(0))[2]) * 100), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def agreement(a: dict, b: dict) -> float:
    keys = a.keys() & b.keys()
    return sum(a[k][0] == b[k][0] for k in keys) / max(len(keys), 1)


def quality(judged: dict, dataset: str) -> dict:
    """Readings at the candidate-pair level: Brier, equal-mass calibration error, and the share of errors within the least certain 30% of candidate pairs."""
    truth = load(dataset)[2]
    rows = [(p, pair in truth) for pair, (_, p) in judged.items() if p is not None]
    p, y = np.array([r[0] for r in rows]), np.array([r[1] for r in rows], dtype=float)
    bins = np.array_split(np.argsort(p, kind="stable"), 10)
    wrong = (p >= 0.5) != (y == 1)
    unsure = np.argsort(np.abs(p - 0.5), kind="stable")[: int(round(0.3 * len(p)))]
    return {"brier": float(np.mean((p - y) ** 2)), "ece": float(sum(len(b) * abs(p[b].mean() - y[b].mean()) for b in bins) / len(p)),
            "errors": int(wrong.sum()), "errors_in_unsure_30": float(wrong[unsure].sum() / max(wrong.sum(), 1))}


def report() -> None:
    results: dict = {"lines": {}}
    say = print
    say("Phase 8: how to ask a semantic join" + (" (rehearsal, every row made up)" if REHEARSAL else ""))
    say("Bars: quality = packed F1 no more than 1.0 point below pairwise; cost = input tokens and time on the subset both at most 50% of pairwise; "
        "order = pairwise agreement between the three orders ≥ 99.0% and F1 within 1.0 point. A form holds only if all three pass on every dataset it applies to.")
    for source in SOURCES:
        say(f"\n==================== {source} ====================")
        verdicts: dict[str, list[bool]] = {f: [] for f in PACKED}
        for dataset in DATASETS:
            left, _, truth, cands = load(dataset)
            sub = subset(dataset, left)
            scope = "full" if source == "jev" else "sub"
            lids = list(left) if scope == "full" else sub
            ref, ref_sub, pair_run = reference(source, dataset, scope), reference(source, dataset, "sub"), read(source, dataset, "pair", "rank", "sub")
            if ref is None or pair_run is None:
                say(f"\n-- {dataset}: no pairwise results, skipped")
                continue
            kept = sum(rid in cands[lid][:K] for lid, rid in truth if lid in set(lids))
            say(f"\n-- {dataset} ({'whole table' if scope == 'full' else 'subset'} {len(lids):,} left records, {sum(len(cands[l][:K]) for l in lids):,} candidate pairs, "
                f"{sum(l in set(lids) for l, _ in truth):,} true matches, {kept:,} of them among the candidates)")
            base = tally(ref, dataset, lids)
            p0, r0, f0 = prf(base.sum(0))
            say(f"   {'form':<8}{'prec':>9}{'recall':>9}{'F1':>8}{'F1 diff vs pairwise (95% CI)':>30}{'agree w/ pairwise':>18}{'mean p diff':>12}")
            say(f"   {'pairwise':<8}{p0:>9.1%}{r0:>9.1%}{f0:>8.1%}")
            if source == "jev" and not REHEARSAL:
                today = agreement(ref_sub, {k: v for k, v in reference(source, dataset, "full").items() if k in ref_sub})
                say(f"   control: today's fresh pairwise answers agree with the cached ones of 5 October on {today:.2%} (subset, {len(ref_sub):,} pairs)")
                results.setdefault("control", {})[dataset] = today
            for form in PACKED:
                got = read(source, dataset, form, "random", scope)
                if got is None:
                    continue
                judged = judgments(got[0], form)
                t = tally(judged, dataset, lids)
                p, r, f = prf(t.sum(0))
                d, lo, hi = f1_difference(t, base)
                gaps = [abs(judged[k][1] - ref[k][1]) for k in judged if k in ref and judged[k][1] is not None]
                say(f"   {form:<10}{p:>9.1%}{r:>9.1%}{f:>8.1%}{d:>+14.1f} ({lo:+.1f} to {hi:+.1f}){agreement(judged, ref):>14.2%}{np.mean(gaps):>12.3f}")
                # Cost: the fresh pass on the subset
                run_sub, pr = read(source, dataset, form, "random", "sub")[1], pair_run[1]
                clean = run_sub["fresh"] == run_sub["requests"] and pr["fresh"] == pr["requests"]
                # A pass with cache hits has no comparable time (this happens when finished steps are re-run); then the cost bar counts as failed
                tokens, seconds = run_sub["input_tokens"] / pr["input_tokens"], run_sub["seconds"] / pr["seconds"] if clean else float("nan")
                # Order
                by_order = {o: judgments(x[0], form) for o in ORDERS if (x := read(source, dataset, form, o, "sub"))}
                f1s = {o: float(prf(tally(j, dataset, sub).sum(0))[2]) for o, j in by_order.items()}
                agree = {f"{a}/{b}": agreement(by_order[a], by_order[b]) for i, a in enumerate(by_order) for b in list(by_order)[i + 1:]}
                l1, l2 = d >= -1.0, tokens <= 0.5 and seconds <= 0.5
                l3 = len(by_order) > 1 and min(agree.values()) >= 0.99 and (max(f1s.values()) - min(f1s.values())) * 100 <= 1.0
                verdicts[form].append(l1 and l2 and l3)
                results["lines"][f"{source}|{dataset}|{form}"] = {
                    "precision": float(p), "recall": float(r), "f1": float(f), "pair_f1": float(f0), "f1_diff": d, "f1_diff_ci": [lo, hi],
                    "agreement_with_pair": agreement(judged, ref), "tokens_ratio": tokens, "seconds_ratio": seconds, "timing_all_fresh": clean,
                    "requests": run_sub["requests"], "pair_requests": pr["requests"], "seconds": run_sub["seconds"], "pair_seconds": pr["seconds"],
                    "latency_p50": run_sub["latency_p50"], "pair_latency_p50": pr["latency_p50"], "f1_by_order": f1s, "agreement_between_orders": agree,
                    "quality": quality(judged, dataset), "pair_quality": quality(ref, dataset), "L1": bool(l1), "L2": bool(l2), "L3": bool(l3),
                    "L1_separable": not lo <= -1.0 <= hi}
            say("   cost (subset, fresh; pairwise = 1) and order (subset)")
            say(f"   {'form':<8}{'requests':>10}{'input tok':>12}{'time':>9}{'latency':>12}   {'F1 per order':<34}{'lowest pairwise agreement':>26}   quality / cost / order")
            pr = pair_run[1]
            say(f"   {'pairwise':<8}{pr['requests']:>10,}{pr['input_tokens']:>12,}{pr['seconds']:>9.1f}s{pr['latency_p50']:>11.2f}s")
            for form in PACKED:
                x = results["lines"].get(f"{source}|{dataset}|{form}")
                if not x:
                    continue
                orders = "  ".join(f"{o} {v:.1%}" for o, v in x["f1_by_order"].items())
                marks = " / ".join("pass" if x[k] else "fail" for k in ("L1", "L2", "L3"))
                note = "" if x["timing_all_fresh"] else " (time includes cache hits, not comparable)"
                say(f"   {form:<10}{x['requests']:>10,}{x['tokens_ratio']:>12.0%}{x['seconds_ratio']:>9.0%}{x['latency_p50']:>12.2f}s   {orders:<38}"
                    f"{min(x['agreement_between_orders'].values(), default=float('nan')):>14.2%}   {marks}{note}"
                    + ("" if x["L1_separable"] else "; the quality bar is inside the interval"))
            say("   candidate-pair level: Brier / calibration error / wrong judgements / share of them in the least certain 30%")
            for form in ("pair", *PACKED):
                x = results["lines"].get(f"{source}|{dataset}|{form}", {}).get("quality") if form != "pair" else quality(ref, dataset)
                if x:
                    say(f"   {('pairwise' if form == 'pair' else form):<8}{x['brier']:>8.4f}{x['ece']:>9.4f}{x['errors']:>7,}{x['errors_in_unsure_30']:>9.0%}")
            position_bias(source, dataset, say, results)
        say("\n   verdict: " + "; ".join(f"{f} {'holds' if v and all(v) else 'does not hold' if v else 'no results'}" for f, v in verdicts.items()))
        results[f"verdict|{source}"] = {f: bool(v and all(v)) for f, v in verdicts.items()}
    scaling(say, results)
    spent = spent_so_far()
    say(f"\ncost: Jev ${spent:.3f}; local models are free.")
    results["spent_usd"] = spent
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n")


def position_bias(source: str, dataset: str, say, results: dict) -> None:
    """Multiple choice: where the chosen position falls relative to the true match. Only left records with exactly one true match among the candidates."""
    if dataset not in SINGLE_MATCH:
        return
    left, _, truth, cands = load(dataset)
    sub = subset(dataset, left)
    one = [lid for lid in sub if sum((lid, r) in truth for r in cands[lid][:K]) == 1]
    none = [lid for lid in sub if not any((lid, r) in truth for r in cands[lid][:K])]
    for order in ORDERS:
        got = read(source, dataset, "C", order, "sub")
        if got is None:
            continue
        rows = {r["lid"]: r for r in got[0] if "error" not in r}
        where = lambda lid: next(i for i, r in enumerate(rows[lid]["rids"]) if (lid, r) in truth)
        picked = lambda lid: None if rows[lid]["choice"] == "none" else int(rows[lid]["choice"][1:]) - 1
        ok = [lid for lid in one if lid in rows]
        first, last = [l for l in ok if where(l) < 3], [l for l in ok if where(l) >= len(rows[l]["rids"]) - 3]
        acc = lambda ls: sum(picked(l) == where(l) for l in ls) / max(len(ls), 1)
        sure = [l for l in rows if (rows[l]["confidence"] or 0) >= 0.9]
        right = lambda l: (picked(l) is None and l in none) or (l in one and picked(l) == where(l))
        scored = [l for l in rows if l in one or l in none]
        out = {"exactly_one": len(ok), "accuracy": acc(ok), "true_in_first_3": len(first), "accuracy_first_3": acc(first), "true_in_last_3": len(last),
               "accuracy_last_3": acc(last), "picked_first_3": sum(picked(l) is not None and picked(l) < 3 for l in rows) / max(len(rows), 1),
               "picked_none": sum(picked(l) is None for l in rows) / max(len(rows), 1), "no_match_in_candidates": len(none),
               "none_when_no_match": sum(picked(l) is None for l in none if l in rows) / max(len(none), 1),
               "confident_share": len(sure) / max(len(rows), 1), "accuracy_confident": sum(right(l) for l in sure if l in scored) / max(sum(l in scored for l in sure), 1),
               "accuracy_not_confident": sum(right(l) for l in scored if l not in sure) / max(sum(l not in sure for l in scored), 1)}
        results.setdefault("position", {})[f"{source}|{dataset}|{order}"] = out
        say(f"   multiple choice, {order:<9}{out['exactly_one']} records with exactly one true match among the candidates: {out['accuracy']:.1%} chosen right; true match in the first 3 positions ({out['true_in_first_3']}): "
            f"{out['accuracy_first_3']:.1%}, in the last 3 ({out['true_in_last_3']}): {out['accuracy_last_3']:.1%}; first 3 positions chosen {out['picked_first_3']:.1%}, "
            f"\"none of them\" chosen {out['picked_none']:.1%} (chosen in {out['none_when_no_match']:.0%} of the {out['no_match_in_candidates']} records with no true match among the candidates); "
            f"confidence ≥0.9 for {out['confident_share']:.1%}, accuracy {out['accuracy_confident']:.1%} there, {out['accuracy_not_confident']:.1%} elsewhere")


def scaling(say, results: dict) -> None:
    dataset = SINGLE_MATCH[0]
    pair = read("jev", dataset, "pair", "random", "scale", 1)
    if pair is None:
        return
    say(f"\n==================== Effect of the number of candidates (jev, first {SCALE_ROWS} rows of the {dataset} subset, concurrency 1, all asked fresh) ====================")
    say(f"   {'form':<8}{'k':>4}{'median latency':>17}{'input tokens per request':>26}{'tokens per pair':>16}{'top-5 judgements same as at k=5':>34}")
    say(f"   {'pairwise':<8}{1:>4}{pair[1]['latency_p50']:>17.3f}s{pair[1]['input_tokens'] / pair[1]['requests']:>18,.0f}{pair[1]['input_tokens'] / pair[1]['pairs']:>14,.0f}")
    _, _, _, cands = load(dataset)
    for form in PACKED:
        first = None
        for k in SCALE_KS:
            got = read("jev", dataset, form, "random", "scale", k)
            if got is None:
                continue
            judged = {key: v for key, v in judgments(got[0], form).items() if key[1] in cands[key[0]][:5]}
            first = first or judged
            same = agreement(judged, first)
            results.setdefault("scaling", {})[f"{form}|{k}"] = {"latency_p50": got[1]["latency_p50"], "tokens_per_request": got[1]["input_tokens"] / got[1]["requests"],
                                                                 "tokens_per_pair": got[1]["input_tokens"] / got[1]["pairs"], "top5_agreement_with_smallest_k": same}
            say(f"   {form:<10}{k:>4}{got[1]['latency_p50']:>17.3f}s{got[1]['input_tokens'] / got[1]['requests']:>18,.0f}{got[1]['input_tokens'] / got[1]['pairs']:>14,.0f}{same:>26.2%}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"] and sys.argv[2:] and all(s in SOURCES for s in sys.argv[2:]):
        for s in sys.argv[2:]:
            run(s)
    elif sys.argv[1:] == ["plan"]:
        plan()
    elif sys.argv[1:] == ["check"]:
        sys.exit(1 if check() else 0)
    elif sys.argv[1:] == ["report"]:
        report()
    else:
        sys.exit(__doc__)
