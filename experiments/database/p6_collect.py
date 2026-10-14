"""Phase 6 collection: the open, Jev-compatible models answer the same questions Jev answered. Questions and reading rules: experiments/database/protocol/p6.txt;
execution steps: experiments/database/protocol/p6_plan.md. The whole run is one command, experiments/database/p6_run.sh, which calls the subcommands below.

source        how                                                           cost
kev-4b        local, Kev's own server, port 8009                            free
kev-9b        local, port 8010                                              free
laya          local, laya-serve, port 8011                                  free
laya-multilingual  the same software with the multilingual weights, port 8013   free
clef-flash    local, scripts/clef_server.py, port 8012                     free
jev.fresh     official Jev API, asked again today (control group)          paid, about $0.43 in total

All data are the original rows and questions previously asked of Jev; state and questions are unchanged to the letter:
movie         the 5,000 movie reviews the open models were also asked in phase 2, "positive?"
mams          phase 4's fixed 1,000 restaurant sentences × 21 predicates, one request per question
mams.packed   the same 1,000 sentences, 21 questions in one request
em            the Walmart-Amazon product-matching test set, 2,049 pairs
movie.repeat  the first 200 movie rows, asked again after all other data, to see whether a model answers the same way twice

Local models are installed and served by scripts/open_models.sh (everything in open_models/, not committed, versions pinned).

    python experiments/database/p6_collect.py estimate            # no model call: request counts, tokens Jev billed at the time, cost of the Jev control group
    python experiments/database/p6_collect.py probe kev-4b        # which of concurrency 1, 4, 16 is fastest (on reviews outside the evaluation; answers are not looked at)
    python experiments/database/p6_collect.py run kev-4b          # runs all five datasets; dataset names after it run only those
    python experiments/database/p6_collect.py control             # does the control group pass: do today's Jev and the cached Jev agree?
    python experiments/database/p6_collect.py check               # are all files complete, how many rows have no answer
    python experiments/database/p6_collect.py merge               # fold each source's cache into the main cache

With the environment variable P6_REHEARSAL=1 this is a rehearsal: the same steps, but on a few dozen made-up rows (not evaluation data), with results under out/database/p6_rehearsal.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time

from kevdb import JevClient, Task
from kevdb.cache import Cache, cache_key
from kevdb.cli import load_dotenv
from kevdb.client import JevAPIError

from kevdb import JEV_MODEL, OPEN_MODELS, OpenModelClient, jev_backend, open_backend
from p2_collect import QUESTIONS, merge as merge_caches, sha, texts
from p4_collect import PREDICATES, question, sentences
from paths import OUT as RESULTS, ROOT

REHEARSAL = os.environ.get("P6_REHEARSAL") == "1"
OUT = RESULTS / ("p6_rehearsal" if REHEARSAL else "p6")
NAMESPACE = "p6.rehearsal:" if REHEARSAL else "p6:"
EM = "walmart-amazon"
DATASETS = ("movie", "mams", "mams.packed", "em", "movie.repeat")
N_MOVIE, N_MAMS, N_REPEAT = 5000, 1000, 200
PROBE, PROBE_ROWS = (1, 4, 16), 200
JEV = "jev.fresh"
# The local open models (names with @ are hosted versions, not run this time) plus the Jev control group
SOURCES = tuple(name for name in OPEN_MODELS if "@" not in name) + (JEV,)
JEV_CONCURRENCY = 16  # the official API has its highest measured throughput at 16, as used in every earlier phase
# The cache namespace Jev's original answers were stored under; estimate reads Jev's billed tokens from it
JEV_NAMESPACE = {"movie": "p2:", "mams": "p4:", "mams.packed": "p4:", "em": "", "movie.repeat": "p2:"}
# Control-group bar: agreement between today's Jev and the cached Jev on movie reviews (phase 3 measured 99.45%)
CONTROL_BAR = 0.97


def movie_item(text: str) -> tuple:
    return {"review": text}, {"q": {"type": "noul", "instructions": QUESTIONS["positive"]}}


def em_question() -> dict:
    sys.path.insert(0, str(ROOT / "experiments" / "entity_match"))
    from run import KINDS

    return {"same": KINDS["product"].payload()}


def made_up(dataset: str) -> tuple[list[dict], list[tuple]]:
    """Made-up rehearsal data: the questions are the real ones, the states are sentences built below, unrelated to any dataset."""
    rng = random.Random("rehearsal:" + dataset.split(".")[0])  # movie and movie.repeat, mams and mams.packed must use the same rows
    pick = rng.choice
    if dataset.startswith("movie"):
        rows = [f"{pick(['A sluggish', 'A dazzling', 'An uneven', 'A heartfelt'])} {pick(['thriller', 'family drama', 'space opera', 'biopic'])} "
                f"{pick(['that never finds its footing', 'anchored by two fine leads', 'that earns its final scene', 'padded with a needless subplot'])}. "
                f"{pick(['I checked my watch twice.', 'I would gladly see it again.', 'It left me cold.'])} (screening {i})" for i in range(40)]
        rows = rows[:10] if dataset == "movie.repeat" else rows
        return [{"text": sha(t)} for t in rows], [movie_item(t) for t in rows]
    if dataset.startswith("mams"):
        rows = [f"{pick(['The dumplings', 'Our waiter', 'The patio', 'The wine list'])} {pick(['was wonderful', 'was a letdown', 'took forever', 'felt overpriced'])}, "
                f"{pick(['but the dessert made up for it', 'and the bill was a shock', 'though the room itself is lovely'])} (table {i})." for i in range(10)]
        ids = [{"sid": f"made-up-{i}"} for i in range(len(rows))]
        if dataset == "mams.packed":
            return ids, [({"sentence": t}, {p: question(p) for p in PREDICATES}) for t in rows]
        return ids, [({"sentence": t}, {"q": question(p)}) for t in rows for p in PREDICATES]
    ids, reqs = [], []
    for i in range(30):
        brand, item, n, same = pick(["Norvale", "Kestrel", "Ombra"]), pick(["wireless optical mouse", "USB-C docking station", "portable SSD 1TB"]), rng.randrange(100, 999), rng.random() < 0.5
        a = {"title": f"{brand} {item} model {n} black", "category": "computer accessories", "brand": brand, "modelno": f"{brand[:2].upper()}{n}", "price": f"{rng.randrange(20, 300)}.99"}
        b = {"title": f"{brand} {item} {n if same else n + 1} (black) retail pack", "category": "electronics - general", "brand": brand, "modelno": f"{brand[:2].upper()}-{n if same else n + 1}"}
        ids.append({"id": f"made-up-{i}", "label": int(same)})
        reqs.append(({"record_a": a, "record_b": b}, em_question()))
    return ids, reqs


def requests(dataset: str) -> tuple[list[dict], list[tuple]]:
    """(row id, request) pairs. mams has 21 requests per row, kept together; everything else one per row."""
    if REHEARSAL:
        return made_up(dataset)
    if dataset.startswith("movie"):
        rows = texts(N_MOVIE, "open")[:N_REPEAT if dataset == "movie.repeat" else None]
        return [{"text": sha(t)} for t in rows], [movie_item(t) for t in rows]
    if dataset.startswith("mams"):
        rows = sentences(N_MAMS)
        ids = [{"sid": r["sid"]} for r in rows]
        if dataset == "mams.packed":
            return ids, [({"sentence": r["text"]}, {p: question(p) for p in PREDICATES}) for r in rows]
        return ids, [({"sentence": r["text"]}, {"q": question(p)}) for r in rows for p in PREDICATES]
    sys.path.insert(0, str(ROOT / "experiments" / "entity_match"))
    from run import load_pairs

    task = Task(name="em", input="", questions={}, state_fields=["record_a", "record_b"])
    pairs = load_pairs(EM, "test")
    return [{"id": p["id"], "label": p["label"]} for p in pairs], [(task.state_of(p), em_question()) for p in pairs]


def probe_rows(level: int) -> list[tuple]:
    """Reviews for the concurrency probe: the evaluation uses the first 5,000, this uses later ones, a different batch per concurrency level."""
    if REHEARSAL:
        return made_up("movie")[1][level * 10:(level + 1) * 10]
    return [movie_item(t) for t in texts(None, "open")[N_MOVIE + level * PROBE_ROWS:N_MOVIE + (level + 1) * PROBE_ROWS]]


def rows_of(dataset: str, ids: list[dict], answers: list[dict]) -> list[dict]:
    """One record per row: p is the probability of yes, None if there is no answer."""
    noul = lambda a, key: None if "_error" in a else a["answers"][key]["noul"]
    # Token count of this row's longest request: to see whether anything exceeds the context window the model was validated on
    longest = lambda got: max((a["usage"]["input_tokens"] for a in got if "_error" not in a), default=0)
    # The server's own report that something was truncated (laya does not fail beyond its window; it only says so in usage)
    cut = lambda got: any(a["usage"].get("truncated") for a in got if "_error" not in a)
    if dataset == "mams":
        k = len(PREDICATES)
        chunks = [answers[i * k:(i + 1) * k] for i in range(len(ids))]
        return [{**row, "p": {name: noul(a, "q") for name, a in zip(PREDICATES, got)}, "request_tokens": longest(got), "truncated": cut(got)}
                for row, got in zip(ids, chunks)]
    if dataset == "mams.packed":
        return [{**row, "p": None if "_error" in a else {name: noul(a, name) for name in PREDICATES}, "error": a.get("_error"),
                 "request_tokens": longest([a]), "truncated": cut([a])} for row, a in zip(ids, answers)]
    key = "same" if dataset == "em" else "q"
    return [{**row, "p": noul(a, key), "error": a.get("_error"), "request_tokens": longest([a]), "truncated": cut([a])} for row, a in zip(ids, answers)]


def backend_for(source: str, **kwargs):
    return jev_backend(**kwargs) if source == JEV else open_backend(source, **kwargs)


def check_alive(source: str) -> None:
    """Send one small request unrelated to the evaluation first: if the server is down or the key is wrong, stop at once instead of letting thousands of requests retry into timeouts."""

    async def ping() -> None:
        client = JevClient(provider="typesafe", model=JEV_MODEL, max_retries=0) if source == JEV else OpenModelClient(OPEN_MODELS[source], max_retries=0)
        async with client:
            await client.ask("Help! My payouts have been failing for 3 days.", {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}})

    try:
        asyncio.run(ping())
    except JevAPIError as e:
        sys.exit(f"{source} does not answer: {e}\nLocal models need their server started first (scripts/open_models.sh serve); Jev needs TYPESAFE_API_KEY in .env.")


def estimate() -> None:
    cache = Cache(ROOT / ".jevcache.sqlite")
    total = 0
    print(f"{'dataset':14s}{'requests':>9s}{'tokens Jev billed':>18s}{'per request':>12s}")
    for dataset in DATASETS:
        _, reqs = requests(dataset)
        hits = [cache.get(cache_key(JEV_NAMESPACE[dataset] + JEV_MODEL, state, questions)) for state, questions in reqs]
        assert all(hits), f"{dataset}: some requests are not in Jev's cache"
        tokens = sum(h["usage"]["input_tokens"] for h in hits)
        total += tokens
        print(f"{dataset:14s}{len(reqs):>10d}{tokens:>18,d}{tokens / len(reqs):>12.0f}")
    print(f"{'total':14s}{'':>10s}{total:>18,d}")
    usd_in, _ = JevClient("k", provider="typesafe").prices
    print(f"\nThe Jev control group ({JEV}) asking all of these again: about ${total * usd_in:.2f}. The local open models are free.")


def probe(source: str) -> None:
    load_dotenv()
    OUT.mkdir(parents=True, exist_ok=True)
    check_alive(source)
    out = {}
    for level, concurrency in enumerate(PROBE):
        backend = backend_for(source, concurrency=concurrency, cache_path=None)
        batch = probe_rows(level)
        started = time.perf_counter()
        answers = backend.ask_many(batch)
        seconds = time.perf_counter() - started
        failed = sum("_error" in a for a in answers)
        out[concurrency] = {"rows_per_second": round((len(batch) - failed) / seconds, 2), "failed": failed, "seconds": round(seconds, 1), "latency_p50": backend.meter.latency(0.5)}
        print(f"{source} concurrency {concurrency}: {out[concurrency]}", flush=True)
        backend.close()
    best = max(PROBE, key=lambda c: out[c]["rows_per_second"])
    (OUT / f"probe.{source}.json").write_text(json.dumps({"concurrency": best, "levels": out}, ensure_ascii=False, indent=1) + "\n")
    print(f"{source} uses concurrency {best}")


def run(source: str, datasets: list[str], concurrency: int | None) -> None:
    load_dotenv()
    OUT.mkdir(parents=True, exist_ok=True)
    if concurrency is None and source == JEV:
        concurrency = JEV_CONCURRENCY
    if concurrency is None:
        chosen = OUT / f"probe.{source}.json"
        if not chosen.exists():
            sys.exit(f"no concurrency set for {source} yet: run probe {source} first, or pass --concurrency")
        concurrency = json.loads(chosen.read_text())["concurrency"]
    check_alive(source)
    backend = backend_for(source, concurrency=concurrency, cache_path=OUT / f"cache.{source}.sqlite")
    for dataset in datasets:
        # The repeat must really ask again, so it has its own namespace
        backend.namespace = NAMESPACE + ("repeat:" if dataset == "movie.repeat" else "")
        ids, reqs = requests(dataset)
        backend.reset()
        started = time.perf_counter()
        for _ in range(4):
            answers = backend.ask_many(reqs)
            if not backend.meter.errors:
                break
        seconds = time.perf_counter() - started
        m = backend.meter
        with (OUT / f"{dataset}.{source}.jsonl").open("w") as f:
            for row in rows_of(dataset, ids, answers):
                f.write(json.dumps(row) + "\n")
        summary = {"source": source, "dataset": dataset, "model": backend.model, "rows": len(ids), "requests": m.calls, "fresh": m.fresh, "errors": m.errors,
                   "seconds": round(seconds, 1), "concurrency": concurrency, "input_tokens": m.input_tokens, "cost_usd": round(m.cost, 4), "latency_p50": m.latency(0.5),
                   "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        # One line is appended per run: after an interrupted run is resumed, throughput is the new requests of each run divided by each run's time
        with (OUT / f"{dataset}.{source}.runs.jsonl").open("a") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(summary, flush=True)
    backend.close()


def control() -> None:
    """Control group: share of the same movie reviews on which today's Jev and the cached Jev agree. Failing means the collection process or Jev itself changed, and the run should not continue."""
    if REHEARSAL:
        print("no cached Jev to compare with in a rehearsal; skipped")
        return
    fresh = {r["text"]: r["p"] for r in map(json.loads, (OUT / f"movie.{JEV}.jsonl").read_text().splitlines())}
    old = {r["text"]: r["p"] for r in map(json.loads, (RESULTS / "p2" / "jev.positive.jsonl").read_text().splitlines())}
    pairs = [(p, old[t]) for t, p in fresh.items() if p is not None]
    agree = sum((a >= 0.5) == (b >= 0.5) for a, b in pairs) / len(fresh)
    drift = sum(abs(a - b) for a, b in pairs) / max(len(pairs), 1)
    print(f"Jev control group: answered {len(pairs)} of {len(fresh)} reviews, agreement with the cached Jev {agree:.2%} (bar {CONTROL_BAR:.0%}), mean probability difference {drift:.3f}")
    if agree < CONTROL_BAR:
        sys.exit(3)


def check(sources: list[str]) -> None:
    """Are the collected files complete, are the row counts right, how many rows have no answer? Counting only; the answers are not looked at."""
    missing, counts = [], {d: len(requests(d)[0]) for d in DATASETS}
    print(f"{'source':20s}{'dataset':14s}{'rows':>8s}{'no answer':>10s}{'share':>8s}{'failed':>8s}{'runs':>6s}")
    for source in sources:
        for dataset in DATASETS:
            path = OUT / f"{dataset}.{source}.jsonl"
            if not path.exists():
                missing.append(f"{source} / {dataset}")
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            empty = sum(r["p"] is None or (isinstance(r["p"], dict) and any(v is None for v in r["p"].values())) for r in rows)
            log = [json.loads(line) for line in (OUT / f"{dataset}.{source}.runs.jsonl").read_text().splitlines()]
            flag = "" if len(rows) == counts[dataset] else f"  should be {counts[dataset]} rows"
            print(f"{source:20s}{dataset:14s}{len(rows):>10d}{empty:>14d}{empty / len(rows):>10.1%}{log[-1]['errors']:>11d}{len(log):>12d}{flag}")
            if flag:
                missing.append(f"{source} / {dataset} (wrong row count)")
    if missing:
        sys.exit("missing: " + "; ".join(missing))
    print("all files complete")


if __name__ == "__main__":
    args = sys.argv[1:]
    concurrency = int(args.pop(args.index("--concurrency") + 1)) if "--concurrency" in args else None
    args = [a for a in args if a != "--concurrency"]
    if args == ["estimate"]:
        estimate()
    elif args[:1] == ["probe"] and len(args) == 2 and args[1] in SOURCES and args[1] != JEV:
        probe(args[1])
    elif args[:1] == ["run"] and args[1:] and args[1] in SOURCES and all(d in DATASETS for d in args[2:]):
        run(args[1], args[2:] or list(DATASETS), concurrency)
    elif args == ["control"]:
        control()
    elif args[:1] == ["check"] and all(s in SOURCES for s in args[1:]):
        check(args[1:] or list(SOURCES))
    elif args == ["merge"]:
        import p2_collect

        p2_collect.OUT = OUT
        merge_caches()
    else:
        sys.exit(__doc__)
