"""Phase 2: collect, once, every judgement the later phases reuse; from then on everything is replayed at no cost.

Rows: every distinct review text in the 2,000-row and 16,000-row versions of the SemBench movie data.
Questions: "is this review positive?" (the human label scoreSentiment is its ground truth) and the stricter "is it clearly positive?".

source               questions    rows
jev                  both         all
jev.repeat           positive     a fixed sample of 2,000, asked again to measure Jev's own consistency
luna.low             positive     all (the LLM chosen in the qualification test; yes or no only)
v41flash.off etc.    positive     a fixed sample of 5,000 (open models with thinking off; output probabilities are read)

One process and one cache file per source, so they can run in parallel; merge folds them into the main cache afterwards. Failed requests are not cached, so a re-run only re-asks the failures.

    python experiments/database/p2_collect.py run jev
    python experiments/database/p2_collect.py merge
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time

import duckdb
from kevdb.cli import load_dotenv

from kevdb import jev_backend, llm_backend
from paths import DATA as DB_DATA, OUT as RESULTS, ROOT

OUT = RESULTS / "p2"
DATA = f"{DB_DATA}/sembench/files/movie/data"
QUESTIONS = {
    "positive": "Is the movie review in `review` positive?",
    "clearly": "Is the movie review in `review` clearly positive?",
}
# source -> (backend, concurrency, which questions, how many rows (None for all), salt for sampling)
SOURCES = {
    "jev": ("jev", 16, ["positive", "clearly"], None, ""),
    "jev.repeat": ("jev", 16, ["positive"], 2000, "repeat"),
    "luna.low": ("luna.low", 48, ["positive"], None, ""),
    "v41flash.off": ("v41flash.off", 16, ["positive"], 5000, "open"),
    "qwen38flash.off": ("qwen38flash.off", 32, ["positive"], 5000, "open"),
    "v4pro.off": ("v4pro.off", 16, ["positive"], 5000, "open"),
}


def sha(text: str, salt: str = "") -> str:
    return hashlib.sha256((salt + text).encode()).hexdigest()[:16]


def texts(n: int | None = None, salt: str = "") -> list[str]:
    """Every distinct review text in both versions. Samples take the first n texts ordered by the hash of (salt + text), independent of run order."""
    rows = duckdb.sql(
        f"SELECT DISTINCT reviewText FROM read_csv(['{DATA}/sf_2000/Reviews.csv', '{DATA}/sf_16000/Reviews.csv'], union_by_name = true)"
    ).fetchall()
    ordered = sorted((t for (t,) in rows), key=lambda t: sha(t, salt))
    return ordered[:n]


def run(source: str) -> None:
    load_dotenv()
    OUT.mkdir(parents=True, exist_ok=True)
    name, concurrency, questions, n, salt = SOURCES[source]
    cache = OUT / f"cache.{source}.sqlite"
    backend = jev_backend(concurrency=concurrency, cache_path=cache) if name == "jev" else llm_backend(name, concurrency=concurrency, cache_path=cache)
    # The repeat must really ask again, so it has its own namespace; the others share p2 and can be replayed later
    backend.namespace = "p2.repeat:" if source == "jev.repeat" else "p2:"
    rows = texts(n, salt)
    for question in questions:
        items = [({"review": t}, {"q": {"type": "noul", "instructions": QUESTIONS[question]}}) for t in rows]
        backend.reset()
        started = time.perf_counter()
        for attempt in range(4):
            answers = backend.ask_many(items)
            if not backend.meter.errors:
                break
        seconds = time.perf_counter() - started
        m = backend.meter
        with (OUT / f"{source}.{question}.jsonl").open("w") as f:
            for t, a in zip(rows, answers):
                q = None if "_error" in a else a["answers"]["q"]
                f.write(json.dumps({"text": sha(t), "p": q and q["noul"], "p_true": q and q.get("p_true"),
                                    "error": a.get("_error"), "latency": a.get("_latency"), "cost": (a.get("usage") or {}).get("cost")}) + "\n")
        summary = {"source": source, "question": question, "model": backend.model, "rows": len(rows), "fresh": m.fresh, "errors": m.errors,
                   "seconds": round(seconds, 1), "concurrency": concurrency, "cost_usd": round(m.cost, 4), "latency_p50": m.latency(0.5)}
        (OUT / f"{source}.{question}.run.json").write_text(json.dumps(summary, ensure_ascii=False) + "\n")
        print(summary, flush=True)
    backend.close()


def merge() -> None:
    main = sqlite3.connect(ROOT / ".jevcache.sqlite")
    main.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, response TEXT NOT NULL)")
    for path in sorted(OUT.glob("cache.*.sqlite")):
        main.execute("ATTACH ? AS src", (str(path),))
        added = main.execute("INSERT OR IGNORE INTO answers SELECT key, response FROM src.answers").rowcount
        main.commit()
        main.execute("DETACH src")
        print(f"{path.name}: merged {added} entries")


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"] and sys.argv[2:] and all(s in SOURCES for s in sys.argv[2:]):
        for s in sys.argv[2:]:
            run(s)
    elif sys.argv[1:2] == ["merge"]:
        merge()
    else:
        sys.exit(__doc__)
