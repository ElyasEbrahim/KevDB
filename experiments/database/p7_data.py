"""Rows and questions of phase 7: does a local decision model inside DuckDB offer anything a cloud API cannot? This file calls no model.

Questions and reading rules: experiments/database/protocol/p7.txt; execution steps: experiments/database/protocol/p7_plan.md.

Data for each of the three parts:
A  semantic-SQL benchmark  the 2,000-row version of SemBench's movie scenario, exactly as in phase 1 (the open mode of experiments/database/p1_movie.py)
B  prefix cache            movie   the distinct review texts of SemBench's 2,000-row version (1,864), ordered by hash; phase 1's four questions
                           mams    phase 4's fixed order of restaurant sentences (3,949); the first three of the 21 predicates
                                   for these two short-text sets: the whole table if its cache fits this model's memory budget, otherwise as many leading rows as fit (see CAPACITY)
                           long    40 long documents: movies in the Rotten Tomatoes source data that are in neither SemBench version; one movie's reviews in id order
                                   joined into one document of at most 24,000 characters; 20 documents where most reviews are positive and 20 where they are not. The three questions are phase 1's
                                   "positive? / sentiment / how much liked", rewritten from one review to a group of reviews; first draft, not edited
                           long1500, long3000, long6000, long12000   shorter prefixes of the first 12 movies per stratum, to see how the speed-up changes with length
C  calibration in the DB   four record-matching datasets (abt-buy, amazon-google, dblp-acm never answered by Kev; walmart-amazon answered in phase 6):
                           1,000 validation pairs chosen by hash as the label pool (the first 200 fit the stored parameters), the whole test split for evaluation

With the environment variable P7_REHEARSAL=1 this is a rehearsal: the same steps, but every row is made up below (the questions are the real ones), with results under out/database/p7_rehearsal.
The environment variable P7_MODEL picks the local model (default kev-4b); results are kept in one directory per model.
"""

from __future__ import annotations

import csv
import functools
import json
import os
import random
import sys
from pathlib import Path

import duckdb
from kevdb import Task

from p1_movie import ANT, PARAMS, TAKEN
from p2_collect import sha
from p4_collect import PREDICATES, question, sentences
from paths import DATA, OUT as RESULTS, ROOT

REHEARSAL = os.environ.get("P7_REHEARSAL") == "1"
MODEL = os.environ.get("P7_MODEL", "kev-4b")
SHARED = RESULTS / ("p7_rehearsal" if REHEARSAL else "p7")  # results independent of the local model (Jev's long-document reference)
OUT = SHARED / MODEL
NAMESPACE = "p7.rehearsal:" if REHEARSAL else "p7:"

# ---------- B: prefix cache ----------
LONG_PER_STRATUM = 2 if REHEARSAL else 20
TIER_PER_STRATUM = 1 if REHEARSAL else 12
LONG_CHARS = 24000
TIERS = (1500, 3000, 6000, 12000)
REPS = 3
# The owner's decision of 2026-10-12: use as much memory as possible. How many short prefixes (states) each model can cache on this machine and how many bytes that takes (bytes)
# were fixed by the capacity probe before collection (experiments/database/p7_capacity.py, made-up rows only); the raw readings are in out/database/p7/<model>/capacity.json
CAPACITY = {
    "kev-4b": {"states": 600, "bytes": 36818366400},
    "kev-0.8b": {"states": 1900, "bytes": 44192689056},
}
BUDGET_STATES, MEMORY_BUDGET = CAPACITY.get(MODEL, {"states": 0})["states"], CAPACITY.get(MODEL, {"bytes": 0})["bytes"]
# Number of distinct texts in each of the two short-text tables
WHOLE_TABLES = {"movie": 1864, "mams": 3949}
# The cap on total cached tokens is set high enough that the number of entries is the only limit that applies
MAX_TOKENS = 4_000_000
WARM_SHORT, WARM_LONG = 20, 2

LEVELS = json.loads(PARAMS["levels"])
MOVIE_QUESTIONS = {
    "clearly": {"type": "noul", "instructions": PARAMS["clearly"]},
    "positive": {"type": "noul", "instructions": PARAMS["positive"]},
    "sentiment": {"type": "choice", "instructions": PARAMS["sentiment"], "criteria": json.loads(PARAMS["options"])},
    "liked": {"type": "score", "instructions": PARAMS["liked"], "criteria": LEVELS},
}
MAMS_QUESTIONS = {name: question(name) for name in list(PREDICATES)[:3]}
# First draft, fixed before collection started and never edited afterwards
LONG_QUESTIONS = {
    "positive": {"type": "noul", "instructions": "Are most of the movie reviews in `reviews` positive?"},
    "sentiment": {"type": "choice", "instructions": "What is the overall sentiment of the movie reviews in `reviews`?",
                  "criteria": {"POSITIVE": "Most of the reviews are positive about the movie.", "NEGATIVE": "Most of the reviews are negative about the movie."}},
    "liked": {"type": "score", "instructions": "How much did the reviewers like the movie, judging from the movie reviews in `reviews`?", "criteria": LEVELS},
}
def n_short(name: str) -> int:
    """How many short-text rows to take: the whole table if it fits the budget together with the warm-up rows, otherwise as many as fit."""
    return 30 if REHEARSAL else min(WHOLE_TABLES[name], BUDGET_STATES - WARM_SHORT)


CACHE_SETS = ("movie", "mams", "long") + tuple(f"long{t}" for t in TIERS)
MAIN_SETS, SWEEP_SETS = CACHE_SETS[:3], CACHE_SETS[3:]
ARMS = ("off", "table", "default")

SEMBENCH = DATA / "sembench/files/movie/data"
RT = DATA / "rotten_tomatoes/rotten_tomatoes_movie_reviews.csv"

# ---------- C: calibration ----------
EM = ("abt-buy", "amazon-google", "dblp-acm", "walmart-amazon")
FRESH = EM[:3]
POOL, LABELS = (60, 20) if REHEARSAL else (1000, 200)


# ---------- Made-up rows (for rehearsals and warm-up), unrelated to any dataset ----------

OPENERS = ["A sluggish", "A dazzling", "An uneven", "A heartfelt", "A noisy", "A patient", "A glossy", "A scrappy"]
KINDS_ = ["thriller", "family drama", "space opera", "biopic", "heist comedy", "road movie", "courtroom drama"]
MIDDLES = ["that never quite finds its footing", "anchored by two fine lead performances", "that fully earns its final scene", "padded out with a needless subplot",
           "with a script sharper than it first appears", "that mistakes volume for excitement"]
GOOD = ["I would gladly see it again.", "It sent me home happy.", "The last act is a small triumph.", "Easily one of the better films this season."]
BAD = ["I checked my watch twice.", "It left me cold.", "The ending undoes whatever goodwill it had built.", "A long two hours with little to show for it."]


def made_review(rng: random.Random, positive: bool | None = None) -> tuple[str, bool]:
    positive = rng.random() < 0.5 if positive is None else positive
    return f"{rng.choice(OPENERS)} {rng.choice(KINDS_)} {rng.choice(MIDDLES)}. {rng.choice(GOOD if positive else BAD)}", positive


def made_reviews(seed: str, n: int) -> list[dict]:
    rng, rows = random.Random("p7:" + seed), []
    for i in range(n):
        text, positive = made_review(rng)
        rows.append({"id": f"made-up-{seed}-{i}", "state": {"review": f"{text} (screening {seed}-{i})"}, "label": int(positive)})
    return rows


def made_sentences(seed: str, n: int) -> list[dict]:
    rng = random.Random("p7:" + seed)
    pick = rng.choice
    return [{"id": f"made-up-{seed}-{i}", "state": {"sentence":
             f"{pick(['The dumplings', 'Our waiter', 'The patio', 'The wine list', 'The tasting menu'])} {pick(['was wonderful', 'was a letdown', 'took forever', 'felt overpriced'])}, "
             f"{pick(['but the dessert made up for it', 'and the bill was a shock', 'though the room itself is lovely'])} (table {seed}-{i})."}} for i in range(n)]


def made_doc(seed: str, i: int, budget: int, share: float) -> dict:
    """One made-up long document: one review per line, with about `share` of them positive."""
    rng = random.Random(f"p7:{seed}:{i}:{budget}")
    lines, labels, size = [], [], 0
    while True:
        text, positive = made_review(rng, rng.random() < share)
        line = f"{len(lines) + 1}. {text}" + (f" (programme {seed}-{i})" if not lines else "")
        if size + len(line) + 1 > budget:
            break
        lines.append(line); labels.append(positive); size += len(line) + 1
    return {"id": f"made-up-{seed}-{i}@{budget}", "movie": f"made-up-{seed}-{i}", "state": {"reviews": "\n".join(lines)},
            "share": sum(labels) / len(labels), "reviews": len(lines), "chars": size - 1}


def made_docs(seed: str, per_stratum: int, budget: int) -> list[dict]:
    return [made_doc(seed, i, budget, 0.25 if i % 2 else 0.75) for i in range(2 * per_stratum)]


def made_pairs(seed: str, n: int) -> list[dict]:
    rng, rows = random.Random("p7:" + seed), []
    pick = rng.choice
    for i in range(n):
        brand, item, k, same = pick(["Norvale", "Kestrel", "Ombra"]), pick(["wireless optical mouse", "USB-C docking station", "portable SSD 1TB"]), rng.randrange(100, 999), rng.random() < 0.35
        a = {"title": f"{brand} {item} model {k} black", "category": "computer accessories", "brand": brand, "modelno": f"{brand[:2].upper()}{k}", "price": f"{rng.randrange(20, 300)}.99"}
        b = {"title": f"{brand} {item} {k if same else k + 1} (black) retail pack {seed}-{i}", "category": "electronics - general", "brand": brand, "modelno": f"{brand[:2].upper()}-{k if same else k + 1}"}
        rows.append({"id": f"made-up-{seed}-{i}", "record_a": a, "record_b": b, "label": int(same)})
    return rows


# ---------- Data for B ----------

@functools.cache
def _long_source() -> dict[str, list[tuple[str, int]]]:
    """movie -> [(review text, is positive)], ordered by review id. Only movies absent from both SemBench versions; repeated texts within a movie keep the lowest id."""
    seen = [f"{SEMBENCH}/sf_2000/Reviews.csv", f"{SEMBENCH}/sf_16000/Reviews.csv"]
    rows = duckdb.sql(f"""
        WITH seen AS (SELECT DISTINCT id FROM read_csv({seen}, union_by_name = true)),
        cleaned AS (SELECT id, reviewId, trim(regexp_replace(reviewText, '\\s+', ' ', 'g')) AS text, scoreSentiment
                    FROM read_csv('{RT}') WHERE reviewText IS NOT NULL AND scoreSentiment IN ('POSITIVE', 'NEGATIVE') AND id IS NOT NULL AND reviewId IS NOT NULL)
        SELECT id, text, (arg_min(scoreSentiment, reviewId) = 'POSITIVE')::INT AS positive, min(reviewId) AS first
        FROM cleaned WHERE length(text) > 0 AND id NOT IN (SELECT id FROM seen) GROUP BY id, text ORDER BY id, first, text""").fetchall()
    movies: dict[str, list[tuple[str, int]]] = {}
    for movie, text, positive, _ in rows:
        movies.setdefault(movie, []).append((text, positive))
    return movies


def _doc(movie: str, reviews: list[tuple[str, int]], budget: int) -> dict | None:
    """This movie's reviews in order, one per line, until one more would exceed `budget` characters. Movies with too few reviews to fill it are skipped (None)."""
    lines, labels, size = [], [], 0
    for text, positive in reviews:
        line = f"{len(lines) + 1}. {text}"
        if size + len(line) + 1 > budget:
            return {"id": f"{movie}@{budget}", "movie": movie, "state": {"reviews": "\n".join(lines)}, "share": sum(labels) / len(labels),
                    "reviews": len(lines), "chars": size - 1}
        lines.append(line); labels.append(positive); size += len(line) + 1
    return None


@functools.cache
def _long_movies() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(movies with mostly positive reviews, movies without), each ordered by the hash of the movie id, first LONG_PER_STRATUM of each. Strata use the 24,000-character document."""
    docs = [d for movie, reviews in _long_source().items() if (d := _doc(movie, reviews, LONG_CHARS))]
    ordered = sorted(docs, key=lambda d: sha(d["movie"], "p7"))
    return (tuple(d["movie"] for d in ordered if d["share"] >= 0.5)[:LONG_PER_STRATUM], tuple(d["movie"] for d in ordered if d["share"] < 0.5)[:LONG_PER_STRATUM])


def long_docs(budget: int = LONG_CHARS) -> list[dict]:
    per = LONG_PER_STRATUM if budget == LONG_CHARS else TIER_PER_STRATUM
    if REHEARSAL:
        return made_docs("doc", per, budget)
    source, (positive, negative) = _long_source(), _long_movies()
    # Alternate the two strata, so every block collected contains both
    movies = [m for pair in zip(positive[:per], negative[:per]) for m in pair]
    return [_doc(m, source[m], budget) for m in movies]


def cache_set(name: str) -> tuple[list[dict], dict[str, dict], int, list[dict]]:
    """(rows, questions, rows per block, made-up warm-up rows). The row order is fixed."""
    if name == "movie":
        if REHEARSAL:
            return made_reviews("rows", n_short(name)), MOVIE_QUESTIONS, 10, made_reviews("warm", WARM_SHORT)
        texts = sorted((t for (t,) in duckdb.sql(f"SELECT DISTINCT reviewText FROM read_csv('{SEMBENCH}/sf_2000/Reviews.csv')").fetchall()), key=lambda t: sha(t, "p7"))
        assert len(texts) == WHOLE_TABLES[name]
        return [{"id": sha(t), "state": {"review": t}} for t in texts[:n_short(name)]], MOVIE_QUESTIONS, 50, made_reviews("warm", WARM_SHORT)
    if name == "mams":
        if REHEARSAL:
            return made_sentences("rows", n_short(name)), MAMS_QUESTIONS, 10, made_sentences("warm", WARM_SHORT)
        every = sentences()
        assert len(every) == WHOLE_TABLES[name]
        return [{"id": r["sid"], "state": {"sentence": r["text"]}} for r in sentences(n_short(name))], MAMS_QUESTIONS, 50, made_sentences("warm", WARM_SHORT)
    budget = LONG_CHARS if name == "long" else int(name.removeprefix("long"))
    rows = long_docs(budget)
    return rows, LONG_QUESTIONS, max(len(rows) // 8, 1) if name == "long" else max(len(rows) // 4, 1), made_docs("warm", 1, budget)[:WARM_LONG]


def server_settings(arm: str, n_rows: int, n_warm: int) -> dict[str, str]:
    """Environment variables for starting the server of this arm. default sets nothing, i.e. the factory setting (4 prefixes, 65,536 tokens in total).
    budget is for the SQL step: the cache opened up to this model's memory budget."""
    if arm == "off":
        return {"KEV_PREFIX_CACHE": "0"}
    if arm == "budget":
        return {"KEV_PREFIX_CACHE": str(30 if REHEARSAL else BUDGET_STATES), "KEV_PREFIX_MAX_TOKENS": str(MAX_TOKENS)}
    if arm == "table":
        return {"KEV_PREFIX_CACHE": str(n_rows + n_warm), "KEV_PREFIX_MAX_TOKENS": str(MAX_TOKENS)}
    return {}


# ---------- Rehearsal tables for A ----------

def made_tables() -> Path:
    """Movies.csv and Reviews.csv for rehearsals: SemBench's columns with made-up content. Both movies named in the queries are present, one with every review appearing twice as in the original table."""
    folder = OUT / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    rng, reviews, movies = random.Random("p7:tables"), [], []
    plan = [(TAKEN, 12, False), (ANT, 8, True), ("made_up_movie_a", 6, False), ("made_up_movie_b", 6, False), ("made_up_movie_c", 6, False)]
    for m, (movie, n, doubled) in enumerate(plan):
        movies.append({"id": movie, "title": movie.replace("_", " ").title(), "audienceScore": rng.randrange(30, 95), "tomatoMeter": rng.randrange(20, 98)})
        for i in range(n):
            text, positive = made_review(rng)
            row = {"id": movie, "reviewId": 1000 * (m + 1) + i, "creationDate": "2020-01-01", "criticName": f"Critic {i}", "isTopCritic": i % 3 == 0,
                   "originalScore": f"{rng.randrange(3, 5) if positive else rng.randrange(1, 3)}/5", "reviewState": "fresh" if positive else "rotten",
                   "publicatioName": "Made-Up Gazette", "reviewText": f"{text} (screening {movie}-{i})", "scoreSentiment": "POSITIVE" if positive else "NEGATIVE", "reviewUrl": ""}
            reviews += [row, row] if doubled else [row]
    for name, rows in (("Movies", movies), ("Reviews", reviews)):
        with (folder / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return folder


# ---------- Data for C ----------

def em_question(dataset: str) -> tuple[str, str]:
    """(question, raw JSON of the criteria): the question asked of Jev in experiments/entity_match/run.py."""
    sys.path.insert(0, str(ROOT / "experiments" / "entity_match"))
    from run import DATASETS, KINDS

    payload = KINDS[DATASETS[dataset]].payload()
    return payload["instructions"], json.dumps(payload["criteria"], ensure_ascii=False)


def em_rows(dataset: str, part: str) -> list[dict]:
    """part is pool (POOL validation pairs chosen by hash; the first LABELS fit the stored parameters) or test (the whole test split). Each row: id, label, raw JSON of the state."""
    if REHEARSAL:
        pairs = made_pairs(f"{dataset}-{part}", POOL if part == "pool" else 40)
    else:
        sys.path.insert(0, str(ROOT / "experiments" / "entity_match"))
        from run import load_pairs

        pairs = load_pairs(dataset, "valid" if part == "pool" else "test")
        if part == "pool":
            pairs = sorted(pairs, key=lambda p: sha(p["id"], f"p7:{dataset}"))[:POOL]
    task = Task(name="em", input="", questions={}, state_fields=["record_a", "record_b"])
    return [{"id": p["id"], "label": p["label"], "state": json.dumps(task.state_of(p), ensure_ascii=False)} for p in pairs]


if __name__ == "__main__":
    if sys.argv[1:] == ["tables"] and REHEARSAL:
        print(made_tables())
    else:
        sys.exit("usage: P7_REHEARSAL=1 python experiments/database/p7_data.py tables   (builds the made-up rehearsal tables and prints their directory)")
