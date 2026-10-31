"""Phase 16 data: the populations, queries and influences of experiments/database/protocol/p16.txt. Reads cached answers only and calls no model.

A population is a set of rows with a human label per predicate (y), the cheap model's probability (p), structured columns and, where cached,
second models' answers. A query is a set of outputs, each a linear functional of the labels: theta_k = sum_i a_ik y_i (the influence a_ik
comes from structured columns only). With P16_REHEARSAL=1 the labels and second-model answers are replaced by simulated ones right after
loading, so a rehearsal never computes anything on real labels.

    python experiments/database/p16_data.py      lists populations and queries (sizes only)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache

import duckdb
import numpy as np
import scipy.sparse as sp

from paths import DATA, HERE, OUT

REHEARSAL = os.environ.get("P16_REHEARSAL") == "1"
RESULTS = OUT / ("p16_rehearsal" if REHEARSAL else "p16")
SEED = 20261031
MOVIE = DATA / "sembench/files/movie/data/sf_16000"
P2, P4, P5, P6 = OUT / "p2", OUT / "p4", OUT / "p5", OUT / "p6"
EDGES = np.array([0.1, 0.3, 0.7, 0.9])  # score bins [0,.1) [.1,.3) [.3,.7) [.7,.9) [.9,1]
# Population index for the random streams; rows shared by two cheap models use the same index (paired comparison)
INDEX = {"D1": 1, "D1-5k": 2, "D1-1.85k": 3, "D2": 4, "D2-1k": 5, "D2-546": 6}
# Budget grids and primary budgets on the full populations (p16.txt "QUERIES")
SCALAR, GROUPED, PUBS, MAMS = (100, 200, 500, 1000, 2000), (500, 1000, 2000, 4000), (2000, 4000, 8000), (100, 200, 400, 800)
GRID = {"Q1": (SCALAR, 500), "Q3a": (SCALAR, 500), "Q3b": (SCALAR, 500), "Q4b-1": (SCALAR, 500), "Q4b-2": (SCALAR, 500),
        "Q4b-2t": (SCALAR, 500), "Q2": (GROUPED, 2000), "Q4a": (GROUPED, 2000), "Q2b": (PUBS, 4000), "Q6": (MAMS, 400)}
FULL = {"D1": 15388, "D2": 3949}


@dataclass
class Population:
    name: str
    keys: np.ndarray  # (N,) row keys, sorted
    y: np.ndarray  # (N, C) human labels, 0/1
    p: np.ndarray  # (N, C) cheap model's probability
    cols: dict  # structured columns
    second: dict = field(default_factory=dict)  # name -> (N, C) second model's answer (yes/no or probability)
    cheap: str = "jev"
    predicates: tuple = ("positive",)

    @property
    def N(self) -> int:
        return len(self.keys)

    @property
    def C(self) -> int:
        return self.y.shape[1]

    @property
    def index(self) -> int:
        return INDEX[self.name]

    @property
    def family(self) -> str:
        return self.name.split("-")[0]

    @property
    def fold(self) -> np.ndarray:
        return folds(self.family, self.keys)

    @property
    def bins(self) -> np.ndarray:
        return np.searchsorted(EDGES, self.p, side="right").astype(np.int64)


def folds(dataset: str, keys) -> np.ndarray:
    """Parity of the first hex digit of sha256("p16|<dataset>|<key>"): fixed and independent of every label."""
    return np.array([int(hashlib.sha256(f"p16|{dataset}|{k}".encode()).hexdigest()[0], 16) % 2 for k in keys], dtype=np.int64)


def _box_office(text) -> float:
    if not isinstance(text, str) or not text.startswith("$"):
        return math.nan
    value = float(text.strip("$MK"))
    return value * 1e6 if text.endswith("M") else value * 1e3 if text.endswith("K") else value


def _jsonl(path, field_name: str = "p") -> dict:
    return {r["text"]: r[field_name] for r in map(json.loads, path.read_text().splitlines())}


@lru_cache(maxsize=1)
def _d1_frame():
    """All 15,388 rows with their structured columns, the cheap models and the second models (NaN where not cached)."""
    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT DISTINCT id, reviewId, criticName, isTopCritic::INT AS top, publicatioName, length(reviewText) AS len,
               (scoreSentiment = 'POSITIVE')::INT AS y, substr(sha256(reviewText), 1, 16) AS text
        FROM read_csv('{MOVIE}/Reviews.csv') ORDER BY id, reviewId""").fetchdf()
    movies = con.execute(f"SELECT id, genre, boxOffice FROM read_csv('{MOVIE}/Movies.csv')").fetchdf()
    for column in ("genre", "boxOffice"):
        if (movies.groupby("id")[column].nunique(dropna=False) > 1).any():
            raise SystemExit(f"Movies.csv: repeated rows of a movie disagree on {column}")
    movies = movies.drop_duplicates("id").set_index("id")
    rows["genres"] = rows["id"].map(lambda m: tuple(sorted(g.strip() for g in movies.at[m, "genre"].split(","))))
    rows["box"] = rows["id"].map(lambda m: _box_office(movies.at[m, "boxOffice"]))
    rows["key"] = rows["id"] + "|" + rows["reviewId"].astype(str)
    rows = rows.sort_values("key").reset_index(drop=True)
    jev = _jsonl(P2 / "jev.positive.jsonl")
    rows["jev"] = rows["text"].map(jev)
    rows["luna"] = rows["text"].map(_jsonl(P2 / "luna.low.positive.jsonl"))
    for name, path in (("kev-4b", P6 / "movie.kev-4b.jsonl"), ("kev-9b", P6 / "movie.kev-9b.jsonl"), ("opus", P5 / "movie.opus.jsonl")):
        rows[name] = rows["text"].map(_jsonl(path))
    for name, stem in (("deepseek", "v41flash.off"), ("v4pro", "v4pro.off"), ("qwen", "qwen38flash.off")):
        rows[name] = rows["text"].map(_jsonl(P2 / f"{stem}.positive.jsonl", "p_true"))
        rows[name + ".yesno"] = rows["text"].map(_jsonl(P2 / f"{stem}.positive.jsonl"))
    assert len(rows) == FULL["D1"] and rows["jev"].notna().all() and rows["luna"].notna().all()
    # Q4b's semi-synthetic fan-out: one standard normal per row of the full table, rows sorted by key
    rows["z"] = np.random.default_rng(SEED).standard_normal(len(rows))
    return rows


def _codes(values) -> np.ndarray:
    _, codes = np.unique(np.asarray(values, dtype=object).astype(str), return_inverse=True)
    return codes.astype(np.int64)


def d1(subset: str | None = None, cheap: str = "jev") -> Population:
    """D1 (all rows), D1-5k (rows whose text the open models answered) or D1-1.85k (rows Opus answered)."""
    rows = _d1_frame()
    name = "D1"
    if subset == "5k":
        rows, name = rows[rows["deepseek"].notna()], "D1-5k"
    elif subset == "1.85k":
        rows, name = rows[rows["opus"].notna()], "D1-1.85k"
    rows = rows.reset_index(drop=True)
    tertiles = np.quantile(_d1_frame()["len"], [1 / 3, 2 / 3])
    cols = {"movie": rows["id"].to_numpy(), "critic": rows["criticName"].to_numpy(), "publication": rows["publicatioName"].to_numpy(),
            "top": rows["top"].to_numpy(np.int64), "genres": list(rows["genres"]), "box": rows["box"].to_numpy(float),
            "length": np.searchsorted(tertiles, rows["len"].to_numpy(), side="right"), "z": rows["z"].to_numpy(float)}
    second = {}
    for model in ("luna", "kev-4b", "kev-9b", "opus", "deepseek", "deepseek.yesno", "v4pro", "qwen"):
        values = rows[model].to_numpy(float)
        if np.isfinite(values).all():
            second[model] = values[:, None]
    p = rows[cheap].to_numpy(float)
    if not np.isfinite(p).all():
        raise SystemExit(f"{cheap} has no answer for some rows of {name}")
    pop = Population(name, rows["key"].to_numpy(), rows["y"].to_numpy(float)[:, None], p[:, None], cols, second, cheap)
    return simulate(pop) if REHEARSAL else pop


def d2(subset: str | None = None, cheap: str = "jev") -> Population:
    """D2 (MAMS, 3,949 sentences x 21 predicates), D2-1k (sentences the LLMs answered) or D2-546 (Opus)."""
    sys.path.insert(0, str(HERE))
    from p4_collect import PREDICATES, sentences

    names = list(PREDICATES)
    rows = sorted(sentences(), key=lambda r: r["sid"])

    def read(path) -> dict:
        return {r["sid"]: r["p"] for r in map(json.loads, path.read_text().splitlines()) if r.get("p")}

    answers = {"jev": read(P4 / "mams.jev.packed.jsonl"), "luna": read(P4 / "mams.luna.low.packed.jsonl"),
               "deepseek": read(P4 / "mams.v41flash.off.packed.jsonl"), "kev-4b": read(P6 / "mams.packed.kev-4b.jsonl"),
               "opus": read(P5 / "mams.opus.packed.jsonl")}
    name = "D2"
    if subset == "1k":
        rows, name = [r for r in rows if r["sid"] in answers["luna"]], "D2-1k"
    elif subset == "546":
        rows, name = [r for r in rows if r["sid"] in answers["opus"]], "D2-546"
    sids = np.array([r["sid"] for r in rows])
    y = np.array([[r["truth"][k] for k in names] for r in rows], dtype=float)
    lengths = np.array([len(r["text"]) for r in rows])
    second = {m: np.array([[answers[m][s][k] for k in names] for s in sids], dtype=float)
              for m in ("luna", "deepseek", "kev-4b", "opus") if all(s in answers[m] for s in sids)}
    if not all(s in answers[cheap] for s in sids):
        raise SystemExit(f"{cheap} has no answer for some sentences of {name}")
    p = np.array([[answers[cheap][s][k] for k in names] for s in sids], dtype=float)
    pop = Population(name, sids, y, p, {"length": lengths}, second, cheap, tuple(names))
    assert name != "D2" or pop.N == FULL["D2"]
    return simulate(pop) if REHEARSAL else pop


def simulate(pop: Population) -> Population:
    """Rehearsal only: labels drawn from the cheap model's probabilities with planted group effects, and second models drawn around them.
    Nothing here reads a real label; the true y is overwritten before any use."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 99, pop.index]))
    logit = np.log(np.clip(pop.p, 0.01, 0.99) / (1 - np.clip(pop.p, 0.01, 0.99)))
    if pop.family == "D1":
        movie = _codes(pop.cols["movie"])
        effect = rng.normal(0, 0.4, movie.max() + 1)[movie][:, None]
        q = 1 / (1 + np.exp(-(1.5 * logit + 1.2 + effect + rng.normal(0, 0.5, logit.shape))))
    else:
        q = 1 / (1 + np.exp(-(1.2 * logit - 0.8 + rng.normal(0, 0.5, pop.C)[None, :] + rng.normal(0, 0.5, logit.shape))))
    y = (rng.random(q.shape) < q).astype(float)
    second = {}
    for name, values in pop.second.items():
        noisy = 1 / (1 + np.exp(-(np.log(q / (1 - q)) + rng.normal(0, 1.0, q.shape))))
        second[name] = (noisy >= 0.5).astype(float) if set(np.unique(values)) <= {0.0, 1.0} else noisy
    return Population(pop.name, pop.keys, y, pop.p, pop.cols, second, pop.cheap, pop.predicates)


@dataclass
class Query:
    name: str
    A: sp.csr_matrix  # (N, K) influence of row i on output k
    h: np.ndarray  # (K,) requirement used for the loss (Q6: from the truth)
    cells: np.ndarray  # (N,) group key of the cells used by QP, G and D
    AG: sp.csr_matrix  # group-key-only influence (design G)
    hG: np.ndarray
    budgets: tuple
    primary: int
    outputs: list
    relative: bool = False  # Q6: designs estimate h_k = max(.01, .25 theta_k) themselves
    abae: dict = field(default_factory=dict)  # groups (N,) codes or None, members (N,) mask, stat (N,) or None, combine (G, K) or None

    def truth(self, y: np.ndarray) -> np.ndarray:
        return apply(self.A, y)

    @property
    def K(self) -> int:
        return self.A.shape[1]


def apply(A: sp.csr_matrix, Z: np.ndarray) -> np.ndarray:
    """theta_k = sum_i A_ik Z_{i, c(k)}: one predicate for every output (C = 1) or output k on predicate k (K = C)."""
    if Z.shape[1] == 1:
        return np.asarray(A.T @ Z[:, 0]).ravel()
    return np.asarray(A.multiply(Z).sum(axis=0)).ravel()


def _membership(codes: np.ndarray, weights: np.ndarray | None = None) -> sp.csr_matrix:
    """Row i -> its group, with influence 1/n_group (or the given weights)."""
    n = np.bincount(codes)
    values = 1.0 / n[codes] if weights is None else weights
    return sp.csr_matrix((values, (np.arange(len(codes)), codes)), shape=(len(codes), n.size))


def scaled(budgets: tuple, pop: Population) -> tuple:
    """The grid of the full population scaled to a subset: B x N_sub / N, rounded, at least 20."""
    full = FULL[pop.family]
    return tuple(max(20, round(b * pop.N / full)) for b in budgets)


def queries(pop: Population) -> dict:
    out = {}
    N = pop.N
    one = sp.csr_matrix(np.full((N, 1), 1.0 / N))

    def grid(name):
        budgets, primary = GRID[name]
        if pop.N == FULL[pop.family]:
            return budgets, primary
        return scaled(budgets, pop), scaled((primary,), pop)[0]

    if pop.family == "D2":
        A = sp.csr_matrix(np.full((N, pop.C), 1.0 / N))
        theta = apply(A, pop.y)
        out["Q6"] = Query("Q6", A, np.maximum(0.01, 0.25 * theta), np.zeros(N, np.int64), A, np.full(pop.C, 0.01), *grid("Q6"),
                          list(pop.predicates), relative=True, abae={"per_predicate": True})
        return out
    movie = _codes(pop.cols["movie"])
    top = pop.cols["top"]
    # Q1: one large group
    out["Q1"] = Query("Q1", one, np.array([0.01]), top, one, np.array([0.01]), *grid("Q1"), ["all"], abae={"groups": None})
    # Q2: per movie; Q2b: per publication
    A2 = _membership(movie)
    names = list(np.unique(pop.cols["movie"].astype(str)))
    out["Q2"] = Query("Q2", A2, np.full(A2.shape[1], 0.10), movie, A2, np.full(A2.shape[1], 0.10), *grid("Q2"), names,
                      abae={"groups": movie})
    pub = _codes(pop.cols["publication"])
    A2b = _membership(pub)
    out["Q2b"] = Query("Q2b", A2b, np.full(A2b.shape[1], 0.10), pub, A2b, np.full(A2b.shape[1], 0.10), *grid("Q2b"),
                       list(np.unique(pop.cols["publication"].astype(str))), abae={"groups": pub})
    # Q3a: mean of the movie shares; Q3b: box-office-weighted mean over the movies with box office
    G = movie.max() + 1
    n_movie = np.bincount(movie)
    a3a = 1.0 / (G * n_movie[movie])
    A3a = sp.csr_matrix(a3a[:, None])
    out["Q3a"] = Query("Q3a", A3a, np.array([0.01]), movie, A3a, np.array([0.01]), *grid("Q3a"), ["mean of movie shares"],
                       abae={"groups": movie, "combine": np.full((G, 1), 1.0 / G)})
    box = pop.cols["box"]
    has = np.isfinite(box)
    movie_box = np.zeros(G)
    movie_box[movie[has]] = box[has]
    with_box = movie_box > 0
    a3b = np.where(has, np.nan_to_num(box) / n_movie[movie], 0.0) / movie_box[with_box].sum()
    aG3b = np.where(has, 1.0 / (with_box.sum() * n_movie[movie]), 0.0)
    out["Q3b"] = Query("Q3b", sp.csr_matrix(a3b[:, None]), np.array([0.01]), movie, sp.csr_matrix(aG3b[:, None]), np.array([0.01]),
                       *grid("Q3b"), ["box-office-weighted mean of movie shares"],
                       abae={"groups": movie, "members": has, "combine": (movie_box / movie_box[with_box].sum())[:, None]})
    # Q4a: per genre through the movie-genre bridge (a review feeds every genre of its movie)
    genres = sorted({g for gs in pop.cols["genres"] for g in gs})
    gindex = {g: k for k, g in enumerate(genres)}
    rows_, cols_ = [], []
    for i, gs in enumerate(pop.cols["genres"]):
        for g in gs:
            rows_.append(i)
            cols_.append(gindex[g])
    member = sp.csr_matrix((np.ones(len(rows_)), (rows_, cols_)), shape=(N, len(genres)))
    n_genre = np.asarray(member.sum(axis=0)).ravel()
    A4a = sp.csr_matrix(member.multiply(1.0 / n_genre[None, :]))
    combine = np.zeros((G, len(genres)))  # movie estimate -> genre estimate: n_movie / n_genre for each genre of the movie
    for i in range(N):
        for g in pop.cols["genres"][i]:
            combine[movie[i], gindex[g]] = n_movie[movie[i]] / n_genre[gindex[g]]
    out["Q4a"] = Query("Q4a", A4a, np.full(len(genres), 0.05), _codes(["|".join(g) for g in pop.cols["genres"]]), A4a,
                       np.full(len(genres), 0.05), *grid("Q4a"), genres, abae={"groups": movie, "combine": combine})
    # Q4b: semi-synthetic fan-out k_i = 1 + floor(exp(sigma z_i + tau top_i))
    for name, sigma, tau in (("Q4b-1", 1.0, 0.0), ("Q4b-2", 2.0, 0.0), ("Q4b-2t", 2.0, math.log(5))):
        k = fanout(pop, sigma, tau)
        A = sp.csr_matrix((k / k.sum())[:, None])
        out[name] = Query(name, A, np.array([0.01]), top, one, np.array([0.01]), *grid(name), ["share over the fact table"],
                          abae={"groups": None, "stat": k})
    return out


def fanout(pop: Population, sigma: float, tau: float) -> np.ndarray:
    return 1.0 + np.floor(np.exp(sigma * pop.cols["z"] + tau * pop.cols["top"]))


def populations() -> dict:
    """Every population of the standard, by name and cheap model."""
    return {("D1", "jev"): d1(), ("D1-5k", "jev"): d1("5k"), ("D1-5k", "kev-4b"): d1("5k", "kev-4b"), ("D1-1.85k", "jev"): d1("1.85k"),
            ("D2", "jev"): d2(), ("D2-1k", "jev"): d2("1k"), ("D2-1k", "kev-4b"): d2("1k", "kev-4b"), ("D2-546", "jev"): d2("546")}


if __name__ == "__main__":
    for (name, cheap), pop in populations().items():
        qs = queries(pop)
        print(f"{name:9s} cheap={cheap:7s} rows={pop.N:6d} predicates={pop.C:2d} second={sorted(pop.second)} "
              f"queries={', '.join(f'{q.name}:{q.K}' for q in qs.values())}")
