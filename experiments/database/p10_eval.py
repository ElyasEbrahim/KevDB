"""Phase 10 evaluation: serve one checkpoint, ask it every request set p10.txt asks of it, stop the server. Criteria: experiments/database/protocol/p10.txt;
runbook: experiments/database/protocol/p10_plan.md. The whole run is one command, experiments/database/p10_run.sh, which calls the subcommands below.

    python experiments/database/p10_eval.py arm kev-4b loo-abt-buy       serve open_models/p10/kev-4b/loo-abt-buy (sandboxed, port 8015), ask its request sets, stop the server
    python experiments/database/p10_eval.py arm kev-4b released          the released v1.0 checkpoint: answers already in the caches of phases 6-8 are replayed, only the rest is asked
    python experiments/database/p10_eval.py trained kev-4b loo-abt-buy   exit 0 if this arm's checkpoint is complete and counts (files, training log, the trainer's own metrics)
    python experiments/database/p10_eval.py verify                       the inputs, the built training files and the label pools against the checksums frozen below
    python experiments/database/p10_eval.py plan kev-4b                  no model call: requests per arm and set, and how many of the released arm's are in earlier caches
    python experiments/database/p10_eval.py check kev-4b                 inventory: counts, timings, attempts and deviations; no answer is looked at

Request sets, one <set>.jsonl / <set>.run.json pair per arm under out/database/p10/<model>/<arm>/:
    test.<D>          the full test split of D (loo-D: D only; all and released: all four)
    pool.<D>          temperature labels: p10_data.pool_pairs (the 200 validation pairs phase 7 fitted on) for each dataset the arm trained on; released: all four, a reading only
    g1                guard: the first 1,000 review texts of phase 6, "positive?"
    g2                guard: the first 100 restaurant sentences of phase 6 x 21 predicates, one question per request
    g3                guard: decision-v7's development split, 1,204 records with 1,468 questions
    e2e.abt-buy       phase 8's abt-buy subset in pair form (400 left records x their top-10 candidates): loo-abt-buy and released
    e2e.dblp-scholar  the same on dblp-scholar: loo-dblp-acm and released
    order.<order>     phase 8's single-choice form on the first 100 left records of the abt-buy subset, candidates in random / rank / reversed order: loo-abt-buy and released

Rules carried out here (p10_plan.md section 8): failed requests are re-asked in up to 4 passes (rule 5); a server that does not start, answers as the wrong
checkpoint or exits during collection ends the attempt, and the whole step is done once more with a new server, answers already given coming from the cache (rule 4).
A set that has its run.json is finished and is never asked again.

With P10_REHEARSAL=1 every row is made up (the questions are the real ones), checkpoints come from open_models/p10_rehearsal and results go to out/database/p10_rehearsal.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import p10_data as D
from paths import OUT as RESULTS, ROOT

if D.REHEARSAL:  # the made-up rows of phases 6 and 8 are reused, so their modules must build made-up rows too
    os.environ.setdefault("P6_REHEARSAL", "1")
    os.environ.setdefault("P8_REHEARSAL", "1")

from kevdb import Backend, OpenModel, OpenModelClient  # noqa: E402
from kevdb.cache import cache_key  # noqa: E402
from kevdb.client import PROVIDERS, JevAuthError, Provider  # noqa: E402

MODELS = ("kev-0.8b", "kev-4b")
ARMS = tuple(f"loo-{d}" for d in D.EM) + ("all", "released")
TRAINED = ARMS[:-1]
OPEN_MODELS_DIR = ROOT / "open_models"
KEV = OPEN_MODELS_DIR / "kev"
RUNS = OPEN_MODELS_DIR / ("p10_rehearsal" if D.REHEARSAL else "p10")
WEIGHTS_TAG = "v1.0"
# A port of its own: a server of an earlier phase left running on 8009 / 8014 can never answer for a checkpoint of this one
PORT = 8015
CONCURRENCY = 16  # as in phases 6-8
PASSES = 4  # rule 5
ATTEMPTS = 2  # rule 4: the step once more after a server failure
START_SECONDS = 900
NAMESPACE = "p10.rehearsal:" if D.REHEARSAL else "p10:"
# Replayed by p10_train.sh from decision-v7's training part (official run; rehearsals use P10_REPLAY, default 16)
REPLAY = int(os.environ.get("P10_REPLAY", "16")) if D.REHEARSAL else 2000
GUARDS = ("g1", "g2", "g3")
ORDERS = ("random", "rank", "reversed")
E2E = {"abt-buy": "loo-abt-buy", "dblp-scholar": "loo-dblp-acm"}  # phase 8 table -> the leave-one-out arm that has seen no record of its source
ORDER_ROWS = 100
G1_TEXTS, G2_SENTENCES = 1000, 100
SUITE = KEV / "evals" / "v7" / "decision-v7"

PROVIDERS.setdefault("p10", Provider(f"http://127.0.0.1:{PORT}/v1", None, "kev-latest"))

# Environment of the server, as scripts/open_models.sh sets it for `serve`: interpreters, caches and weights inside open_models/, never online.
# The server runs at its factory settings, so settings inherited from outside are removed (as p7_run.sh and p8_run.sh do).
SERVE_ENV = {
    "HF_HOME": str(OPEN_MODELS_DIR / "huggingface"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1", "HF_HUB_DISABLE_XET": "1",
    "UV_CACHE_DIR": str(OPEN_MODELS_DIR / "uv" / "cache"), "UV_PYTHON_INSTALL_DIR": str(OPEN_MODELS_DIR / "uv" / "python"), "UV_PYTHON_BIN_DIR": str(OPEN_MODELS_DIR / "uv" / "bin"),
    "UV_PYTHON_PREFERENCE": "only-managed", "UV_NO_CONFIG": "1", "XDG_CACHE_HOME": str(OPEN_MODELS_DIR / "cache"), "XDG_DATA_HOME": str(OPEN_MODELS_DIR / "data"),
    "PIP_CACHE_DIR": str(OPEN_MODELS_DIR / "cache" / "pip"), "TORCH_HOME": str(OPEN_MODELS_DIR / "cache" / "torch"),
}
FACTORY = ("KEV_PREFIX_CACHE", "KEV_PREFIX_MAX_TOKENS", "KEV_PREFIX_MIN_TOKENS", "KEV_TEMPERATURE", "KEV_DTYPE", "KEV_BACKEND", "KEV_TRUNCATE_STATES", "KEV_DATE_FACTS")
SANDBOX = '(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))'

# Frozen before the run (2026-10-15, see the block appended to p10.txt): (sha256, lines). The step-1 acceptance check of p10_plan.md section 6.
INPUTS = {
    "experiments/entity_match/data/abt-buy/train.txt": "c1370966c08c246e9cf844b7cd2cf580b2559e9831f1eb3cca560fd53c1bcbb2",
    "experiments/entity_match/data/amazon-google/train.txt": "84ef5f7747296990a3775d2c8774949e720df643106b1b447682265f41929b26",
    "experiments/entity_match/data/walmart-amazon/train.txt": "092fccd11c1cc1ffe54a702af276641218ce9506e6d60fb5a29a3c67d87892c3",
    "experiments/entity_match/data/dblp-acm/train.txt": "4f757d2d5dae4f7d7d118b796c683a232b4e11bf000e2012b64f8af95a981a4e",
    "open_models/kev/evals/v7/decision-v7/train.jsonl": "7ed5254b5cb5291baefaceb09edf7e13110258211518c8038f4a12c11bd628ad",
    "open_models/kev/evals/v7/decision-v7/development.jsonl": "8d5765d7aec4d08c61854f4664ca79ec2ba44ed092e9b967eaf61ed86c496d9c",
}
# The five training files and four label pools p10_data.py builds from them (official run, then the made-up rehearsal ones)
BUILT = {
    "train/all.jsonl": ("71ee73a9492c568ffdc0aa7c831455f121e201d11d2f954bade9218112636689", 8000),
    "train/loo-abt-buy.jsonl": ("08e8693c27bf089c1bddbce57f972f0ec7e06b68550828c87a5935bff82913f3", 6000),
    "train/loo-amazon-google.jsonl": ("15161f3dae3e7d1d32729d923ae1d0d032e86c1a2d96f6f112f824707225a236", 6000),
    "train/loo-dblp-acm.jsonl": ("4b7927c9d50b0474c5a408b443348ce19eefca2d205bfeb29060ea9ba1ba46a8", 6000),
    "train/loo-walmart-amazon.jsonl": ("beacb5811c8717ca5d4b28e521844af28c1e539bdda77fd8400302189c37855e", 6000),
    "pool/abt-buy.jsonl": ("680e99142956a652d32bd8eddee4d17851c56bcd666f40917336c836f78e2b8a", 200),
    "pool/amazon-google.jsonl": ("01f6c5dc215904012101f19b21a7ed0c9f8dc3511d141f0b52a3be1182c6c85b", 200),
    "pool/dblp-acm.jsonl": ("cd743fdb29b36c67c9d98488c76fc6577c9e06ebd530d26c01932ee243551860", 200),
    "pool/walmart-amazon.jsonl": ("f1be3c88aa5e7ce64a069d152f369248bf7815dba78c8eb434e3ce229a84d58c", 200),
}
BUILT_REHEARSAL = {
    "train/all.jsonl": ("706116866956829e6906a8f8ee6f1167bc8d86fa7bc8fc257f6123d98ad9d691", 96),
    "train/loo-abt-buy.jsonl": ("9b4bc14ba09d48df4d376a009cd8e62d28d0e21c043904caa659f22b8652e064", 72),
    "train/loo-amazon-google.jsonl": ("5d8fcd1a941072a4ed81ba3dcc495d0a1937c0a76f4fd82c568db38a148dcf84", 72),
    "train/loo-dblp-acm.jsonl": ("f04028cc337b018c604e50fb055d0c62eb79cf67d3350a495bf55d29e5f26551", 72),
    "train/loo-walmart-amazon.jsonl": ("fa428dfc6a8cff8b4610d429817f46ab3db356951617f7e7489f42e10a39d009", 72),
    "pool/abt-buy.jsonl": ("a8ba9e066e742de9b3adc09a357ad2c5e90f2beb9c940e5654008cfd1a43d9fa", 12),
    "pool/amazon-google.jsonl": ("dc3d84a653e458e8ab61988401d740f05e63e4f9c4bedaf1a928cb8656a9a918", 12),
    "pool/dblp-acm.jsonl": ("b7213bf7faa6c8f60cfe6205b2049a952582753e2661e52a876de18cdedb1d0d", 12),
    "pool/walmart-amazon.jsonl": ("e1c62453259b4da8e15d58b5b202038b07504948508e0ffbc743240964ff4534", 12),
}


# ---------- Paths ----------

def out_dir(model: str, arm: str | None = None) -> Path:
    base = D.OUT / model
    return base / arm if arm else base


def checkpoint(model: str, arm: str) -> Path:
    return RUNS / model / arm


def sets(arm: str) -> list[str]:
    """The request sets of one arm, in the order they are asked."""
    held = D.EM if arm in ("all", "released") else (arm.removeprefix("loo-"),)
    trained_on = D.EM if arm == "released" else D.ARMS[arm]
    names = [f"test.{d}" for d in held] + [f"pool.{d}" for d in trained_on] + list(GUARDS)
    if arm in ("loo-abt-buy", "released"):
        names += ["e2e.abt-buy"] + [f"order.{o}" for o in ORDERS]
    if arm in ("loo-dblp-acm", "released"):
        names += ["e2e.dblp-scholar"]
    return names


# ---------- Rows of each request set. Each row: id, state, questions (exactly what is sent; never a label), and its labels kept beside ----------

def _pairs(pairs: list[dict], dataset: str) -> list[dict]:
    return [{"id": p["id"], "label": p["label"], **D.request(p, dataset)} for p in pairs]


def _g1() -> list[dict]:
    """Phase 6's request for the first 1,000 of its 5,000 review texts; labels are the human ones of every review row with that text (phase 6 scored them per row)."""
    from p2_collect import QUESTIONS
    question = {"q": {"type": "noul", "instructions": QUESTIONS["positive"]}}
    if D.REHEARSAL:
        import p6_collect
        ids, items = p6_collect.made_up("movie")
        return [{"id": i["text"], "labels": [int("gladly" in s["review"])], "state": s, "questions": q} for i, (s, q) in zip(ids, items)]
    import duckdb
    from p2_collect import sha, texts
    from p3_aggregates import BIG
    from p6_collect import N_MOVIE

    rows = texts(N_MOVIE, "open")[:G1_TEXTS]
    # The same rows as p3_aggregates.load(), which phase 6 scored: one per distinct review row of the 16,000-row version, keyed by the text's hash
    labels: dict[str, list[int]] = {}
    for y, text in duckdb.sql(f"""SELECT y, text FROM (SELECT DISTINCT id, reviewId, isTopCritic, (scoreSentiment = 'POSITIVE')::INT AS y,
                                  substr(sha256(reviewText), 1, 16) AS text FROM read_csv('{BIG}')) ORDER BY id, reviewId""").fetchall():
        labels.setdefault(text, []).append(int(y))
    return [{"id": sha(t), "labels": labels.get(sha(t), []), "state": {"review": t}, "questions": question} for t in rows]


def _g2() -> list[dict]:
    from p4_collect import PREDICATES, question
    if D.REHEARSAL:
        import p6_collect
        ids, items = p6_collect.made_up("mams")  # 21 requests per sentence, in predicate order
        names = list(PREDICATES)
        return [{"id": f"{ids[i // len(names)]['sid']}|{names[i % len(names)]}", "label": int(i % 3 == 0), "state": s, "questions": q} for i, (s, q) in enumerate(items)]
    from p4_collect import sentences

    return [{"id": f"{r['sid']}|{name}", "label": int(r["truth"][name]), "state": {"sentence": r["text"]}, "questions": {"q": question(name)}}
            for r in sentences(G2_SENTENCES) for name in PREDICATES]


def _g3() -> list[dict]:
    """decision-v7's development records: questions as Kev's evaluation asks them (type, instructions, criteria), labels kept beside."""
    rows = []
    records = [json.loads(line) for line in (SUITE / "development.jsonl").read_text(encoding="utf-8").splitlines()]
    if D.REHEARSAL:
        # Rehearsal: the questions of the first 10 records with each question type (choice, noul, score), each with a made-up state of about the
        # same length; no development state is ever shown to a rehearsal checkpoint, so the labels mean nothing here
        picked = [r for kind in ("choice", "noul", "score") for r in [r for r in records if any(q["type"] == kind for q in r["questions"].values())][:10]]
        filler = "The parcel arrived two days late and the box was dented, but the contents were fine and support answered quickly. "
        records = [r | {"state": (filler * (1 + len(json.dumps(r["state"])) // len(filler)))[:max(len(json.dumps(r["state"])), 40)] + f" (note {i})"} for i, r in enumerate(picked)]
    for r in records:
        questions = {name: {k: q[k] for k in ("type", "instructions", "criteria") if k in q} for name, q in r["questions"].items()}
        rows.append({"id": r["_meta"]["id"] if not D.REHEARSAL else f"made-up-{r['_meta']['id']}", "labels": {n: q["label"] for n, q in r["questions"].items()},
                     "types": {n: q["type"] for n, q in r["questions"].items()}, "state": r["state"], "questions": questions})
    return rows


def p8_module():
    """Phase 8's request builders, with the two raw tables and the stored top-10 candidates. The deeper candidate lists phase 8 rebuilt for its
    candidate-count stretch are not needed here, so they are not rebuilt (load() would re-run blocking over the whole abt-buy table)."""
    import p8_join as P8
    for dataset in E2E:
        if dataset not in P8._DATA:
            if P8.REHEARSAL:
                P8._DATA[dataset] = P8.made_up(dataset)
            else:
                spec = P8.tables.DATASETS[dataset]
                P8._DATA[dataset] = (P8.tables.load_table(spec["left"]), P8.tables.load_table(spec["right"]), P8.tables.load_truth(spec["truth"]),
                                     P8.tables.load_candidates(dataset, P8.K))
    return P8


def _e2e(dataset: str) -> list[dict]:
    P8 = p8_module()
    lids = P8.subset(dataset, P8._DATA[dataset][0])
    meta, items = P8.requests(dataset, "pair", "rank", lids, P8.K)
    truth = P8._DATA[dataset][2]
    return [{"id": f"{m['lid']}|{m['rids'][0]}", "lid": m["lid"], "rid": m["rids"][0], "label": int((m["lid"], m["rids"][0]) in truth), "state": s, "questions": q}
            for m, (s, q) in zip(meta, items)]


def _order(order: str) -> list[dict]:
    P8 = p8_module()
    lids = P8.subset("abt-buy", P8._DATA["abt-buy"][0])[:ORDER_ROWS]
    meta, items = P8.requests("abt-buy", "C", order, lids, P8.K)
    return [{"id": m["lid"], "lid": m["lid"], "rids": m["rids"], "state": s, "questions": q} for m, (s, q) in zip(meta, items)]


_ROWS: dict[str, list[dict]] = {}


def rows(name: str) -> list[dict]:
    if name not in _ROWS:
        kind, _, arg = name.partition(".")
        _ROWS[name] = {
            "test": lambda: _pairs(D.load_pairs(arg, "test"), arg), "pool": lambda: _pairs(D.pool_pairs(arg), arg),
            "g1": _g1, "g2": _g2, "g3": _g3, "e2e": lambda: _e2e(arg), "order": lambda: _order(arg),
        }[kind]()
    return _ROWS[name]


# ---------- Answers ----------

def extract(row: dict, answer: dict) -> dict:
    """What is kept of one answer: the probability of yes (noul), the chosen option and its probabilities (choice), the most probable level (score)."""
    got = answer["answers"]
    out = {}
    for name, q in row["questions"].items():
        a = got[name]
        if q["type"] == "noul":
            out[name] = {"p": a["noul"]}
        elif q["type"] == "choice":
            out[name] = {"choice": a["choice"], "probabilities": a.get("probabilities"), "confidence": a.get("confidence")}
        else:
            probs = {int(k): v for k, v in a["probabilities"].items()}
            out[name] = {"level": max(probs, key=lambda k: (probs[k], -k)), "probabilities": a["probabilities"]}
    return out


def record(row: dict, answer: dict) -> dict:
    kept = {k: v for k, v in row.items() if k not in ("state", "questions")}
    if "_error" in answer:
        return kept | {"error": answer["_error"]}
    got = extract(row, answer)
    only = next(iter(got.values())) if len(got) == 1 else None
    body = {"p": only["p"]} if only and "p" in only else {"answers": got}
    return kept | body | {"tokens": answer["usage"]["input_tokens"], "latency": round(answer["_latency"], 4), "from": answer.get("_replayed_from") or ("cache" if answer["_cached"] else "fresh")}


# ---------- Replay of the released checkpoint's earlier answers ----------

def replay_sources(model: str) -> list[tuple[str, Path, tuple[str, ...]]]:
    """(label, cache file, namespaces) of the earlier phases that asked this released checkpoint. Only files that exist; they are opened read-only."""
    tag = ".rehearsal" if D.REHEARSAL else ""
    base = RESULTS
    found = [("p7", base / f"p7{tag.replace('.', '_')}" / model / "cache.sqlite", (f"p7{tag}:",)),
             ("p6", base / f"p6{tag.replace('.', '_')}" / f"cache.{model}.sqlite", (f"p6{tag}:",)),
             ("p8", base / f"p8{tag.replace('.', '_')}" / f"cache.{model}.sqlite", (f"p8{tag}:fresh:", f"p8{tag}:"))]
    return [s for s in found if s[1].exists()]


def forms(questions: dict) -> list[tuple[dict, dict[str, str]]]:
    """The question dict as earlier phases may have sent it, with the map back to this phase's names. The name of a question is not part of what Kev
    reads (kev/api.py to_record keeps only instructions and options), so phase 7's SQL requests (one question named "q") ask the same thing as
    this phase's (named "same"). Requests with several questions are only looked up as they are."""
    out = [(questions, {n: n for n in questions})]
    if len(questions) == 1:
        (name, q), = questions.items()
        out += [({alias: q}, {alias: name}) for alias in ("q", "same") if alias != name]
    return out


def replay(model: str, cache_id: str, items: list[tuple], cache_path: Path, sources: list | None = None) -> dict[str, int]:
    """Copy into this phase's cache every answer of the released checkpoint an earlier phase already has for these requests, so that asking
    them afterwards finds them in the cache and sends nothing. Answers already in this phase's cache are left alone. Returns counts per source."""
    sources = replay_sources(model) if sources is None else sources
    counts: dict[str, int] = {}
    if not sources:
        return counts
    target = sqlite3.connect(cache_path)
    target.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, response TEXT NOT NULL)")
    opened = [(label, sqlite3.connect(f"file:{path}?mode=ro", uri=True), namespaces) for label, path, namespaces in sources]
    for state, questions in items:
        key = cache_key(NAMESPACE + cache_id, state, questions)
        if target.execute("SELECT 1 FROM answers WHERE key = ?", (key,)).fetchone():
            continue
        hit = None
        for label, con, namespaces in opened:
            for ns in namespaces:
                for sent, back in forms(questions):
                    row = con.execute("SELECT response FROM answers WHERE key = ?", (cache_key(ns + cache_id, state, sent),)).fetchone()
                    if row:
                        hit = label, json.loads(row[0]), back
                        break
                if hit:
                    break
            if hit:
                break
        if hit:
            label, resp, back = hit
            resp["answers"] = {back[n]: a for n, a in resp["answers"].items() if n in back}
            if set(resp["answers"]) != set(questions):
                continue
            resp["_replayed_from"] = label
            target.execute("INSERT OR REPLACE INTO answers VALUES (?, ?)", (key, json.dumps(resp, ensure_ascii=False)))
            counts[label] = counts.get(label, 0) + 1
    target.commit()
    target.close()
    for _, con, _ in opened:
        con.close()
    return counts


# ---------- The server ----------

class ServerFailed(Exception):
    """The server did not start, or answers as something other than the checkpoint asked for (rule 4)."""


class ServerGone(JevAuthError):
    """The server exited during collection. A JevAuthError, so that the backend stops the whole batch at once instead of letting thousands of
    requests run into connection errors and retries (rule 4 then redoes the step)."""

    def __init__(self) -> None:
        super().__init__(0, "the server exited during collection")

    def __str__(self) -> str:
        return "the server exited during collection"


class Watched(OpenModelClient):
    """Refuses every request once the server is known to have exited."""

    gone: threading.Event

    async def ask(self, state, questions):
        if self.gone.is_set():
            raise ServerGone()
        return await super().ask(state, questions)


def serve_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in FACTORY}
    return env | SERVE_ENV


def run_name(model: str, arm: str) -> str:
    return f"jaredpalmer/{model}@{WEIGHTS_TAG}" if arm == "released" else str(checkpoint(model, arm))


def check_card(card: dict, model: str, arm: str) -> None:
    """kev.serve falls back to another run (runs/smoke) when the one asked for has no head.pt; a fine-tuned checkpoint carries temperature 1.0."""
    if card.get("run") != run_name(model, arm):
        raise ServerFailed(f"the server answers as {card.get('run')!r}, not {run_name(model, arm)!r}")
    if arm != "released" and abs(float(card.get("temperature", -1)) - 1.0) > 1e-9:
        raise ServerFailed(f"a fine-tuned checkpoint should carry temperature 1.0, the server reports {card.get('temperature')}")
    if (card.get("prefix_cache") or {}).get("size") != 4:
        raise ServerFailed(f"the server is not at its factory settings: prefix cache {card.get('prefix_cache')}")


def port_in_use() -> bool:
    return bool(subprocess.run(["lsof", "-nP", "-t", f"-iTCP:{PORT}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout.strip())


class Server:
    def __init__(self, model: str, arm: str, log: Path) -> None:
        self.model, self.arm, self.log = model, arm, log
        self.proc: subprocess.Popen | None = None
        self.gone = threading.Event()
        self._stop = threading.Event()

    def start(self) -> dict:
        if port_in_use():
            raise ServerFailed(f"port {PORT} is already in use")
        command = ["sandbox-exec", "-p", SANDBOX, ".venv/bin/python", "-m", "kev.serve", "--run", run_name(self.model, self.arm), "--host", "127.0.0.1", "--port", str(PORT)]
        with self.log.open("a") as f:
            f.write(f"\n==== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {' '.join(command[3:])}\n")
        self.proc = subprocess.Popen(command, cwd=KEV, env=serve_env(), stdout=self.log.open("a"), stderr=subprocess.STDOUT)
        started = time.time()
        while time.time() - started < START_SECONDS:
            if self.proc.poll() is not None:
                raise ServerFailed(f"the server exited with code {self.proc.returncode} while starting, see {self.log}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as r:
                    card = json.load(r)["models"][0]
                break
            except OSError:
                time.sleep(2)
        else:
            raise ServerFailed(f"the server did not answer within {START_SECONDS} s")
        check_card(card, self.model, self.arm)
        self.seconds_to_start = round(time.time() - started, 1)
        threading.Thread(target=self._watch, daemon=True).start()
        return card

    def _watch(self) -> None:
        while not self._stop.is_set():
            if self.proc.poll() is not None:
                self.gone.set()
                return
            time.sleep(1)

    def stop(self) -> None:
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        for _ in range(30):  # wait until the port is really free before anything else uses the GPU
            if not port_in_use():
                return
            time.sleep(1)


# ---------- Asking ----------

def cache_id(model: str, arm: str) -> str:
    """Name under which answers are cached: the released checkpoint's own name, or this arm plus a digest of its weights, so that a retrained
    checkpoint never inherits the answers of an earlier one."""
    if arm == "released":
        return f"jaredpalmer/{model}@{WEIGHTS_TAG}"
    h = hashlib.sha256()
    for name in ("head.pt", "adapter_model.safetensors"):
        with (checkpoint(model, arm) / name).open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return f"p10/{model}/{arm}@{h.hexdigest()[:16]}"


def ask_set(backend: Backend, model: str, arm: str, name: str, out: Path, gone: threading.Event) -> dict:
    """Ask one request set (rule 5: up to 4 passes) and write its rows and summary. Raises ServerGone if the server exited meanwhile; nothing is written then."""
    got = rows(name)
    items = [(r["state"], r["questions"]) for r in got]
    replayed = replay(model, backend.model, items, out / "cache.sqlite") if arm == "released" else {}
    backend.reset()
    started = time.perf_counter()
    for attempt in range(1, PASSES + 1):
        answers = backend.ask_many(items)
        if not backend.meter.errors:
            break
    seconds = time.perf_counter() - started
    if gone.is_set():  # requests that failed because the server died are not rule-5 failures: the step is redone (rule 4)
        raise ServerGone()
    m = backend.meter
    with (out / f"{name}.jsonl").open("w") as f:
        for row, answer in zip(got, answers):
            f.write(json.dumps(record(row, answer), ensure_ascii=False) + "\n")
    summary = {"set": name, "arm": arm, "model": backend.model, "rows": len(got), "requests": m.calls, "fresh": m.fresh, "replayed": replayed,
               "errors": m.errors, "rows_without_answer": sum("_error" in a for a in answers), "passes": attempt, "seconds": round(seconds, 1),
               "concurrency": CONCURRENCY, "input_tokens": m.input_tokens, "latency_p50": m.latency(0.5),
               "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / f"{name}.run.json").write_text(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"   {name}: {summary['rows']} rows, {m.calls} requests, {m.fresh} sent, {sum(replayed.values())} replayed, {m.errors} without answer, {seconds:.0f} s", flush=True)
    return summary


def evaluate(model: str, arm: str) -> int:
    """One arm's evaluation step. Exit code 0: every set collected; 3: the step failed twice (rule 4) and what exists is kept; 2: nothing to evaluate."""
    out = out_dir(model, arm)
    out.mkdir(parents=True, exist_ok=True)
    state_file = out / "eval.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"model": model, "arm": arm, "attempts": []}
    todo = lambda: [s for s in sets(arm) if not (out / f"{s}.run.json").exists()]
    if not todo():
        state["complete"] = True
        state_file.write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n")
        print(f"{model} {arm}: every set already collected")
        return 0
    failures = sum(1 for a in state["attempts"] if a.get("failure"))
    if failures >= ATTEMPTS:
        print(f"{model} {arm}: the step already failed {failures} times (rule 4); keeping what exists")
        return 3
    if arm != "released" and (reason := not_trained(model, arm)):
        print(f"{model} {arm}: no checkpoint that counts ({reason}); nothing to evaluate")
        return 2
    ident = cache_id(model, arm)
    while failures < ATTEMPTS and todo():
        attempt = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cache_id": ident, "sets": []}
        server, backend = Server(model, arm, out / "serve.log"), None
        print(f"{model} {arm}: starting the server ({run_name(model, arm)}), {len(todo())} sets to ask", flush=True)
        try:
            attempt["card"] = server.start()
            attempt["seconds_to_start"] = server.seconds_to_start
            backend = Backend(f"p10-{model}-{arm}", lambda: _client(ident, server.gone), concurrency=CONCURRENCY, cache_path=out / "cache.sqlite", namespace=NAMESPACE)
            # Once the server is gone every queued request raises ServerGone; only the first reaches this function, the rest need not be logged
            backend._loop.set_exception_handler(lambda loop, context: None if isinstance(context.get("exception"), ServerGone) else loop.default_exception_handler(context))
            for name in todo():
                attempt["sets"].append(ask_set(backend, model, arm, name, out, server.gone)["set"])
        except (ServerFailed, ServerGone) as e:
            attempt["failure"] = str(e)
            failures += 1
            print(f"{model} {arm}: {e} (rule 4, failure {failures} of {ATTEMPTS})", flush=True)
            deviation(model, f"rule 4  {arm}: {e}; " + ("the evaluation step is done once more with a new server" if failures < ATTEMPTS else "second failure, what exists is kept"))
        finally:
            if backend:
                backend.close()
            server.stop()
            attempt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            state["attempts"].append(attempt)
            state["complete"] = not todo()
            state_file.write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n")
    return 0 if state["complete"] else 3


def deviation(model: str, text: str) -> None:
    """One line in the model's deviation log (p10_run.sh writes there too); the inventory and the report list them."""
    with (out_dir(model) / "deviations.txt").open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  {text}\n")


def _client(ident: str, gone: threading.Event) -> Watched:
    client = Watched(OpenModel("p10", "kev-latest", ident))
    client.gone = gone
    return client


# ---------- Checkpoints ----------

def training_records(arm: str) -> int:
    return sum(1 for _ in (D.OUT / "train" / f"{arm}.jsonl").open()) + REPLAY


def not_trained(model: str, arm: str) -> str:
    """'' if this arm's checkpoint counts, else the reason. Counts = complete files, the wrapper's end line, no dropped record, no truncated or
    rejected record, every requested record trained on (rule 3; a non-finite loss stops kev.train before it saves)."""
    path = checkpoint(model, arm)
    log = path.parent / f"{arm}.log"
    missing = [n for n in ("head.pt", "adapter_model.safetensors", "training_metrics.json") if not (path / n).exists()]
    if missing:
        return f"missing {', '.join(missing)}"
    text = log.read_text(errors="replace") if log.exists() else ""
    if f"end {model} {arm}" not in text or f"saved {path}" not in text:
        return f"{log.name} has no end line"
    if any(line.startswith("dropped ") for line in text.splitlines()) or "non-finite" in text:
        return "records were dropped or the loss was not finite"
    metrics = json.loads((path / "training_metrics.json").read_text())
    if metrics.get("truncated_records") or metrics.get("rejected_records"):
        return f"truncated {metrics.get('truncated_records')} / rejected {metrics.get('rejected_records')} records"
    if metrics.get("requested_records") != training_records(arm):
        return f"trained on {metrics.get('requested_records')} records, expected {training_records(arm)}"
    return ""


# ---------- Verification, plan, inventory ----------

def sha256(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), data.count(b"\n")


def verify() -> int:
    bad = 0
    for rel, digest in INPUTS.items():
        got = sha256(ROOT / rel)[0] if (ROOT / rel).exists() else "missing"
        if got != digest:
            print(f"input changed: {rel} ({got[:16]})")
            bad += 1
    frozen = BUILT_REHEARSAL if D.REHEARSAL else BUILT
    for rel, (digest, lines) in frozen.items():
        path = D.OUT / rel
        got = sha256(path) if path.exists() else ("missing", 0)
        if got != (digest, lines):
            print(f"built file differs from the frozen one: {rel} ({got[0][:16]}, {got[1]} lines; frozen {digest[:16]}, {lines} lines)")
            bad += 1
    if not frozen:
        print("no frozen checksums for the built files yet")
        bad += 1
    print(f"verify: {len(INPUTS) + len(frozen) - bad} of {len(INPUTS) + len(frozen)} files as frozen")
    return bad


def plan(model: str) -> None:
    total = 0
    sources = replay_sources(model)
    print(f"{model}: replay sources {', '.join(f'{label} {path.relative_to(ROOT)}' for label, path, _ in sources) or 'none'}")
    for arm in ARMS:
        print(f"== {arm}")
        for name in sets(arm):
            got = rows(name)
            distinct = {cache_key("", r["state"], r["questions"]) for r in got}
            line = f"   {name:<20}{len(got):>7,} rows {len(distinct):>7,} requests"
            if arm == "released" and sources:
                found = _found(model, got, sources)
                line += f"   {found:>7,} in earlier caches, {len(distinct) - found:,} to ask"
                total += len(distinct) - found
            else:
                total += len(distinct)
            print(line)
    print(f"{total:,} requests to send in all")


def _found(model: str, got: list[dict], sources: list) -> int:
    """How many distinct requests an earlier phase already asked the released checkpoint (only existence is checked; no answer is read)."""
    ident, opened, seen, found = f"jaredpalmer/{model}@{WEIGHTS_TAG}", [(sqlite3.connect(f"file:{p}?mode=ro", uri=True), ns) for _, p, ns in sources], set(), 0
    for r in got:
        k = cache_key("", r["state"], r["questions"])
        if k in seen:
            continue
        seen.add(k)
        found += any(con.execute("SELECT 1 FROM answers WHERE key = ?", (cache_key(n + ident, r["state"], sent),)).fetchone()
                     for con, nss in opened for n in nss for sent, _ in forms(r["questions"]))
    for con, _ in opened:
        con.close()
    return found


def status(model: str, arm: str) -> str:
    """complete, missing (settled for good: training failed twice, rule 2, or the evaluation step failed twice, rule 4) or pending."""
    out = out_dir(model, arm)
    if (out / "MISSING").exists():
        return "missing: " + (out / "MISSING").read_text().strip()
    if arm != "released" and not_trained(model, arm):
        return "pending"
    state = json.loads((out / "eval.json").read_text()) if (out / "eval.json").exists() else {}
    if state.get("complete"):
        return "complete"
    if sum(1 for a in state.get("attempts", []) if a.get("failure")) >= ATTEMPTS:
        return "missing: the evaluation step failed twice (rule 4); " + "; ".join(a["failure"] for a in state["attempts"] if a.get("failure"))
    return "pending"


def check(model: str) -> int:
    """Inventory of one model: training attempts, the checkpoint of every arm, every request set; deviations as logged by p10_run.sh. No answer is looked at.
    Returns the number of arms still pending (neither complete nor settled as missing)."""
    base = out_dir(model)
    print(f"inventory {model}{' (rehearsal)' if D.REHEARSAL else ''}")
    training = base / "training.tsv"
    if training.exists():
        print("   training attempts (arm, attempt, outcome, seconds, peak memory GB):")
        for line in training.read_text().splitlines()[1:]:
            arm, attempt, outcome, seconds, peak = line.split("\t")[:5]
            print(f"      {arm:<20}{attempt:>3}  {outcome:<34}{seconds:>8}{float(peak) / 2**30:>8.1f}")
    complete = pending = 0
    for arm in ARMS:
        reason = "" if arm == "released" else not_trained(model, arm)
        settled = status(model, arm)
        pending += settled == "pending"
        state = json.loads((base / arm / "eval.json").read_text()) if (base / arm / "eval.json").exists() else {}
        attempts = state.get("attempts", [])
        summaries = [json.loads((base / arm / f"{s}.run.json").read_text()) for s in sets(arm) if (base / arm / f"{s}.run.json").exists()]
        complete += settled == "complete"
        if settled.startswith("missing"):
            print(f"   {arm:<20}{settled}")
        elif reason:
            print(f"   {arm:<20}checkpoint does not count yet: {reason}")
        failures = [a["failure"] for a in attempts if a.get("failure")]
        print(f"   {arm:<20}{len(summaries)} of {len(sets(arm))} sets, {sum(s['requests'] for s in summaries):,} requests, "
              f"{sum(s['fresh'] for s in summaries):,} sent, {sum(sum(s['replayed'].values()) for s in summaries):,} replayed, "
              f"{sum(s['rows_without_answer'] for s in summaries):,} rows without answer, {sum(s['seconds'] for s in summaries):,.0f} s asking, "
              f"{len(attempts)} server start(s)" + (f"; failures: {'; '.join(failures)}" if failures else ""))
        for s in summaries:
            if s["rows_without_answer"] or s["passes"] > 1:
                print(f"      {s['set']}: {s['rows_without_answer']} rows without answer after {s['passes']} passes")
    deviations = base / "deviations.txt"
    if deviations.exists() and deviations.read_text().strip():
        print("   deviations:")
        for line in deviations.read_text().splitlines():
            print(f"      {line}")
    else:
        print("   deviations: none")
    print(f"   {complete} of {len(ARMS)} arms complete, {len(ARMS) - complete - pending} missing for good, {pending} pending")
    return pending


def main(argv: list[str]) -> int:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # so that a stopped evaluation stops its server on the way out
    command, rest = (argv[:1] or [""])[0], argv[1:]
    if command == "arm" and len(rest) == 2 and rest[0] in MODELS and rest[1] in ARMS:
        return evaluate(*rest)
    if command == "trained" and len(rest) == 2 and rest[0] in MODELS and rest[1] in TRAINED:
        reason = not_trained(*rest)
        print(reason or "complete")
        return 1 if reason else 0
    if command == "verify" and not rest:
        return 1 if verify() else 0
    if command == "plan" and len(rest) == 1 and rest[0] in MODELS:
        plan(rest[0])
        return 0
    if command == "check" and len(rest) == 1 and rest[0] in MODELS:
        return 1 if check(rest[0]) else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
