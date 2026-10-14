"""Follow-up: with a truly stronger upper-tier model (Opus 5.5), find out why the cascade failed on the movie and restaurant reviews.

The budget is the whole OpenRouter balance. Three parts are collected in order; the first two have a fixed size, the third runs as many sentences as the remaining money allows:

    movie   1,850 movie reviews, question "positive?". The first 1,850 of the 5,000 texts the open models were also asked in phase 2, so every source can be paired
    em      positive control: the Walmart-Amazon product-matching test set, all 214 pairs Jev is unsure about plus a sample of 250 confident ones (as in experiments/table_match/strong.py)
    mams    restaurant reviews, 21 questions packed, in phase 4's fixed order of 1,000 sentences, starting with the first

The questions to answer and the reading rules are in experiments/database/protocol/p5.txt, written before collection.

    python experiments/database/p5_strong.py collect
    python experiments/database/p5_strong.py report
"""

from __future__ import annotations

import json
import random
import sys
import time
import urllib.request

from kevdb import Task
from kevdb.cli import load_dotenv
from kevdb.client import JevAuthError

from kevdb import llm_backend
from p2_collect import QUESTIONS, sha, texts
from p4_collect import PREDICATES, question, sentences
from paths import OUT as RESULTS, ROOT

OUT = RESULTS / "p5"
EM = "walmart-amazon"
N_MOVIE = 1850
RESERVE = 0.15  # keep a margin: requests are rejected outright when the balance runs out
WORST_MAMS_CALL = 0.017  # the most expensive packed request in the probe cost $0.0158


def credits_left() -> float:
    import os

    req = urllib.request.Request("https://openrouter.ai/api/v1/credits", headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    data = json.loads(urllib.request.urlopen(req, timeout=30).read())["data"]
    return data["total_credits"] - data["total_usage"]


def em_pairs() -> list[dict]:
    """All unsure pairs, plus 100 + 150 sampled confident pairs. Same selection as experiments/table_match/strong.py used for Abt-Buy."""
    sys.path.insert(0, str(ROOT / "experiments" / "entity_match"))
    from run import load_pairs

    p = {r["id"]: r["p"] for r in map(json.loads, (ROOT / f"out/entity_match/{EM}.test.jev-1.13.0.jsonl").read_text().splitlines())}
    stratum = lambda q: "sure_yes" if q >= 0.9 else "sure_no" if q <= 0.1 else "unsure"
    by: dict[str, list[dict]] = {"unsure": [], "sure_yes": [], "sure_no": []}
    for pair in load_pairs(EM, "test"):
        by[stratum(p[pair["id"]])].append({**pair, "p": p[pair["id"]], "stratum": stratum(p[pair["id"]])})
    rng = random.Random(0)
    return by["unsure"] + rng.sample(by["sure_yes"], 100) + rng.sample(by["sure_no"], 150)


def em_question() -> dict:
    sys.path.insert(0, str(ROOT / "experiments" / "entity_match"))
    from run import KINDS

    return {"same": KINDS["product"].payload()}


def collect() -> None:
    load_dotenv()
    OUT.mkdir(parents=True, exist_ok=True)
    start = credits_left()
    backend = llm_backend("opus", concurrency=12)
    left = lambda: start - backend.spent - RESERVE
    log: dict = {"start_balance": start, "model": backend.model}

    def ask(items: list, namespace: str) -> list[dict]:
        """Ask a batch; failures are re-asked twice (failures are not cached)."""
        backend.namespace = namespace
        for _ in range(3):
            answers = backend.ask_many(items)
            if not any("_error" in a for a in answers):
                break
        return answers

    def phase(name: str, started: float, answers: list[dict], spent_before: float) -> None:
        done = [a for a in answers if "_error" not in a]
        lat = sorted(a["_latency"] for a in done)
        log[name] = {"requests": len(answers), "failed": len(answers) - len(done), "seconds": round(time.perf_counter() - started, 1),
                     "cost_usd": round(sum(a["usage"]["cost"] for a in done), 4), "new_spend_usd": round(backend.spent - spent_before, 4),
                     "latency_p50": lat[len(lat) // 2] if lat else None,
                     "input_tokens": sum(a["usage"]["input_tokens"] for a in done), "output_tokens": sum(a["usage"]["output_tokens"] for a in done)}
        print(name, log[name], f"balance still usable ${left():.2f}", flush=True)
        (OUT / "collect.json").write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n")

    # 1. Movies
    rows = texts(N_MOVIE, "open")
    t, before = time.perf_counter(), backend.spent
    answers = ask([({"review": x}, {"q": {"type": "noul", "instructions": QUESTIONS["positive"]}}) for x in rows], "p2:")
    with (OUT / "movie.opus.jsonl").open("w") as f:
        for x, a in zip(rows, answers):
            f.write(json.dumps({"text": sha(x), "p": None if "_error" in a else a["answers"]["q"]["noul"], "error": a.get("_error")}) + "\n")
    phase("movie", t, answers, before)

    # 2. Product matching (positive control)
    pairs = em_pairs()
    task = Task(name="em", input="", questions={}, state_fields=["record_a", "record_b"])
    t, before = time.perf_counter(), backend.spent
    answers = ask([(task.state_of(pair), em_question()) for pair in pairs], "p5.em:")
    with (OUT / "em.opus.jsonl").open("w") as f:
        for pair, a in zip(pairs, answers):
            f.write(json.dumps({"id": pair["id"], "label": pair["label"], "stratum": pair["stratum"], "jev_p": pair["p"],
                                "p": None if "_error" in a else a["answers"]["same"]["noul"], "error": a.get("_error")}) + "\n")
    phase("em", t, answers, before)

    # 3. Restaurant reviews: as many sentences as the remaining money allows; near the end, only a few sentences per batch
    todo = sentences(1000)
    packed = {p: question(p) for p in PREDICATES}
    done: list[tuple[dict, dict]] = []
    t, before = time.perf_counter(), backend.spent
    # OpenRouter reserves cost per in-flight request based on max_tokens, and the reserved total of concurrent requests may not exceed a small share of the balance, otherwise 402.
    # So max_tokens is lowered from 8000 here (a packed answer with thinking measured about 340 tokens); on rejection, first reduce concurrency, then lower the limit further.
    # After a rejection, wait for in-flight requests to finish: they still hold credit, and retrying at once only gets rejected again (that is how the second collection stopped early).
    limits, level, cap, strikes = (2000, 1200, 1000), 0, 12 if left() > 3 else 3, 0
    backend.client.max_tokens = limits[level]
    while len(done) < len(todo):
        k = min(cap, int(left() / WORST_MAMS_CALL), len(todo) - len(done))
        if k < 1:
            break
        batch = todo[len(done):len(done) + k]
        try:
            got = ask([({"sentence": r["text"]}, packed) for r in batch], "p4:")
        except JevAuthError as e:
            print(f"rejected (batch of {k} sentences, max_tokens {limits[level]}, balance still usable ${left():.2f}): {str(e)[:90]}", flush=True)
            time.sleep(15)
            if k > 1:
                cap = max(1, k // 2)
            elif level + 1 < len(limits):
                level += 1
                backend.client.max_tokens = limits[level]
            else:
                strikes += 1
                if strikes == 3:
                    break  # one sentence at a time, the smallest limit, and still rejected after three rounds of waiting: the balance cannot be used any further. Everything asked so far is cached
            continue
        strikes = 0
        done += list(zip(batch, got))
    with (OUT / "mams.opus.packed.jsonl").open("w") as f:
        for r, a in done:
            f.write(json.dumps({"sid": r["sid"], "p": None if "_error" in a else {key: a["answers"][key]["noul"] for key in PREDICATES}, "error": a.get("_error")}) + "\n")
    phase("mams", t, [a for _, a in done], before)
    log["sentences"] = len(done)
    log["end_balance"] = credits_left()
    log["spent_usd"] = round(backend.spent, 4)
    (OUT / "collect.json").write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n")
    print(f"spent ${backend.spent:.2f} in total, balance from ${start:.2f} to ${log['end_balance']:.2f}", flush=True)
    backend.close()


if __name__ == "__main__":
    if sys.argv[1:] == ["collect"]:
        collect()
    elif sys.argv[1:] == ["report"]:
        from p5_report import report

        report()
    else:
        sys.exit(__doc__)
