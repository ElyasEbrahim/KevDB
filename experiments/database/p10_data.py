"""Phase 10 training data: turn the training splits of the four Magellan record-matching datasets into files for fine-tuning Kev. Criteria: experiments/database/protocol/p10.txt.

    python experiments/database/p10_data.py build            # all five arms: four leave-one-out arms (loo-<held-out dataset>) and one arm using all four (all)
    python experiments/database/p10_data.py build loo-abt-buy
    python experiments/database/p10_data.py pools            # human labels for fitting the temperature: each dataset's fixed 200 validation pairs (the same 200 as in phase 7)

Each training record's state, question and criteria are identical to the request sent at evaluation (the first-draft question of experiments/entity_match/run.py), plus one human label.
Each dataset contributes the first N_PER_DATASET pairs of its training split ordered by sha256("p10|<dataset>|<id>"), keeping the original match / non-match ratio, without resampling.
The training splits are downloaded by experiments/database/p10_fetch.sh. No output of Jev is involved.

With the environment variable P10_REHEARSAL=1 this is a rehearsal: the same steps, but every row is made up below (the questions are the real ones), with results under out/database/p10_rehearsal.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

from paths import OUT as RESULTS, ROOT

sys.path.insert(0, str(ROOT / "experiments" / "entity_match"))
from run import DATASETS, FIELD, KINDS  # noqa: E402

REHEARSAL = os.environ.get("P10_REHEARSAL") == "1"
OUT = RESULTS / ("p10_rehearsal" if REHEARSAL else "p10")
EM = ("abt-buy", "amazon-google", "walmart-amazon", "dblp-acm")
# A leave-one-out arm trains on the other three and is evaluated on the held-out dataset's test split; all trains on all four, the reference for "with in-domain labels"
ARMS = {f"loo-{held}": tuple(d for d in EM if d != held) for held in EM} | {"all": EM}
# Pairs per dataset for training. Set from Kev's author's experience (5,219 records gave a clear gain, 400 a gain within noise) and this machine's training speed; see section 7 of p10_plan.md
N_PER_DATASET = 24 if REHEARSAL else 2000
# Human labels for fitting the temperature: the first 200 pairs of phase 7's label pool (the first 1,000 validation pairs ordered by p7_key), the same 200 used to fit calibration in phase 7
POOL_LABELS = 12 if REHEARSAL else 200


def sha(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def p7_key(dataset: str, pair_id: str) -> str:
    """The key phase 7 used to order its label pool (em_rows in experiments/database/p7_data.py: p2_collect.sha(id, "p7:<dataset>")), copied here to avoid pulling in all of phase 7's dependencies."""
    return hashlib.sha256(f"p7:{dataset}{pair_id}".encode()).hexdigest()[:16]


def parse_record(text: str) -> dict[str, str]:
    parts = FIELD.split(text.strip())[1:]
    return {name: value.strip() for name, value in zip(parts[::2], parts[1::2]) if value.strip()}


# ---------- Made-up rows (for rehearsals) ----------

BRANDS = ["Norvale", "Kestrel", "Ombra", "Tavish", "Lumeo", "Brackwell"]
ITEMS = ["wireless optical mouse", "USB-C docking station", "portable SSD", "noise cancelling headphones", "27 inch monitor", "mechanical keyboard", "laser printer"]
BLURBS = ["ergonomic design with adjustable settings", "includes a two year limited warranty", "compatible with windows and mac", "energy efficient low power mode",
          "compact travel friendly housing", "high speed data transfer up to 10 gbps", "retail packaging with quick start guide", "matte finish that resists fingerprints"]
TOPICS = ["query optimization", "entity resolution", "stream processing", "index selection", "transaction scheduling", "approximate aggregation"]
SURNAMES = ["Arlund", "Besker", "Corwin", "Dalmas", "Eitner", "Faroux", "Grell", "Hobart"]


def made_pairs(seed: str, n: int, kind: str) -> list[dict]:
    """Made-up record pairs shaped like the four real datasets (a few hundred characters for both records together), with content unrelated to any dataset."""
    rng, rows = random.Random("p10:" + seed), []
    pick = rng.choice
    for i in range(n):
        same = rng.random() < 0.15
        if kind == "publication":
            topic, year, a1, a2 = pick(TOPICS), rng.randrange(1994, 2004), pick(SURNAMES), pick(SURNAMES)
            title = f"a {pick(['cost based', 'adaptive', 'scalable', 'incremental'])} approach to {topic} in {pick(['relational', 'distributed', 'main memory'])} database systems {seed} {i}"
            a = {"title": title, "authors": f"{a1[0].lower()} {a1.lower()} , {a2[0].lower()} {a2.lower()}", "venue": "sigmod conference", "year": str(year)}
            b = {"title": title if same else title.replace(topic, pick([t for t in TOPICS if t != topic])), "authors": f"{a1.lower()} , {a2.lower()}", "venue": "international conference on management of data", "year": str(year if same else year + 1)}
        else:
            brand, item, k = pick(BRANDS), pick(ITEMS), rng.randrange(100, 999)
            blurb = " ".join(rng.sample(BLURBS, rng.randrange(1, 5)))
            a = {"title": f"{brand} {item} model {k} black {seed} {i}", "category": "computer accessories", "brand": brand, "modelno": f"{brand[:2].upper()}{k}", "description": blurb, "price": f"{rng.randrange(20, 300)}.99"}
            b = {"title": f"{brand} {item} {k if same else k + 1} ( black ) retail pack", "category": "electronics - general", "brand": brand, "modelno": f"{brand[:2].upper()}-{k if same else k + 1}", "description": " ".join(rng.sample(BLURBS, 2))}
        rows.append({"id": f"made-up-{seed}-{i}", "record_a": a, "record_b": b, "label": int(same)})
    return rows


# ---------- Real rows ----------

def load_pairs(dataset: str, split: str) -> list[dict]:
    if REHEARSAL:
        return made_pairs(f"{dataset}-{split}", 60, DATASETS[dataset])
    path = ROOT / "experiments" / "entity_match" / "data" / dataset / f"{split}.txt"
    if not path.exists():
        sys.exit(f"{path} is missing: run ./experiments/database/p10_fetch.sh first")
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        left, right, label = line.split("\t")
        rows.append({"id": f"{split}-{i}", "record_a": parse_record(left), "record_b": parse_record(right), "label": int(label)})
    return rows


def train_pairs(dataset: str) -> list[dict]:
    """The pairs of this dataset that go into training: the first N_PER_DATASET in the fixed hash order."""
    return sorted(load_pairs(dataset, "train"), key=lambda p: sha("p10", dataset, p["id"]))[:N_PER_DATASET]


def pool_pairs(dataset: str) -> list[dict]:
    """Human labels for fitting the temperature: the same pairs as the first 200 of phase 7's em_rows(dataset, "pool")."""
    return sorted(load_pairs(dataset, "valid"), key=lambda p: p7_key(dataset, p["id"]))[:POOL_LABELS]


def request(pair: dict, dataset: str) -> dict:
    """The same state and question as the request sent at evaluation (Task.state_of: non-empty fields only)."""
    state = {name: pair[name] for name in ("record_a", "record_b") if pair[name]}
    return {"state": state, "questions": {"same": KINDS[DATASETS[dataset]].payload()}}


def labelled(pair: dict, dataset: str) -> dict:
    record = request(pair, dataset)
    record["questions"]["same"]["label"] = bool(pair["label"])
    return record


def write(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def build(arms: list[str]) -> None:
    for arm in arms:
        rows, counts = [], []
        for dataset in ARMS[arm]:
            pairs = train_pairs(dataset)
            rows += [labelled(p, dataset) for p in pairs]
            counts.append(f"{dataset} {len(pairs)} pairs ({sum(p['label'] for p in pairs)} matches)")
        # The training script shuffles with its own seed; here the datasets are simply written one after another
        digest = write(OUT / "train" / f"{arm}.jsonl", rows)
        print(f"{arm}: {len(rows)} records, {'; '.join(counts)}; sha256 {digest[:16]}")


def pools() -> None:
    for dataset in EM:
        pairs = pool_pairs(dataset)
        write(OUT / "pool" / f"{dataset}.jsonl", [labelled(p, dataset) for p in pairs])
        print(f"{dataset}: {len(pairs)} pairs ({sum(p['label'] for p in pairs)} matches)")


if __name__ == "__main__":
    command, rest = (sys.argv[1:2] or [""])[0], sys.argv[2:]
    if command == "build" and all(a in ARMS for a in rest):
        build(rest or list(ARMS))
    elif command == "pools" and not rest:
        pools()
    else:
        sys.exit(f"usage: python experiments/database/p10_data.py build [{' | '.join(ARMS)}]   or   python experiments/database/p10_data.py pools")
