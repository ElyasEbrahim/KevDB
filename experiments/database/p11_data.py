"""Phase 11 training data: why did fine-tuned Kev-4B not improve on dblp-acm in phase 10? Criteria: experiments/database/protocol/p11.txt.

Every arm holds out dblp-acm, like phase 10's loo-dblp-acm (the control), and trains on its 6,000 product pairs plus one added ingredient:
    words    the same 6,000 product pairs, each asked with one of five wordings of the product question (no new pairs)
    domains  plus every training pair of three other domains, each with its own question: beer (268), restaurants (567), songs (321)
    pubs     plus 2,000 pairs of another publication dataset, DBLP-GoogleScholar, asked with dblp-acm's own publication question

    python experiments/database/p11_data.py build      # the three training files
    python experiments/database/p11_data.py pools      # temperature labels of the added datasets (the product pools are phase 10's)
    python experiments/database/p11_data.py check      # counts, matches, and that no training record occurs in dblp-acm's test split

A training record is exactly the request sent at evaluation plus the human label, as in phase 10. The product pairs are phase 10's own
(p10_data.train_pairs), so the control differs from each arm only by the ingredient. The added data are downloaded by p11_fetch.sh. No output
of Jev is involved. With P11_REHEARSAL=1 every row is made up (the questions are the real ones) and results go to out/database/p11_rehearsal.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

REHEARSAL = os.environ.get("P11_REHEARSAL") == "1"
if REHEARSAL:  # phase 10's builders then produce made-up product and publication rows
    os.environ["P10_REHEARSAL"] = "1"

import p10_data as P10  # noqa: E402
from paths import OUT as RESULTS, ROOT  # noqa: E402
from kevdb import Question  # noqa: E402

OUT = RESULTS / ("p11_rehearsal" if REHEARSAL else "p11")
HELD_OUT = "dblp-acm"
PRODUCTS = P10.ARMS["loo-dblp-acm"]  # abt-buy, amazon-google, walmart-amazon
DOMAINS = ("beer", "fodors-zagats", "itunes-amazon")
PUBS = "dblp-googlescholar"
ARMS = {"words": PRODUCTS, "domains": PRODUCTS + DOMAINS, "pubs": PRODUCTS + (PUBS,)}
N_PUBS = 24 if REHEARSAL else 2000
POOL_LABELS = P10.POOL_LABELS  # 200 per dataset (fewer where the validation split is smaller)
WORDINGS = 5

# First drafts, written on 2026-10-20 before any model saw these data; never edited against answers
QUESTIONS = {
    "beer": Question(
        "noul",
        "Do `record_a` and `record_b` describe the same beer?",
        {
            "true": "The same beer from the same brewery, allowing for abbreviations, typos, and differently written style names or alcohol content.",
            "false": "Different beers, including different beers from the same brewery or beers with similar names.",
        },
    ),
    "restaurant": Question(
        "noul",
        "Do `record_a` and `record_b` describe the same restaurant?",
        {
            "true": "The same restaurant at the same place, allowing for abbreviations, differently formatted addresses or phone numbers, and different cuisine labels.",
            "false": "Different restaurants, including different branches of the same chain or restaurants with similar names.",
        },
    ),
    "song": Question(
        "noul",
        "Do `record_a` and `record_b` describe the same song recording?",
        {
            "true": "The same track by the same artist, allowing for differently formatted titles, album names, prices, or release dates.",
            "false": "Different recordings, including other songs by the same artist, other versions of the same song, or songs with similar titles.",
        },
    ),
}
KIND = {"beer": "beer", "fodors-zagats": "restaurant", "itunes-amazon": "song", PUBS: "publication"}
# The product question as phase 10 asked it (wording 0) and four rewordings with the same meaning, for the words arm
PRODUCT_WORDINGS = [P10.KINDS["product"]] + [
    Question("noul", instructions, {"true": true, "false": false}) for instructions, true, false in (
        ("Are `record_a` and `record_b` two listings of one and the same product?",
         "Both listings sell the identical item: same manufacturer and same model number or edition. Wording, missing details and price may differ.",
         "The listings are for different items, for example another model, size, capacity, colour, edition or version."),
        ("Would a buyer of `record_a` receive the same item as a buyer of `record_b`?",
         "Yes: the same item from the same maker, the same model or edition; descriptions, missing fields and prices can differ.",
         "No: a different item, such as another model, size, capacity, colour, edition or version of something similar."),
        ("Is `record_b` the same product as `record_a`?",
         "Same brand and model or edition, even if worded differently, with missing fields or another price.",
         "A different product: another model, size, capacity, colour, edition or version."),
        ("Do the catalogue entries `record_a` and `record_b` refer to the identical product?",
         "Identical product: the same make and the same model or edition; ignore differences in wording, missing fields and price.",
         "Not identical: a different model, size, capacity, colour, edition or version of a similar product."),
    )
]


def sha(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def question(dataset: str) -> Question:
    return P10.KINDS["publication"] if dataset == PUBS else QUESTIONS[KIND[dataset]]


def wording(dataset: str, pair_id: str) -> int:
    """Which of the five product wordings a pair of the words arm is asked with: fixed by hash, about a fifth each."""
    return int(sha("p11", "words", dataset, pair_id), 16) % WORDINGS


# ---------- Made-up rows of the added domains (rehearsals) ----------

def made_pairs(dataset: str, split: str, n: int) -> list[dict]:
    kind = KIND[dataset]
    if kind == "publication":
        return [r | {"raw": (f"{r['id']}-a", f"{r['id']}-b")} for r in P10.made_pairs(f"{dataset}-{split}", n, "publication")]
    rng, rows = random.Random(f"p11:{dataset}:{split}"), []
    for i in range(n):
        same = rng.random() < 0.2
        k = rng.randrange(100, 999)
        if kind == "beer":
            a = {"Beer_Name": f"Norvale Hollow {k} Ale", "Brew_Factory_Name": "Norvale Brewing", "Style": "American Pale Ale", "ABV": f"{rng.randrange(40, 90) / 10}%"}
            b = {"Beer_Name": f"Norvale Hollow {k if same else k + 1} Pale", "Brew_Factory_Name": "Norvale Brewing Co.", "Style": "Pale Ale", "ABV": a["ABV"] if same else "5.1%"}
        elif kind == "restaurant":
            a = {"name": f"kestrel kitchen {k}", "addr": f"{k} lindqvist ave.", "city": "ombra city", "phone": f"555-0{k}", "type": "bistro"}
            b = {"name": f"kestrel kitchen {k if same else k + 3}", "addr": f"{k if same else k + 3} lindqvist avenue", "city": "ombra city", "phone": f"555 0{k if same else k + 3}", "type": "french"}
        else:
            a = {"Song_Name": f"Tavish Lights {k}", "Artist_Name": "The Brackwells", "Album_Name": "Low Tide", "Genre": "Rock", "Price": "$0.99", "Time": "3:41"}
            b = {"Song_Name": f"Tavish Lights {k}" if same else f"Tavish Lights {k + 2} (Live)", "Artist_Name": "Brackwells", "Album_Name": "Low Tide",
                 "Genre": "Pop", "Price": "$1.29", "Time": "3:41" if same else "4:02"}
        rows.append({"id": f"made-up-{dataset}-{split}-{i}", "record_a": a, "record_b": b, "label": int(same), "raw": (f"{dataset}-{split}-{i}-a", f"{dataset}-{split}-{i}-b")})
    return rows


# ---------- Real rows ----------

def load_pairs(dataset: str, split: str) -> list[dict]:
    """Rows of an added dataset; "raw" keeps both records as written in the file, for the overlap check."""
    if REHEARSAL:
        return made_pairs(dataset, split, 60 if dataset == PUBS else 12)
    path = ROOT / "experiments" / "entity_match" / "data" / dataset / f"{split}.txt"
    if not path.exists():
        sys.exit(f"{path} is missing: run ./experiments/database/p11_fetch.sh first")
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        left, right, label = line.split("\t")
        rows.append({"id": f"{split}-{i}", "record_a": P10.parse_record(left), "record_b": P10.parse_record(right), "label": int(label),
                     "raw": (left.strip(), right.strip())})
    return rows


def held_out_records() -> set[str]:
    """Every record of dblp-acm's test split, as written in the file (both sides)."""
    if REHEARSAL:
        return set()
    path = ROOT / "experiments" / "entity_match" / "data" / HELD_OUT / "test.txt"
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        left, right, _ = line.split("\t")
        out |= {left.strip(), right.strip()}
    return out


def added_pairs(dataset: str) -> tuple[list[dict], int]:
    """(training pairs of an added dataset, pairs dropped because a record occurs in dblp-acm's test split). DBLP-GoogleScholar: the first
    N_PUBS of the remaining pairs in the order of sha("p11|<dataset>|<id>"); the three small domains: every pair."""
    held = held_out_records()
    pairs = load_pairs(dataset, "train")
    kept = [p for p in pairs if not ({*p["raw"]} & held)]
    if dataset == PUBS:
        kept = sorted(kept, key=lambda p: sha("p11", dataset, p["id"]))[:N_PUBS]
    return kept, len(pairs) - len([p for p in pairs if not ({*p["raw"]} & held)])


def pool_pairs(dataset: str) -> list[dict]:
    """Temperature labels of an added dataset: the first POOL_LABELS validation pairs in the order of sha("p11-pool|<dataset>|<id>")."""
    return sorted(load_pairs(dataset, "valid"), key=lambda p: sha("p11-pool", dataset, p["id"]))[:POOL_LABELS]


def request(pair: dict, q: Question) -> dict:
    state = {name: pair[name] for name in ("record_a", "record_b") if pair[name]}
    return {"state": state, "questions": {"same": q.payload()}}


def labelled(pair: dict, q: Question) -> dict:
    record = request(pair, q)
    record["questions"]["same"]["label"] = bool(pair["label"])
    return record


def arm_records(arm: str) -> tuple[list[dict], list[str]]:
    rows, counts = [], []
    for dataset in PRODUCTS:
        pairs = P10.train_pairs(dataset)
        if arm == "words":
            rows += [labelled(p, PRODUCT_WORDINGS[wording(dataset, p["id"])]) for p in pairs]
        else:
            rows += [P10.labelled(p, dataset) for p in pairs]
        counts.append(f"{dataset} {len(pairs)} ({sum(p['label'] for p in pairs)} matches)")
    for dataset in ARMS[arm][len(PRODUCTS):]:
        pairs, dropped = added_pairs(dataset)
        rows += [labelled(p, question(dataset)) for p in pairs]
        counts.append(f"{dataset} {len(pairs)} ({sum(p['label'] for p in pairs)} matches; {dropped} dropped for sharing a record with {HELD_OUT}'s test split)")
    return rows, counts


def build() -> None:
    for arm in ARMS:
        rows, counts = arm_records(arm)
        digest = P10.write(OUT / "train" / f"{arm}.jsonl", rows)
        print(f"{arm}: {len(rows)} records; {'; '.join(counts)}; sha256 {digest[:16]}")


def pools() -> None:
    for dataset in DOMAINS + (PUBS,):
        pairs = pool_pairs(dataset)
        P10.write(OUT / "pool" / f"{dataset}.jsonl", [labelled(p, question(dataset)) for p in pairs])
        print(f"{dataset}: {len(pairs)} pairs ({sum(p['label'] for p in pairs)} matches)")


def check() -> int:
    """Counting only. No training record may be a record of dblp-acm's test split."""
    held, bad = held_out_records(), 0
    for arm in ARMS:
        rows, counts = arm_records(arm)
        print(f"{arm}: {len(rows)} training records; {'; '.join(counts)}")
    for dataset in DOMAINS + (PUBS,):
        pairs, _ = added_pairs(dataset)
        bad += sum(bool({*p["raw"]} & held) for p in pairs)
    for dataset in PRODUCTS:  # phase 10 found no record shared between datasets; checked again here on the raw files
        if REHEARSAL:
            break
        path = ROOT / "experiments" / "entity_match" / "data" / dataset / "train.txt"
        bad += sum(bool({part.strip() for part in line.split("\t")[:2]} & held) for line in path.read_text(encoding="utf-8").splitlines())
    print(f"training records that occur in {HELD_OUT}'s test split: {bad}")
    return bad


if __name__ == "__main__":
    command = {"build": build, "pools": pools, "check": lambda: sys.exit(1 if check() else 0)}.get(sys.argv[1] if len(sys.argv) == 2 else "")
    if not command:
        sys.exit(__doc__)
    command()
