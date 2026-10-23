"""Phase 12 data: with how many of their own labels does Kev-4B reach Jev's zero-shot level on a matching dataset it has not seen?
Criteria: experiments/database/protocol/p12.txt.

    python experiments/database/p12_data.py build    every run's training file and temperature slice, and each dataset's test pairs
    python experiments/database/p12_data.py check    counts, and that no record of a test pair occurs in any label

Every record of a dataset, as written in the file, goes to the test side or the label side by the parity of sha256("p12|<dataset>|<record>").
Test pairs: the pairs of the test split, and of the 600 validation pairs Jev was asked (experiments/entity_match/run.py --sample 600), whose
two records are both on the test side. Label pool: the training split's pairs whose two records are both on the label side, in the order of
sha256("p12|<dataset>|<pair id>"). A budget of n labels is the first n pairs of the pool, so budgets are nested; its first 80% are trained
on and its last 20% fit the temperature. A training record is exactly the request sent at evaluation (phase 10's first-draft question) plus
the human label. No output of Jev is involved: Jev's cached file is opened only to check which validation pairs it was asked.

With P12_REHEARSAL=1 every row is made up (the questions are the real ones) and results go to out/database/p12_rehearsal.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys

REHEARSAL = os.environ.get("P12_REHEARSAL") == "1"
if REHEARSAL:  # phase 10's modules then read and write their rehearsal folders
    os.environ["P10_REHEARSAL"] = "1"

import p10_data as P10  # noqa: E402
from paths import OUT as RESULTS, ROOT  # noqa: E402

OUT = RESULTS / ("p12_rehearsal" if REHEARSAL else "p12")
EM = P10.EM
BUDGETS = (10, 20, 40) if REHEARSAL else (100, 300, 1000)
REFERENCE_BUDGETS = BUDGETS[1:]  # the released starting point, for comparison
HELD_FOR_TEMPERATURE = 0.2  # "keep 10-20% of the file aside" (Kev's README, Fine-Tune on Your Own Data)
JEV_VALID_SAMPLE = 60 if REHEARSAL else 600  # experiments/entity_match/run.py --split valid --sample 600
JEV_MODEL = "jev-1.13.0"
STARTS = ("loo", "rel")  # phase 10's loo-<dataset> checkpoint (main), the released checkpoint (reference)
EPOCHS = 2  # the recipe's; p12_train.sh passes the same, and kev.train counts epochs x records
MADE_UP = {"train": 240, "valid": 120, "test": 120}  # rehearsal rows per split; the first 60 are phase 10's rehearsal rows, so its cached answers are found


def sha(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def on_test_side(dataset: str, record: str) -> bool:
    return int(sha("p12", dataset, record), 16) % 2 == 1


# ---------- Runs ----------

def arm(dataset: str, start: str, n: int) -> str:
    return f"{dataset}.{start}.{n}"


def parse(name: str) -> tuple[str, str, int]:
    dataset, start, n = name.rsplit(".", 2)
    if dataset not in EM or start not in STARTS or not n.isdigit():
        raise ValueError(f"not a phase 12 run: {name}")
    return dataset, start, int(n)


def trained_arms() -> list[str]:
    """Every run that is trained, in the order it is trained: per dataset the main budgets, then the reference ones."""
    return [arm(d, s, n) for d in EM for s, budgets in (("loo", BUDGETS), ("rel", REFERENCE_BUDGETS)) for n in budgets]


# ---------- Rows ----------

def load_pairs(dataset: str, split: str) -> list[dict]:
    """Rows of one split; "raw" keeps both records as written in the file (made up in rehearsals: the record as sorted JSON)."""
    if REHEARSAL:
        rows = P10.made_pairs(f"{dataset}-{split}", MADE_UP[split], P10.DATASETS[dataset])
        return [r | {"raw": (json.dumps(r["record_a"], sort_keys=True), json.dumps(r["record_b"], sort_keys=True))} for r in rows]
    path = ROOT / "experiments" / "entity_match" / "data" / dataset / f"{split}.txt"
    if not path.exists():
        sys.exit(f"{path} is missing: run ./experiments/database/p10_fetch.sh first")
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        left, right, label = line.split("\t")
        rows.append({"id": f"{split}-{i}", "record_a": P10.parse_record(left), "record_b": P10.parse_record(right), "label": int(label),
                     "raw": (left.strip(), right.strip())})
    return rows


def jev_valid(dataset: str) -> list[dict]:
    """The validation pairs Jev was asked, drawn as experiments/entity_match/run.py drew them: random.Random(0).sample of the split, in that order."""
    pairs = load_pairs(dataset, "valid")
    return random.Random(0).sample(pairs, JEV_VALID_SAMPLE) if JEV_VALID_SAMPLE < len(pairs) else pairs


def both(dataset: str, pair: dict, test_side: bool) -> bool:
    return all(on_test_side(dataset, r) == test_side for r in pair["raw"])


def test_pairs(dataset: str) -> list[dict]:
    """Pairs of the test split (in file order), then of Jev's validation sample (in its order), whose two records are both on the test side."""
    return [p for p in load_pairs(dataset, "test") + jev_valid(dataset) if both(dataset, p, True)]


def label_pool(dataset: str) -> list[dict]:
    pool = [p for p in load_pairs(dataset, "train") if both(dataset, p, False)]
    return sorted(pool, key=lambda p: sha("p12", dataset, p["id"]))


def budget(dataset: str, n: int) -> tuple[list[dict], list[dict]]:
    """(pairs trained on, pairs held out for the temperature) of a budget of n labels."""
    pool = label_pool(dataset)
    if len(pool) < n:
        raise ValueError(f"{dataset}: {len(pool)} pairs in the label pool, fewer than a budget of {n}")
    cut = n - round(HELD_FOR_TEMPERATURE * n)
    return pool[:cut], pool[cut:n]


def request(pair: dict, dataset: str) -> dict:
    return P10.request(pair, dataset)


def labelled(pair: dict, dataset: str) -> dict:
    return P10.labelled(pair, dataset)


# ---------- Commands ----------

def build() -> None:
    for d in EM:
        pairs = test_pairs(d)
        digest = P10.write(OUT / "side" / f"{d}.jsonl", [{"id": p["id"], "label": p["label"]} for p in pairs])
        print(f"{d}: {len(pairs)} test pairs ({sum(p['label'] for p in pairs)} matches); sha256 {digest[:16]}")
        for n in BUDGETS:
            train, held = budget(d, n)
            a = P10.write(OUT / "train" / f"{d}.{n}.jsonl", [labelled(p, d) for p in train])
            b = P10.write(OUT / "slice" / f"{d}.{n}.jsonl", [labelled(p, d) for p in held])
            print(f"   budget {n}: {len(train)} trained on ({sum(p['label'] for p in train)} matches), {len(held)} for the temperature "
                  f"({sum(p['label'] for p in held)} matches); sha256 {a[:16]} / {b[:16]}")


def check() -> int:
    """Counting only. No record of a test pair may occur in any label; budgets must be nested; Jev's validation sample must be the one asked."""
    bad = 0
    for d in EM:
        tests, pool = test_pairs(d), label_pool(d)
        test_records = {r for p in tests for r in p["raw"]}
        label_records = {r for p in pool for r in p["raw"]}
        shared = len(test_records & label_records)
        bad += shared
        for n in BUDGETS:  # each budget is the first n of the same pool, so the budgets are nested
            train, held = budget(d, n)
            bad += bool({p["id"] for p in train} & {p["id"] for p in held})
            bad += [p["id"] for p in train + held] != [p["id"] for p in pool[:n]]
        if not REHEARSAL:
            path = ROOT / "out" / "entity_match" / f"{d}.valid.{JEV_MODEL}.jsonl"
            asked = [json.loads(line)["id"] for line in path.read_text().splitlines()]
            same = asked == [p["id"] for p in jev_valid(d)]
            bad += not same
        else:
            same = True
        print(f"{d}: {len(test_records):,} test-side and {len(label_records):,} label-side records in use; {len(tests)} test pairs "
              f"({sum(p['label'] for p in tests)} matches), label pool {len(pool)} ({sum(p['label'] for p in pool)} matches); "
              f"records on both sides {shared}; Jev's validation sample {'as asked' if same else 'DIFFERS'}")
    print(f"problems: {bad}")
    return bad


if __name__ == "__main__":
    command = {"build": build, "check": lambda: sys.exit(1 if check() else 0)}.get(sys.argv[1] if len(sys.argv) == 2 else "")
    if not command:
        sys.exit(__doc__)
    command()
