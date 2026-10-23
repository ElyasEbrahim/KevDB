"""Phase 11 evaluation: serve each arm's checkpoint and ask it what experiments/database/protocol/p11.txt asks. A thin layer over phase 10's
p10_eval.py: its serving (sandboxed kev.serve on port 8015, identity checks), caching, request rows, re-ask passes, checkpoint checks and
inventory are reused unchanged, pointed at phase 11's folders and request sets by configure().

    python experiments/database/p11_eval.py arm kev-4b pubs        serve open_models/p11/kev-4b/pubs, ask its sets, stop the server
    python experiments/database/p11_eval.py trained kev-4b pubs    exit 0 if the checkpoint counts
    python experiments/database/p11_eval.py verify                 inputs and built files against the checksums frozen below
    python experiments/database/p11_eval.py check kev-4b           inventory: counts, attempts, deviations; no answer is looked at

Request sets of every arm: test.dblp-acm (the held-out dataset, asked with its own publication question), the temperature pools of the
datasets it trained on (pool.<product> as in phase 10, pool11.<added dataset>), the guards g1-g3 of phase 10, and e2e.dblp-scholar (phase
8's raw table) for words and domains only; pubs trains on DBLP-GoogleScholar, which comes from the same source as that table. The
released checkpoint and the control (phase 10's loo-dblp-acm) are not asked again: their answers to these same rows are in out/database/p10.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

REHEARSAL = os.environ.get("P11_REHEARSAL") == "1"
if REHEARSAL:
    os.environ["P10_REHEARSAL"] = "1"

import p10_eval as E  # noqa: E402
import p11_data as D11  # noqa: E402
from paths import ROOT  # noqa: E402

ARMS = ("pubs", "words", "domains")  # in the order they are trained
RUNS = ROOT / "open_models" / ("p11_rehearsal" if REHEARSAL else "p11")
NAMESPACE = "p11.rehearsal:" if REHEARSAL else "p11:"

INPUTS = {
    "experiments/entity_match/data/beer/train.txt": "88efa35036f8f8c2c60bd10abfb79a2207517faaac7c1c0a12eaca29e868a2bc",
    "experiments/entity_match/data/beer/valid.txt": "1c8974c31b49cc73925cca57226df2152ae5af3f715a9a26bef542a636f26cdf",
    "experiments/entity_match/data/fodors-zagats/train.txt": "5bc0d4095183b325811460407086ced9ec65e634d6b4e713a87b0e823d32b28a",
    "experiments/entity_match/data/fodors-zagats/valid.txt": "c9022749f6c396b75c8cd0e28e3e17710282cfa3b066c6a72172a0c3e3a3453b",
    "experiments/entity_match/data/itunes-amazon/train.txt": "2a6fa5bcbf2b91af2418631db5235ac5a3ba53594509902b43a8b2abc87cea64",
    "experiments/entity_match/data/itunes-amazon/valid.txt": "5c17220ad1f6e721c3ecb5ebb3afa73022865c0618573beeb060a735a796fc42",
    "experiments/entity_match/data/dblp-googlescholar/train.txt": "7f3530e4dad75ee2d2f60271036e8390e18fe449ae38c9350409c9dde93cd838",
    "experiments/entity_match/data/dblp-googlescholar/valid.txt": "167193bb11b635e10d8d792f8aaba4f6c5dee82c8b78164cc58c51cc59b95a65",
    "experiments/entity_match/data/dblp-acm/test.txt": "ded891e469b1d08e21269038c984dc761ae2fd5b142b00c46c30fce45adc6e68",
}
# The three training files and four added pools p11_data.py builds (official run, then the made-up rehearsal ones)
BUILT = {
    "train/domains.jsonl": ("5f5135e19fb4cc146d2653898434391986a9c3fd6c1131a88109ada943158aad", 7156),
    "train/pubs.jsonl": ("5343b91d30636886220a064c44ad7091e17ce8df74b254cae87a41bb8190bb4d", 8000),
    "train/words.jsonl": ("2b74da02e60411f8ed8400e3e5e4527f4690abe0b6ef94b7abb03c0f7b6ac425", 6000),
    "pool/beer.jsonl": ("36c6cff22e57e721260b38c4aa7188f1c639c54f4662db9f58911c440b62f233", 91),
    "pool/dblp-googlescholar.jsonl": ("be118a1b4eafc35dd2b280a58f430dcf47639742761ecfde07fad9490a5b5f8d", 200),
    "pool/fodors-zagats.jsonl": ("9703ad6d585a40cc88636e3ca912a228e8ae8caecaf2fa5597f42503fa11a67f", 190),
    "pool/itunes-amazon.jsonl": ("a50e6c69f6e46ed50e4845430a0df3b4feb7334c6d67034e59558b15df4b6ca3", 109),
}
BUILT_REHEARSAL = {
    "train/domains.jsonl": ("6117477ac27de471fce70528405b581487b99e57d95b59ac4637a8f8e77ad2fc", 108),
    "train/pubs.jsonl": ("7c898c7dd5b6bd5f3c1a177d5600e39c3cf6f96f9342d4b14723b9c3bec64298", 96),
    "train/words.jsonl": ("9daa139855ffb8a7f61ec963348835337ea809ce80ce97ee4f3444032bba654f", 72),
    "pool/beer.jsonl": ("e247896c7fd35c4dc8f53069214d1a515e3f0f3c388c63a3324569cb404eca97", 12),
    "pool/dblp-googlescholar.jsonl": ("a8d8e10e8afec3052fd3ecd9eb1e65d8a3c678294b7fbaf2ebf5c09ab8a11817", 12),
    "pool/fodors-zagats.jsonl": ("b12fa30006e2e0ad018881aab4c1866cc73c2ec823134528965d3dbe05eb8e52", 12),
    "pool/itunes-amazon.jsonl": ("dccbac91c669ebdbdb4217b3242b430a24a6fab9a6a9a03617f9ee34069ffc0f", 12),
}


def sets(arm: str) -> list[str]:
    added = D11.ARMS[arm][len(D11.PRODUCTS):]
    names = ["test.dblp-acm"] + [f"pool.{d}" for d in D11.PRODUCTS] + [f"pool11.{d}" for d in added] + list(E.GUARDS)
    return names + ([] if arm == "pubs" else ["e2e.dblp-scholar"])


_p10_rows = E.rows
_p10_cache_id = E.cache_id


def rows(name: str) -> list[dict]:
    if name.startswith("pool11."):
        if name not in E._ROWS:
            dataset = name.split(".", 1)[1]
            E._ROWS[name] = [{"id": p["id"], "label": p["label"], **D11.request(p, D11.question(dataset))} for p in D11.pool_pairs(dataset)]
        return E._ROWS[name]
    return _p10_rows(name)


def cache_id(model: str, arm: str) -> str:
    return _p10_cache_id(model, arm).replace("p10/", "p11/", 1)


def configure() -> None:
    """Point phase 10's evaluation code at phase 11. Called by this script's and p11_report.py's entry points only, never on import, so that
    phase 10's code and tests are unaffected wherever both are loaded."""
    E.RUNS = RUNS
    E.D.OUT = D11.OUT
    E.NAMESPACE = NAMESPACE
    E.ARMS = ARMS
    E.TRAINED = ARMS
    E.sets = sets
    E.rows = rows
    E.cache_id = cache_id


def sha256(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), data.count(b"\n")


def verify() -> int:
    bad = 0
    inputs = INPUTS | {k: v for k, v in E.INPUTS.items() if "development" in k or "decision-v7/train" in k or "/train.txt" in k}
    for rel, digest in inputs.items():
        got = sha256(ROOT / rel)[0] if (ROOT / rel).exists() else "missing"
        if got != digest:
            print(f"input changed: {rel} ({got[:16]})")
            bad += 1
    frozen = BUILT_REHEARSAL if REHEARSAL else BUILT
    for rel, (digest, lines) in frozen.items():
        path = D11.OUT / rel
        got = sha256(path) if path.exists() else ("missing", 0)
        if got != (digest, lines):
            print(f"built file differs from the frozen one: {rel} ({got[0][:16]}, {got[1]} lines; frozen {digest[:16]}, {lines} lines)")
            bad += 1
    if not frozen:
        print("no frozen checksums for the built files yet")
        bad += 1
    print(f"verify: {len(inputs) + len(frozen) - bad} of {len(inputs) + len(frozen)} files as frozen")
    return bad


def main(argv: list[str]) -> int:
    configure()
    command, rest = (argv[:1] or [""])[0], argv[1:]
    if command == "arm" and len(rest) == 2 and rest[0] in E.MODELS and rest[1] in ARMS:
        return E.evaluate(*rest)
    if command == "trained" and len(rest) == 2 and rest[0] in E.MODELS and rest[1] in ARMS:
        reason = E.not_trained(*rest)
        print(reason or "complete")
        return 1 if reason else 0
    if command == "verify" and not rest:
        return 1 if verify() else 0
    if command == "check" and len(rest) == 1 and rest[0] in E.MODELS:
        return 1 if E.check(rest[0]) else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    sys.exit(main(sys.argv[1:]))
