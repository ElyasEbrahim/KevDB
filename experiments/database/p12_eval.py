"""Phase 12 evaluation: serve each run's checkpoint and ask it what experiments/database/protocol/p12.txt asks. A thin layer over phase 10's
p10_eval.py, as in phase 11: serving (sandboxed kev.serve on port 8015, identity checks), caching, re-ask passes, checkpoint checks and
inventory are phase 10's, pointed at phase 12's folders, runs and request sets by configure().

    python experiments/database/p12_eval.py arm kev-4b abt-buy.loo.300      serve open_models/p12/kev-4b/abt-buy.loo.300, ask its sets, stop the server
    python experiments/database/p12_eval.py arm kev-4b abt-buy.loo.0        n = 0: phase 10's loo-abt-buy checkpoint, its answers in phase 10's cache reused
    python experiments/database/p12_eval.py arm kev-4b released             the released checkpoint on every dataset's test pairs, cached answers reused
    python experiments/database/p12_eval.py trained kev-4b abt-buy.loo.300  exit 0 if the checkpoint counts
    python experiments/database/p12_eval.py verify                          inputs and built files against the checksums frozen below
    python experiments/database/p12_eval.py check kev-4b                    inventory: counts, attempts, deviations; no answer is looked at
    python experiments/database/p12_eval.py arms                            every run in the order p12_run.sh takes them

Request sets: side.<D> (the dataset's test pairs, asked with phase 10's first-draft question), slice.<D>.<n> (the 20% of the run's budget
held out for its temperature) and the guards g1-g3 of phase 10 for every trained run; side.<D> only for <D>.loo.0, whose temperature and
guard answers are phase 10's; side.<D> of all four datasets for the released checkpoint, whose guard answers are phase 10's too.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REHEARSAL = os.environ.get("P12_REHEARSAL") == "1"
if REHEARSAL:
    os.environ["P10_REHEARSAL"] = "1"

import p10_eval as E  # noqa: E402
import p12_data as D12  # noqa: E402
from paths import OUT as RESULTS, ROOT  # noqa: E402

MODEL = "kev-4b"
RUNS = ROOT / "open_models" / ("p12_rehearsal" if REHEARSAL else "p12")
P10_RUNS = ROOT / "open_models" / "p10"  # the starting checkpoints, in rehearsals too (phase 10's rehearsal checkpoints are deleted)
P10_OUT = RESULTS / ("p10_rehearsal" if REHEARSAL else "p10")
P10_NAMESPACE = E.NAMESPACE  # phase 10's, read before configure() replaces it
NAMESPACE = "p12.rehearsal:" if REHEARSAL else "p12:"
TRAINED = tuple(D12.trained_arms())
ARMS = ("released",) + tuple(a for d in D12.EM for a in [D12.arm(d, "loo", 0)] + [x for x in TRAINED if D12.parse(x)[0] == d])
# The weights phase 10 evaluated as loo-<dataset> (digest in its cache name, out/database/p10/kev-4b/loo-<dataset>/eval.json): n = 0 counts only on these
P10_DIGESTS = {"abt-buy": "b0a501e195df290e", "amazon-google": "526074011cd9d283", "walmart-amazon": "51d9bf5cd5644933", "dblp-acm": "51066e4189ffa85c"}

# Frozen before the run (see the block appended to p12.txt): sha256 of every input, (sha256, lines) of every built file
INPUTS: dict[str, str] = {
    "experiments/entity_match/data/abt-buy/train.txt": "c1370966c08c246e9cf844b7cd2cf580b2559e9831f1eb3cca560fd53c1bcbb2",
    "experiments/entity_match/data/abt-buy/valid.txt": "14b2c80673434b852febacd1a741c89e2ca0aba1658f4420efe4cbb46f4ef607",
    "experiments/entity_match/data/abt-buy/test.txt": "df323df317af4ca33132e3c3a666983c5cbc69ef517f61df18f381b4d1662352",
    "experiments/entity_match/data/amazon-google/train.txt": "84ef5f7747296990a3775d2c8774949e720df643106b1b447682265f41929b26",
    "experiments/entity_match/data/amazon-google/valid.txt": "baa53c43706e979441aaf653698bb477294b6fa16328fd272fcdc56ca040c094",
    "experiments/entity_match/data/amazon-google/test.txt": "fb4a50692f296435d9ecd4b310acc1318ff7d6d1e13924f0b211d1f906dacbef",
    "experiments/entity_match/data/walmart-amazon/train.txt": "092fccd11c1cc1ffe54a702af276641218ce9506e6d60fb5a29a3c67d87892c3",
    "experiments/entity_match/data/walmart-amazon/valid.txt": "4b07b12d5eb411736da03385d7543a34c77e5d9d364c25d67e987ebabc58c92c",
    "experiments/entity_match/data/walmart-amazon/test.txt": "85ee132c4e45073a4b5ab8168d5db8af3e354b7b46dae19fd111757c0a6a97f6",
    "experiments/entity_match/data/dblp-acm/train.txt": "4f757d2d5dae4f7d7d118b796c683a232b4e11bf000e2012b64f8af95a981a4e",
    "experiments/entity_match/data/dblp-acm/valid.txt": "b13444a2bf1a503ed558e782f42b247cf56e16e67be2e58d92d7435161299e2e",
    "experiments/entity_match/data/dblp-acm/test.txt": "ded891e469b1d08e21269038c984dc761ae2fd5b142b00c46c30fce45adc6e68",
    "out/entity_match/abt-buy.test.jev-1.13.0.jsonl": "28cf9607085de21f225a5bbae4d6cf1efa0fb82d1235e475c10d76a981ef001e",
    "out/entity_match/abt-buy.valid.jev-1.13.0.jsonl": "98d94b2e287f7500cc90ee0b917d20077cd3de931c96c25470398e00f60e83ac",
    "out/entity_match/amazon-google.test.jev-1.13.0.jsonl": "88c3fedd4978ad51d1471dd5bf36d4dc37179329f3b4d5bcc02681b697f20064",
    "out/entity_match/amazon-google.valid.jev-1.13.0.jsonl": "aba2859d413410458ebb26e7d4fa4a683e23415e11ffc208aedb0853afc5ef01",
    "out/entity_match/walmart-amazon.test.jev-1.13.0.jsonl": "3630643d9d3f828c7709acf766c88355187a6536d3850fafb83475322af4380d",
    "out/entity_match/walmart-amazon.valid.jev-1.13.0.jsonl": "d36bafc46a58fdf57b2639b1bfeaa977b086c83efe530b2990b6e02aa5a2d2e5",
    "out/entity_match/dblp-acm.test.jev-1.13.0.jsonl": "43cfd89209ebdf820357c62317a12f05b0b70bec52ac86da215ce9dc0af6a44e",
    "out/entity_match/dblp-acm.valid.jev-1.13.0.jsonl": "fea1e703043ec21f7fe69cc5b27bf51813edeafad175a4b0af7b7fa5918b223f",
}
BUILT: dict[str, tuple[str, int]] = {
    "side/abt-buy.jsonl": ("e8f77cd2034989e6036eaf96c5609f34dfe8a91ab1ed625a84037fcf499ddfdc", 632),
    "side/amazon-google.jsonl": ("59900ac499cb4e5522b9eba96ba87bd95f7109e79d81b219b8e5af65416bab42", 664),
    "side/walmart-amazon.jsonl": ("76887e3107e90043a0bb14726a9fd44fe7699b87cf38d679b345b7fe9727abda", 669),
    "side/dblp-acm.jsonl": ("4577a56dc25103b17665ed408fd6ff652851964597eb23d64a6be03d4a30a47c", 766),
    "train/abt-buy.100.jsonl": ("ff4723f71a3245cdefa41a96c20122315d5505cc62f61a2316cdc56bd53ba410", 80),
    "slice/abt-buy.100.jsonl": ("2bd5cebf65090f6f5155632bedc643f594070eb53742d77dd5a614eaf133993b", 20),
    "train/abt-buy.300.jsonl": ("04df99a79e0fa244d23269c9fd1cd362fc48030e335a6b3647650d5a3dc3593c", 240),
    "slice/abt-buy.300.jsonl": ("f33bec41138015e7adae62feeda857e1a92ac1fe8790a7967c24d0b28669d3fa", 60),
    "train/abt-buy.1000.jsonl": ("3b927c8c7d1de151e219fa8396264d922c3d3858bf82a8f5bef8acc1649a38fe", 800),
    "slice/abt-buy.1000.jsonl": ("075908b970a7288916b7fbbbb6e098fdf4a264c65b662c69362835668fc69ae8", 200),
    "train/amazon-google.100.jsonl": ("4a5c96d076c92e82c737608dc054bc166be69762ed49a134be51e1924ceed909", 80),
    "slice/amazon-google.100.jsonl": ("74d4b8261e8960b24e74bdb4219d7001b74ef168931e3975653f92c6eed83763", 20),
    "train/amazon-google.300.jsonl": ("e6bb5b2ca76fe9d4219e3455c620025ef3c178d8cc97307ae9697032405af723", 240),
    "slice/amazon-google.300.jsonl": ("126212e79975aa73c39be0bc995ba5c7fbdc4ed1875e2422262642272a29f52a", 60),
    "train/amazon-google.1000.jsonl": ("f361c730e08485bcc4fb02b19fa41eb3fec417b05f1bc762c582bd624691b3ee", 800),
    "slice/amazon-google.1000.jsonl": ("534c0118232b1ee525efbbf34c3b2828559e934813f565d2c62f866840f343b8", 200),
    "train/walmart-amazon.100.jsonl": ("c1d771b8c97fd3e8f69f52666045c20e63b85d040e1a4aee5d8fb45f9ebaad91", 80),
    "slice/walmart-amazon.100.jsonl": ("daf12d1477e7923782dbd9339c24d94c35918fd85fea0c4156c7971f2ebd6c3c", 20),
    "train/walmart-amazon.300.jsonl": ("856aa09f97f29c310903d174ae5e95927cd6ffcee769f4950f5bfd9eb9c50b56", 240),
    "slice/walmart-amazon.300.jsonl": ("2b64a18da6fe41b9e79bc1005d85138e646702b0a452a033d08b26ef45f66a8a", 60),
    "train/walmart-amazon.1000.jsonl": ("00bbae80a679ab430ab85db10b17260803baf8111035ffbd9e15dede45e33f21", 800),
    "slice/walmart-amazon.1000.jsonl": ("62d21da5b4b99b060a15e74e998bb47b5cfceff0509d0faf585264a104d4c919", 200),
    "train/dblp-acm.100.jsonl": ("44f4999677fd03b34fe69566d88c546c0330d0fc54f230da4393f81c4398cc6c", 80),
    "slice/dblp-acm.100.jsonl": ("f03100483dcaa66017ca6328345f651ba7b1cecc4dd3bc399dc62630fb78a9f2", 20),
    "train/dblp-acm.300.jsonl": ("02be67b42d651e82a27227d2a22590e7e0395693e2254ae48665f88e2e6615ae", 240),
    "slice/dblp-acm.300.jsonl": ("4dc58dd640a9b8360291495656422a1d56ab2102afade76169ca465cc3e196a8", 60),
    "train/dblp-acm.1000.jsonl": ("c2d956f870493be79e0c84f8905cb0953f7963863bff67524eacff579633f85d", 800),
    "slice/dblp-acm.1000.jsonl": ("31763451fc92b8b0702a8715ac40add76594754674f5c0ea04a32d391086a42d", 200),
}
BUILT_REHEARSAL: dict[str, tuple[str, int]] = {
    "side/abt-buy.jsonl": ("a87823b628f2f3ceffa7ad891781297a010e7f6a1552e4acc8c043dab61ed537", 43),
    "side/amazon-google.jsonl": ("0e56ebb7a1a3524d7281d5a0d3f98ec9d39e564bd3e39e2cdc57aaefea36ff89", 36),
    "side/walmart-amazon.jsonl": ("d58900d5d1613c47f1c4cd541edfb527eef3fe9819d3df3ed88f91e1ed24ca89", 39),
    "side/dblp-acm.jsonl": ("474468899dc3c6f34268fd04c3897b318d063356aed17e0b8c7a9102aeebf395", 47),
    "train/abt-buy.10.jsonl": ("2b8bca885db6c33f8c90e2e82fa9f6122eaf0e1d02c03fb1ec5ffc76e41832c3", 8),
    "slice/abt-buy.10.jsonl": ("ba761d1c10043edcfff2076b68101e2578cecc0aa32f621c5bf69901a7be2b97", 2),
    "train/abt-buy.20.jsonl": ("f6166acbcf6eaed67f9059cf5d17ca302889f75480c0f76811726cdd078ee86b", 16),
    "slice/abt-buy.20.jsonl": ("3da21c79b565bd43fea5d305b6c9a989cb487d18bd3618cf56c982ed0661d393", 4),
    "train/abt-buy.40.jsonl": ("6a7b00e403c1c6d7e7b9f215742f9c83c4dace9db0b27cce73e27d478dca604c", 32),
    "slice/abt-buy.40.jsonl": ("baa92862c57a4904f6b9e76b11a491f5d4e60e70d3578c53124289a000c95a5a", 8),
    "train/amazon-google.10.jsonl": ("fe04cee44c83ce5ba5e58bcf20ae7e2978b528436885884045eb519698769bbf", 8),
    "slice/amazon-google.10.jsonl": ("77001d3540771e2b047199f75019a44281ecfc1863f4320aa714718a3b39a035", 2),
    "train/amazon-google.20.jsonl": ("29085e952a94f50c5b7b5142326d5da5fdb997ac1ebda0dc8570e4e9cbc1cc0f", 16),
    "slice/amazon-google.20.jsonl": ("25775247047bd3e32ebbf3db6cb166bc78aa551b0b6529804316526613ad2f99", 4),
    "train/amazon-google.40.jsonl": ("4102426845242d389e5699ca1f87f914b66326997f0bc0b7fb79ff63dd1b6cf4", 32),
    "slice/amazon-google.40.jsonl": ("8140e45e770953864bc7eb97cd1837eab996fd67ca278bf9396d80e20cf73836", 8),
    "train/walmart-amazon.10.jsonl": ("accd2a356717cf5fa76acf93ae541280ff26bedad706c9d7367bb68032c736f3", 8),
    "slice/walmart-amazon.10.jsonl": ("9485222aa08c4efd70f6706c9d6668f2b411928d343bebb38a748b77e2f319dd", 2),
    "train/walmart-amazon.20.jsonl": ("7e3b9f3dcc65b56cc0ca98d1820b757fd5b654e567b2a68b7bb06324c76b78b4", 16),
    "slice/walmart-amazon.20.jsonl": ("f9296af757a6a535d63f124fdc36838e6c0fb34abef74863c4ea01ddfb196060", 4),
    "train/walmart-amazon.40.jsonl": ("b99d9b12b9f8e50f50ba5512485b5cc0630b739a8068e30ba18e8aba91e5ff30", 32),
    "slice/walmart-amazon.40.jsonl": ("d35b8ff4d600891c844fc41d04a51e150f41c7ec7f49f89471a08c2c55459f37", 8),
    "train/dblp-acm.10.jsonl": ("96468a61b284db2a13aedb21c4fddad8af4fb83284b0bd58bc4441ae4f937e2f", 8),
    "slice/dblp-acm.10.jsonl": ("f8c9a39d816028b5981d66d08d7426b235f5e05ed9dc950fd82c5ea8d6bf35db", 2),
    "train/dblp-acm.20.jsonl": ("4fccc92a74c2d059c47851634a215078acd984578b170622aed16b1711cd09f9", 16),
    "slice/dblp-acm.20.jsonl": ("01fb856fefe4c0a9ef2a0dc77450f256701635ae7d0218478b71df66b4c6464e", 4),
    "train/dblp-acm.40.jsonl": ("43ceb48bfa05ae1a6894c10a778c9868060c58f0dcb20a22d3169993388325e0", 32),
    "slice/dblp-acm.40.jsonl": ("7a0e2855ef84d3165ee0ec63a8079e88431ab95e5e1b2017774655f9cf2f3b0f", 8),
}


def start_of(arm: str) -> bool:
    """True for <dataset>.loo.0: phase 10's checkpoint itself, n = 0."""
    return arm != "released" and D12.parse(arm)[2] == 0


def checkpoint(model: str, arm: str) -> Path:
    if start_of(arm):
        return P10_RUNS / model / f"loo-{D12.parse(arm)[0]}"
    return RUNS / model / arm


def weights_digest(path: Path) -> str:
    """As phase 10's cache_id: sha256 over head.pt then adapter_model.safetensors, first 16 hex digits."""
    h = hashlib.sha256()
    for name in ("head.pt", "adapter_model.safetensors"):
        with (path / name).open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()[:16]


def cache_id(model: str, arm: str) -> str:
    """The released checkpoint's own name; for n = 0 phase 10's name of the same weights, so its cached answers are found; else the run plus a digest."""
    if arm == "released":
        return f"jaredpalmer/{model}@{E.WEIGHTS_TAG}"
    dataset, _, n = D12.parse(arm)
    digest = weights_digest(checkpoint(model, arm))
    return f"p10/{model}/loo-{dataset}@{digest}" if n == 0 else f"p12/{model}/{arm}@{digest}"


def sets(arm: str) -> list[str]:
    if arm == "released":
        return [f"side.{d}" for d in D12.EM]
    dataset, _, n = D12.parse(arm)
    return [f"side.{dataset}"] if n == 0 else [f"side.{dataset}", f"slice.{dataset}.{n}", *E.GUARDS]


_p10_rows = E.rows
_p10_not_trained = E.not_trained
_p10_ask_set = E.ask_set


def rows(name: str) -> list[dict]:
    kind, _, arg = name.partition(".")
    if kind not in ("side", "slice"):
        return _p10_rows(name)  # g1-g3
    if name not in E._ROWS:
        if kind == "side":
            dataset, pairs = arg, D12.test_pairs(arg)
        else:
            dataset, n = arg.rsplit(".", 1)
            pairs = D12.budget(dataset, int(n))[1]
        E._ROWS[name] = [{"id": p["id"], "label": p["label"], **D12.request(p, dataset)} for p in pairs]
    return E._ROWS[name]


def training_records(arm: str) -> int:
    dataset, _, n = D12.parse(arm)
    with (D12.OUT / "train" / f"{dataset}.{n}.jsonl").open() as f:
        return D12.EPOCHS * sum(1 for _ in f)


def not_trained(model: str, arm: str) -> str:
    """'' if the run's checkpoint counts. n = 0: phase 10's checkpoint, unchanged; trained runs: phase 10's rules (files, end and saved lines, no
    dropped, truncated or rejected record, every requested record trained on: epochs x the training file)."""
    if arm == "released":
        return ""
    if start_of(arm):
        path = checkpoint(model, arm)
        if not all((path / n).exists() for n in ("head.pt", "adapter_model.safetensors")):
            return f"phase 10's checkpoint {path} is missing"
        got, want = weights_digest(path), P10_DIGESTS[D12.parse(arm)[0]]
        return "" if got == want else f"{path} holds other weights ({got}) than phase 10 evaluated ({want})"
    return _p10_not_trained(model, arm)


def replay_from_phase_10(model: str, arm: str) -> list[tuple[str, Path, tuple[str, ...]]]:
    """Phase 10's cache of the same weights: the released checkpoint's, or loo-<dataset>'s for n = 0. Opened read-only."""
    if arm == "released":
        path = P10_OUT / model / "released" / "cache.sqlite"
    elif start_of(arm):
        path = P10_OUT / model / f"loo-{D12.parse(arm)[0]}" / "cache.sqlite"
    else:
        return []
    return [("p10", path, (P10_NAMESPACE,))] if path.exists() else []


def ask_set(backend, model: str, arm: str, name: str, out: Path, gone) -> dict:
    """Phase 10's ask_set, after copying in what phase 10 already asked the same weights; the summary counts those answers as replayed."""
    copied: dict[str, int] = {}
    sources = replay_from_phase_10(model, arm)
    if sources:
        copied = E.replay(model, backend.model, [(r["state"], r["questions"]) for r in rows(name)], out / "cache.sqlite", sources)
    summary = _p10_ask_set(backend, model, arm, name, out, gone)
    if copied:
        summary["replayed"] = {k: summary["replayed"].get(k, 0) + copied.get(k, 0) for k in {*summary["replayed"], *copied}}
        (out / f"{name}.run.json").write_text(json.dumps(summary, ensure_ascii=False) + "\n")
        print(f"      of which {sum(copied.values())} answers came from phase 10's cache of the same weights", flush=True)
    return summary


def configure() -> None:
    """Point phase 10's evaluation code at phase 12. Called by this script's and p12_report.py's entry points only, never on import, so that
    phase 10's and phase 11's code and tests are unaffected wherever they are loaded together."""
    E.RUNS = RUNS
    E.D.OUT = D12.OUT
    E.NAMESPACE = NAMESPACE
    E.ARMS = ARMS
    E.TRAINED = TRAINED
    E.sets = sets
    E.rows = rows
    E.cache_id = cache_id
    E.checkpoint = checkpoint
    E.not_trained = not_trained
    E.training_records = training_records
    E.ask_set = ask_set


def sha256(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), data.count(b"\n")


def built_files() -> list[str]:
    return [f"side/{d}.jsonl" for d in D12.EM] + [f"{kind}/{d}.{n}.jsonl" for d in D12.EM for n in D12.BUDGETS for kind in ("train", "slice")]


def verify() -> int:
    bad = 0
    inputs = INPUTS | {k: v for k, v in E.INPUTS.items() if "development" in k}
    for rel, digest in inputs.items():
        got = sha256(ROOT / rel)[0] if (ROOT / rel).exists() else "missing"
        if got != digest:
            print(f"input changed: {rel} ({got[:16]})")
            bad += 1
    frozen = BUILT_REHEARSAL if REHEARSAL else BUILT
    for rel in built_files():
        path = D12.OUT / rel
        got = sha256(path) if path.exists() else ("missing", 0)
        if rel not in frozen or got != frozen[rel]:
            print(f"built file differs from the frozen one: {rel} ({got[0][:16]}, {got[1]} lines; frozen {frozen.get(rel, ('none', 0))[0][:16]})")
            bad += 1
    if not INPUTS or not frozen:
        print("no frozen checksums yet")
        bad += 1
    for d, want in P10_DIGESTS.items():
        got = weights_digest(P10_RUNS / MODEL / f"loo-{d}") if (P10_RUNS / MODEL / f"loo-{d}" / "head.pt").exists() else "missing"
        if got != want:
            print(f"starting checkpoint loo-{d} differs from the one phase 10 evaluated ({got})")
            bad += 1
    total = len(inputs) + len(built_files()) + len(P10_DIGESTS)
    print(f"verify: {total - bad} of {total} files and checkpoints as frozen")
    return bad


def main(argv: list[str]) -> int:
    configure()
    command, rest = (argv[:1] or [""])[0], argv[1:]
    if command == "arm" and rest[:1] == [MODEL] and len(rest) == 2 and rest[1] in ARMS:
        return E.evaluate(*rest)
    if command == "trained" and rest[:1] == [MODEL] and len(rest) == 2 and rest[1] in TRAINED:
        reason = not_trained(*rest)
        print(reason or "complete")
        return 1 if reason else 0
    if command == "verify" and not rest:
        return 1 if verify() else 0
    if command == "arms" and not rest:
        print(" ".join(ARMS))
        return 0
    if command == "check" and rest == [MODEL]:
        return 1 if E.check(MODEL) else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    sys.exit(main(sys.argv[1:]))
