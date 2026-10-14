"""Phase 4 collection: 21 semantic predicates on MAMS restaurant reviews (3,949 sentences, each labelled by humans with several aspects and their sentiment).

7 aspects (without the catch-all "miscellaneous") × 3 question forms: is it mentioned, is there a negative opinion, is there a positive opinion.
Ground truth: the aspect is labelled (any sentiment) / (aspect, negative) is labelled / (aspect, positive) is labelled.
The questions are first drafts written from the category names, without reading the labelling guidelines and without editing them against the labels.

source        how                                    rows
jev           one request per sentence and question  all
jev.packed    one request per sentence, all 21 questions together   all (does packing change the judgements?)
luna.low      one request per sentence and question  a fixed sample of 1,000 sentences (is the LLM stronger than Jev on this data?)
luna.low.packed, v41flash.off, v41flash.off.packed      the same 1,000 sentences (added after the pre-check, see question D in experiments/database/protocol/p4.txt)

    python experiments/database/p4_collect.py run jev
    python experiments/database/p4_collect.py merge
"""

from __future__ import annotations

import json
import sys
import time
import xml.etree.ElementTree as ET

from kevdb.cli import load_dotenv

from kevdb import jev_backend, llm_backend
from p2_collect import merge as merge_caches, sha
from paths import DATA, OUT as RESULTS

OUT = RESULTS / "p4"
ASPECTS = {
    "food": "the food or drinks",
    "staff": "the staff (for example waiters, hosts, bartenders or managers)",
    "service": "the service",
    "place": "the place itself (for example its location, space or seating)",
    "menu": "the menu or the range of dishes on offer",
    "price": "prices or value for money",
    "ambience": "the ambience or atmosphere",
}
KINDS = {
    "mention": "Does the restaurant review sentence in `sentence` say something about {}?",
    "negative": "Does the restaurant review sentence in `sentence` express a negative opinion about {}?",
    "positive": "Does the restaurant review sentence in `sentence` express a positive opinion about {}?",
}
PREDICATES = {f"{kind}:{aspect}": template.format(desc) for kind, template in KINDS.items() for aspect, desc in ASPECTS.items()}
SOURCES = {
    "jev": ("jev", 16, None), "jev.packed": ("jev", 16, None), "luna.low": ("luna.low", 48, 1000),
    # Added after the pre-check: can the LLM also answer 21 questions at once, and are an open model's output probabilities still on par with Jev on a second dataset?
    "luna.low.packed": ("luna.low", 16, 1000), "v41flash.off": ("v41flash.off", 16, 1000), "v41flash.off.packed": ("v41flash.off", 16, 1000),
}


def sentences(n: int | None = None) -> list[dict]:
    """Each sentence: id, text and the human label of every predicate. Samples take the first n sentences ordered by text hash."""
    rows = []
    for split in ("train", "val", "test"):
        for i, s in enumerate(ET.parse(DATA / f"mams/{split}.xml").getroot().iter("sentence")):
            marked = {(c.get("category"), c.get("polarity")) for c in s.iter("aspectCategory")}
            truth = {f"mention:{a}": int(any(c == a for c, _ in marked)) for a in ASPECTS}
            truth.update({f"{pol}:{a}": int((a, pol) in marked) for pol in ("negative", "positive") for a in ASPECTS})
            rows.append({"sid": f"{split}-{i}", "text": s.find("text").text, "truth": truth})
    return sorted(rows, key=lambda r: sha(r["text"], "mams"))[:n] if n else rows


def question(name: str) -> dict:
    return {"type": "noul", "instructions": PREDICATES[name]}


def run(source: str) -> None:
    load_dotenv()
    OUT.mkdir(parents=True, exist_ok=True)
    name, concurrency, n = SOURCES[source]
    cache = OUT / f"cache.{source}.sqlite"
    backend = jev_backend(concurrency=concurrency, cache_path=cache) if name == "jev" else llm_backend(name, concurrency=concurrency, cache_path=cache)
    backend.namespace = "p4:"
    rows = sentences(n)
    packed = source.endswith(".packed")
    if packed:
        items = [({"sentence": r["text"]}, {p: question(p) for p in PREDICATES}) for r in rows]
    else:
        items = [({"sentence": r["text"]}, {"q": question(p)}) for r in rows for p in PREDICATES]
    started = time.perf_counter()
    for _ in range(4):
        answers = backend.ask_many(items)
        if not backend.meter.errors:
            break
    seconds = time.perf_counter() - started
    m = backend.meter
    with (OUT / f"mams.{source}.jsonl").open("w") as f:
        for i, r in enumerate(rows):
            token_p = None
            if packed:
                a = answers[i]
                p = None if "_error" in a else {k: a["answers"][k]["noul"] for k in PREDICATES}
            else:
                got = answers[i * len(PREDICATES):(i + 1) * len(PREDICATES)]
                p = {k: (None if "_error" in a else a["answers"]["q"]["noul"]) for k, a in zip(PREDICATES, got)}
                # Open model with thinking off: the probability of "yes" at the answer token
                token_p = {k: (None if "_error" in a else a["answers"]["q"].get("p_true")) for k, a in zip(PREDICATES, got)}
            f.write(json.dumps({"sid": r["sid"], "p": p, "p_true": token_p, "error": answers[i].get("_error") if packed else None}) + "\n")
    summary = {"source": source, "model": backend.model, "sentences": len(rows), "requests": m.calls, "fresh": m.fresh, "errors": m.errors,
               "seconds": round(seconds, 1), "concurrency": concurrency, "input_tokens": m.input_tokens, "cost_usd": round(m.cost, 4), "latency_p50": m.latency(0.5)}
    (OUT / f"mams.{source}.run.json").write_text(json.dumps(summary, ensure_ascii=False) + "\n")
    print(summary, flush=True)
    backend.close()


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"] and sys.argv[2:] and all(s in SOURCES for s in sys.argv[2:]):
        for s in sys.argv[2:]:
            run(s)
    elif sys.argv[1:2] == ["merge"]:
        import p2_collect
        p2_collect.OUT = OUT
        merge_caches()
    else:
        sys.exit(__doc__)
