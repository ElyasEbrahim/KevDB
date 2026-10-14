"""Phase 6 post-hoc check: how different are Kev's answers in half precision (the default used in the experiment) and in full precision (what its author used for the reported numbers)?

In the experiment Kev ran the way its server does by default: MLX backend, bfloat16, half the memory of full precision. Its author states that the reported numbers
come from the PyTorch backend in float32. Here the first few hundred rows of each dataset are asked again in full precision and compared with the answers collected in the experiment.
This is not part of the analysis plan written beforehand (protocol/p6.txt); read it as post hoc.

First start a full-precision server on port 8019 in another terminal (about 16 GB):
    cd open_models/kev && HF_HOME=$PWD/../huggingface HF_HUB_OFFLINE=1 KEV_DTYPE=fp32 \\
        .venv/bin/python -m kev.serve --run jaredpalmer/kev-4b@v1.0 --host 127.0.0.1 --port 8019
    python experiments/database/p6_precision.py kev-4b | tee out/database/p6/precision.kev-4b.txt
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from kevdb.client import PROVIDERS, Provider
from kevdb import Backend
from kevdb.open_models import OpenModel, OpenModelClient
from p4_collect import PREDICATES
from p6_collect import OUT, requests

ROWS = {"movie": 300, "em": 300, "mams": 20}  # for mams these are sentences, 21 requests each


def main(source: str) -> None:
    PROVIDERS["kev-fp32"] = Provider("http://127.0.0.1:8019/v1", None, "kev-latest")
    backend = Backend("kev-fp32", lambda: OpenModelClient(OpenModel("kev-fp32", "kev-latest", f"{source}#fp32-torch")), concurrency=1, cache_path=None)
    for dataset, n in ROWS.items():
        ids, reqs = requests(dataset)
        per_row = len(reqs) // len(ids)
        started = time.perf_counter()
        answers = backend.ask_many(reqs[: n * per_row])
        seconds = time.perf_counter() - started
        full = np.array([a["answers"][next(iter(a["answers"]))]["noul"] for a in answers])
        rows = [json.loads(line) for line in (OUT / f"{dataset}.{source}.jsonl").read_text().splitlines()][:n]
        half = np.array([r["p"][k] for r in rows for k in PREDICATES] if dataset == "mams" else [r["p"] for r in rows])
        gap = np.abs(full - half)
        flipped = int(((full >= 0.5) != (half >= 0.5)).sum())
        print(f"{dataset:6s} {len(full):4d} requests ({seconds / len(full):.2f} s each): {flipped} decisions differ ({flipped / len(full):.2%}), "
              f"mean probability difference {gap.mean():.4f}, max {gap.max():.4f}, {(gap > 0.01).mean():.1%} differ by more than 0.01")
    backend.close()


if __name__ == "__main__":
    main(sys.argv[1])
