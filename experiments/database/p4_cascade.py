"""Phase 4, a replay added afterwards: Jev answers packed first and passes its least certain rows to the packed LLM; how much of the gap does that recover?

Design and reading rule are in the last section of experiments/database/protocol/p4.txt (written after seeing question D's result, before computing this). Calls no model.

    python experiments/database/p4_cascade.py
"""

from __future__ import annotations

import json

import numpy as np

from p4_collect import PREDICATES, sentences
from p4_mams import f1
from paths import OUT as RESULTS

OUT = RESULTS / "p4"
SHARES, DRAWS, SEED = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0), 200, 20261007


def main() -> None:
    rows = sentences(1000)
    names = list(PREDICATES)
    read = lambda source: {r["sid"]: r["p"] for r in map(json.loads, (OUT / f"mams.{source}.jsonl").read_text().splitlines())}
    jev, llm = read("jev.packed"), read("luna.low.packed")
    Y = np.array([[r["truth"][k] for k in names] for r in rows]) == 1
    P = np.array([[jev[r["sid"]][k] for k in names] for r in rows])
    J, L = P >= 0.5, np.array([[llm[r["sid"]][k] for k in names] for r in rows]) >= 0.5
    cost = {s: 1000 * json.loads((OUT / f"mams.{s}.run.json").read_text())["cost_usd"] / json.loads((OUT / f"mams.{s}.run.json").read_text())["sentences"]
            for s in ("jev.packed", "luna.low.packed")}
    order = np.argsort(-(P * (1 - P)).sum(axis=1))
    rng = np.random.default_rng(SEED)

    def score(escalated: np.ndarray) -> tuple[float, float]:
        pred = np.where(escalated[:, None], L, J)
        return (pred == Y).mean(), float(np.mean([f1(pred[:, j], Y[:, j].astype(int)) for j in range(len(names))]))

    n = len(rows)
    base, full = score(np.zeros(n, bool)), score(np.ones(n, bool))
    print(f"{n} sentences × {len(names)} predicates. Jev only: accuracy {base[0]:.2%}, F1 {base[1]:.3f}; everything to the LLM: accuracy {full[0]:.2%}, F1 {full[1]:.3f}")
    print(f"{'escalated':>9s}{'least certain first':>20s}{'random':>16s}{'gap recovered':>14s}{'per 1k sent.':>13s}")
    out = {"jev": base, "llm": full, "curve": {}}
    for b in SHARES:
        k = int(round(b * n))
        chosen = np.zeros(n, bool)
        chosen[order[:k]] = True
        acc, f = score(chosen)
        randoms = []
        for _ in range(DRAWS):
            r = np.zeros(n, bool)
            r[rng.choice(n, k, replace=False)] = True
            randoms.append(score(r))
        racc, rf = np.mean([x[0] for x in randoms]), np.mean([x[1] for x in randoms])
        recovered = (acc - base[0]) / (full[0] - base[0])
        usd = cost["jev.packed"] + b * cost["luna.low.packed"]
        out["curve"][b] = {"accuracy": acc, "f1": f, "random_accuracy": racc, "random_f1": rf, "recovered": recovered, "cost_per_1000": usd}
        print(f"{b:>10.0%}{acc:>12.2%} / {f:.3f}{racc:>12.2%} / {rf:.3f}{recovered:>11.0%}{usd:>12.3f}")
    r30 = out["curve"][0.3]["recovered"]
    print(f"escalating 30% of the rows recovers {r30:.0%} of the gap (target ≥60%) → {'worth it' if r30 >= 0.6 else 'below target'}")
    out["passed"] = bool(r30 >= 0.6)
    (OUT / "cascade.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
