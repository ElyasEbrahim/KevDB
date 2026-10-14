"""Phase 6 post-hoc breakdown: where do the three Qwen-based models lose to Jev, on which kind of question, and by missing or by false alarms?

This is not in the analysis plan written beforehand (protocol/p6.txt); it was added after seeing the results to explain them, so read its conclusions as post hoc.
Reads the collected judgements only and calls no model.

    python experiments/database/p6_breakdown.py | tee out/database/p6/breakdown.txt
"""

from __future__ import annotations

import numpy as np

import p6_report as R
from p4_collect import PREDICATES

MODELS = ["kev-4b", "kev-9b", "clef-flash"]
SHOW = ["jev"] + MODELS


def rates(decide: np.ndarray, truth: np.ndarray) -> tuple[float, float, float, float]:
    """Accuracy, share judged yes, miss rate (true rows judged no) and false-alarm rate (false rows judged yes)."""
    return (decide == truth).mean(), decide.mean(), (~decide & truth).sum() / truth.sum(), (decide & ~truth).sum() / (~truth).sum()


def main() -> None:
    names = list(PREDICATES)
    Y, single, _ = R.mams(MODELS)
    _, packed, _ = R.mams(MODELS, packed=True)
    print("Restaurant reviews by question kind (mentions / negative opinion / positive opinion, 7 aspects each)")
    for kind in ("mention", "negative", "positive"):
        cols = [j for j, n in enumerate(names) if n.startswith(kind)]
        truth = Y[:, cols] == 1
        print(f"  {kind}: {truth.mean():.1%} actually true")
        for s in SHOW:
            acc, yes, miss, false = rates(single[s][:, cols] >= 0.5, truth)
            acc_p, yes_p, _, _ = rates(packed[s][:, cols] >= 0.5, truth)
            print(f"    {R.NAMES[s]:26s}one by one: accuracy {acc:.1%}, yes {yes:.1%}, missed {miss:.1%}, false alarms {false:.1%}; packed: accuracy {acc_p:.1%}, yes {yes_p:.1%}")

    for title, loader, positive in (("Product matching", R.em, "matching"), ("Movie reviews", R.movie, "positive")):
        Yd, P, _ = loader(MODELS)
        truth = Yd[:, 0] == 1
        print(f"\n{title}: {len(truth)} rows, {truth.sum()} of them {positive}")
        for s in SHOW:
            p = P[s][:, 0]
            decide = p >= 0.5
            print(f"    {R.NAMES[s]:26s}judged {positive}: {decide.sum()} rows; missed {(~decide & truth).sum()}, false alarms {(decide & ~truth).sum()}; "
                  f"mean probability on rows not {positive} {p[~truth].mean():.3f}, on {positive} rows {p[truth].mean():.3f}")


if __name__ == "__main__":
    main()
