"""Phase 4, question D: is "many questions in one request" specific to Jev, or does it work for LLMs too? Plus a second-dataset check of open models' output probabilities.

Reads the judgements collected by p4_collect.py only and calls no model. The reading rule was written before collection, see experiments/database/protocol/p4.txt:
packing "works equally well" for an LLM only if packed accuracy is within 1 point of one-by-one and a packed request takes less than 3x the latency of a single-question request.

    python experiments/database/p4_packing.py
"""

from __future__ import annotations

import json

import numpy as np

from p4_collect import PREDICATES, sentences
from p4_mams import auc, f1
from paths import OUT as RESULTS

OUT = RESULTS / "p4"


def main() -> None:
    rows = {r["sid"]: r for r in sentences(1000)}
    names = list(PREDICATES)
    sid = list(rows)
    Y = np.array([[rows[s]["truth"][k] for k in names] for s in sid])
    read = lambda source, field="p": {r["sid"]: r for r in map(json.loads, (OUT / f"mams.{source}.jsonl").read_text().splitlines())}
    run = lambda source: json.loads((OUT / f"mams.{source}.run.json").read_text())
    out: dict = {}
    print(f"the same {len(sid)} sentences × {len(names)} predicates: one question per request vs 21 questions packed")
    print(f"{'model':22s}{'acc single':>11s}{'acc packed':>11s}{'diff':>7s}{'agree':>8s}{'incompl.':>9s}{'latency/request':>17s}{'cost per 1k sent.':>18s}{'requests per 1k':>16s}")
    for label, source in (("Jev", "jev"), ("GPT 6 Luna low reasoning", "luna.low"), ("DeepSeek V4.1 Flash", "v41flash.off")):
        single, packed = read(source), read(f"{source}.packed")
        missing = sum(packed[s]["p"] is None for s in sid)
        ok = [s for s in sid if packed[s]["p"] is not None]
        yy = np.array([[rows[s]["truth"][k] for k in names] for s in ok])
        ps = np.array([[single[s]["p"][k] for k in names] for s in ok]) >= 0.5
        pk = np.array([[packed[s]["p"][k] for k in names] for s in ok]) >= 0.5
        a1, a2 = (ps == (yy == 1)).mean(), (pk == (yy == 1)).mean()
        r1, r2 = run(source), run(f"{source}.packed")
        # Jev's one-by-one and packed runs covered all 3,949 sentences; cost is scaled per 1,000 sentences
        c1, c2 = 1000 * r1["cost_usd"] / r1["sentences"], 1000 * r2["cost_usd"] / r2["sentences"]
        slow = r2["latency_p50"] / r1["latency_p50"]
        usable = abs(a2 - a1) < 0.01 and slow < 3
        out[label] = {"acc_single": a1, "acc_packed": a2, "agreement": (ps == pk).mean(), "missing": missing, "latency_single": r1["latency_p50"],
                      "latency_packed": r2["latency_p50"], "cost_single_per_1000": c1, "cost_packed_per_1000": c2, "packing_usable": bool(usable),
                      "f1_single": float(np.mean([f1(ps[:, j], yy[:, j]) for j in range(len(names))])), "f1_packed": float(np.mean([f1(pk[:, j], yy[:, j]) for j in range(len(names))]))}
        print(f"{label:22s}{a1:>11.2%}{a2:>12.2%}{100 * (a2 - a1):>+8.2f}{(ps == pk).mean():>10.2%}{missing:>7d}"
              f"{r1['latency_p50']:>8.2f} → {r2['latency_p50']:.2f}s ({slow:.1f}x)  ${c1:.3f} → ${c2:.3f} ({c1 / c2:.1f}x cheaper){21000:>8d} → 1000   {'usable' if usable else 'not met'}")
    jev_p, llm_p = out["Jev"], out["GPT 6 Luna low reasoning"]
    print(f"  packed, per 1,000 sentences: Jev ${jev_p['cost_packed_per_1000']:.3f}, {jev_p['latency_packed']:.2f} s; Luna ${llm_p['cost_packed_per_1000']:.3f}, {llm_p['latency_packed']:.2f} s; "
          f"DeepSeek ${out['DeepSeek V4.1 Flash']['cost_packed_per_1000']:.3f}, {out['DeepSeek V4.1 Flash']['latency_packed']:.2f} s")

    # Attribution check on the second dataset: Jev's probabilities vs an open model's output probabilities with thinking off
    jev, open_ = read("jev"), read("v41flash.off")
    PJ = np.array([[jev[s]["p"][k] for k in names] for s in sid])
    PO = np.array([[open_[s]["p_true"][k] for k in names] for s in sid], dtype=float)
    HO = np.array([[open_[s]["p"][k] for k in names] for s in sid])
    print(f"\nJev's probabilities vs DeepSeek V4.1 Flash's output probabilities ({len(sid)} sentences, {len(names)} predicates, {np.isnan(PO).mean():.2%} missing)")
    PO = np.where(np.isnan(PO), HO, PO)
    aj = np.array([auc(PJ[:, j], Y[:, j]) for j in range(len(names))])
    ao = np.array([auc(PO[:, j], Y[:, j]) for j in range(len(names))])
    eff = lambda F: np.array([Y[:, j].var() / (Y[:, j] - F[:, j]).var() for j in range(len(names))])
    reg = lambda F: np.array([1 / (1 - np.corrcoef(Y[:, j], F[:, j])[0, 1] ** 2) for j in range(len(names))])
    print(f"  mean AUC: Jev {aj.mean():.3f}, DeepSeek {ao.mean():.3f}; Jev higher on {int((aj > ao).sum())} predicates, DeepSeek on {int((ao > aj).sum())}")
    print(f"  mean per-row accuracy: Jev {((PJ >= 0.5) == (Y == 1)).mean():.2%}, DeepSeek {((HO >= 0.5) == (Y == 1)).mean():.2%}")
    print(f"  labelling saved by bias correction (median over predicates): difference Jev {np.median(eff(PJ)):.2f} / DeepSeek {np.median(eff(PO)):.2f}; regression Jev {np.median(reg(PJ)):.2f} / DeepSeek {np.median(reg(PO)):.2f}")
    print(f"  share of probabilities near 0 or 1: Jev {((PJ < 0.02) | (PJ > 0.98)).mean():.0%}, DeepSeek {((PO < 0.02) | (PO > 0.98)).mean():.0%}")
    out["proxy"] = {"auc_jev": aj.mean(), "auc_open": ao.mean(), "jev_better": int((aj > ao).sum()), "open_better": int((ao > aj).sum()),
                    "eff_diff_jev": float(np.median(eff(PJ))), "eff_diff_open": float(np.median(eff(PO))), "eff_reg_jev": float(np.median(reg(PJ))), "eff_reg_open": float(np.median(reg(PO)))}
    (OUT / "packing.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
