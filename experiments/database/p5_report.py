"""Analysis of the follow-up experiment. Reads the collected judgements only and calls no model. Questions and reading rules: experiments/database/protocol/p5.txt."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from p3_aggregates import load
from p4_collect import PREDICATES, sentences
from paths import OUT as RESULTS, ROOT

OUT, P4 = RESULTS / "p5", RESULTS / "p4"
SHARES, BOOT, SEED = (0.1, 0.2, 0.3, 0.5), 2000, 20261007


def mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    only_a, only_b = int((a & ~b).sum()), int((b & ~a).sum())
    n, k = only_a + only_b, min(only_a, only_b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def f1(pred: np.ndarray, y: np.ndarray) -> float:
    return 2 * (pred & y).sum() / max(pred.sum() + y.sum(), 1)


def concentration(wrong: np.ndarray, unsure_first: np.ndarray) -> dict[float, float]:
    """Rank rows from least to most certain for Jev; the share of its errors within the top b. However strong the upper model, a cascade can recover at most this much."""
    cum = np.cumsum(wrong[unsure_first])
    return {b: float(cum[int(round(b * len(wrong))) - 1] / max(wrong.sum(), 1)) for b in SHARES}


def cascade(J: np.ndarray, U: np.ndarray, Y: np.ndarray, score: np.ndarray, rng: np.random.Generator) -> dict:
    """J, U, Y: row × predicate boolean matrices (Jev, upper model, human truth). A larger score means less certain."""
    n = len(Y)

    def run(ix: np.ndarray) -> tuple[float, float, dict[float, float]]:
        j, u, y, order = J[ix], U[ix], Y[ix], np.argsort(-score[ix], kind="stable")
        base, full = (j == y).mean(), (u == y).mean()
        rec = {}
        for b in SHARES:
            up = np.zeros(len(ix), bool)
            up[order[: int(round(b * len(ix)))]] = True
            acc = (np.where(up[:, None], u, j) == y).mean()
            rec[b] = (acc - base) / (full - base) if full != base else float("nan")
        return base, full, rec

    base, full, rec = run(np.arange(n))
    boots = [run(rng.integers(0, n, n)) for _ in range(BOOT)]
    gap = np.array([f - b for b, f, _ in boots])
    r30 = np.array([r[0.3] for _, _, r in boots])
    return {"n": n, "jev": float(base), "upper": float(full), "gap_pp": 100 * float(full - base), "gap_ci": [100 * float(np.quantile(gap, q)) for q in (0.025, 0.975)],
            "recovered": {b: float(v) for b, v in rec.items()}, "recovered30_ci": [float(np.nanquantile(r30, q)) for q in (0.025, 0.975)]}


def show_cascade(title: str, c: dict) -> None:
    print(f"{title}: Jev {c['jev']:.2%}, upper {c['upper']:.2%}, difference {c['gap_pp']:+.2f} points (95% interval {c['gap_ci'][0]:+.2f} to {c['gap_ci'][1]:+.2f})")
    print("  escalating the least certain " + ", ".join(f"{b:.0%} → recovers {r:.0%}" for b, r in c["recovered"].items()) +
          f"; 95% interval at 30%: {c['recovered30_ci'][0]:.0%} to {c['recovered30_ci'][1]:.0%} (random picking gives 30%, target ≥60%)")


def report() -> None:
    rng = np.random.default_rng(SEED)
    out: dict = {"collect": json.loads((OUT / "collect.json").read_text())}

    # ---------- The free part: how concentrated are Jev's errors among its least certain rows? ----------
    print("1. Are Jev's errors concentrated in its least certain rows? (with a perfect upper model, the most of the gap that escalating the top b of rows can recover)")
    print(f"{'data':34s}{'rows':>7s}{'Jev acc':>10s}" + "".join(f"{'top ' + format(b, '.0%'):>9s}" for b in SHARES))
    conc = {}

    def add(name: str, p: np.ndarray, y: np.ndarray) -> None:
        wrong = (p >= 0.5) != (y == 1)
        conc[name] = {"n": len(y), "accuracy": float(1 - wrong.mean()), **{str(b): v for b, v in concentration(wrong, np.argsort(np.abs(p - 0.5), kind="stable")).items()}}
        print(f"{name:34s}{len(y):>8d}{1 - wrong.mean():>11.1%}" + "".join(f"{conc[name][str(b)]:>10.0%}" for b in SHARES))

    for label, path in (("Products Abt-Buy", "entity_match/abt-buy.test"), ("Products Walmart-Amazon", "entity_match/walmart-amazon.test"), ("Software Amazon-Google", "entity_match/amazon-google.test"),
                        ("Papers DBLP-ACM", "entity_match/dblp-acm.test"), ("End-to-end Abt-Buy pairs", "table_match/abt-buy"), ("End-to-end DBLP-Scholar pairs", "table_match/dblp-scholar")):
        rows = [json.loads(line) for line in (ROOT / f"out/{path}.jev-1.13.0.jsonl").read_text().splitlines()]
        add(label, np.array([r["p"] for r in rows]), np.array([r["label"] for r in rows]))
    t = load()
    add("Movie reviews \"positive\"", t["p"], t["y"])
    rows = sentences()
    single = {r["sid"]: r["p"] for r in map(json.loads, (P4 / "mams.jev.jsonl").read_text().splitlines())}
    per = []
    for k in PREDICATES:
        p, y = np.array([single[r["sid"]][k] for r in rows]), np.array([r["truth"][k] for r in rows])
        wrong = (p >= 0.5) != (y == 1)
        per.append([1 - wrong.mean()] + list(concentration(wrong, np.argsort(np.abs(p - 0.5), kind="stable")).values()))
    per = np.array(per).mean(axis=0)
    conc["Restaurant reviews, mean of 21 predicates"] = {"n": len(rows), "accuracy": float(per[0]), **{str(b): float(v) for b, v in zip(SHARES, per[1:])}}
    print(f"{'Restaurant reviews, mean of 21 predicates':34s}{len(rows):>8d}{per[0]:>11.1%}" + "".join(f"{v:>10.0%}" for v in per[1:]))
    out["concentration"] = conc

    # ---------- Movies ----------
    opus = {r["text"]: r["p"] for r in map(json.loads, (OUT / "movie.opus.jsonl").read_text().splitlines()) if r["p"] is not None}
    m = np.array([x in opus for x in t["text"]])
    y, pj = t["y"][m] == 1, t["p"][m]
    J, L, O = pj >= 0.5, t["llm"][m] == 1, np.array([opus[x] for x in t["text"][m]]) >= 0.5
    F = t["v41flash.off.hard"][m] >= 0.5
    band = (pj > 0.1) & (pj < 0.9)
    acc = lambda pred, mask=slice(None): float((pred[mask] == y[mask]).mean())
    print(f"\n2. Movie reviews ({m.sum()} reviews, against human labels)")
    print(f"  accuracy: Jev {acc(J):.2%}, GPT 6 Luna {acc(L):.2%}, DeepSeek V4.1 Flash {acc(F):.2%}, Opus 5.5 {acc(O):.2%}; paired test Opus vs Jev p = {mcnemar(O == y, J == y):.3f}")
    print(f"  rows Jev is unsure about (probability 0.1-0.9, {band.mean():.1%} of rows): Jev {acc(J, band):.2%}, Luna {acc(L, band):.2%}, Opus {acc(O, band):.2%}; other rows: Jev {acc(J, ~band):.2%}, Opus {acc(O, ~band):.2%}")
    print(f"  Opus agrees with Jev on {(O == J).mean():.2%}; judged positive: Opus {O.mean():.2%}, Jev {J.mean():.2%}, humans {y.mean():.2%}")
    movie = cascade(J[:, None], O[:, None], y[:, None], -np.abs(pj - 0.5), rng)
    show_cascade("  cascade", movie)
    out["movie"] = {"n": int(m.sum()), "acc": {"jev": acc(J), "luna": acc(L), "v41flash": acc(F), "opus": acc(O)}, "p_opus_vs_jev": mcnemar(O == y, J == y),
                    "band_share": float(band.mean()), "band_acc": {"jev": acc(J, band), "luna": acc(L, band), "opus": acc(O, band)},
                    "sure_acc": {"jev": acc(J, ~band), "opus": acc(O, ~band)}, "cascade": movie}

    # ---------- Restaurant reviews ----------
    names = list(PREDICATES)
    read = lambda path: {r["sid"]: r["p"] for r in map(json.loads, Path(path).read_text().splitlines()) if r["p"] is not None}
    op, jp, lp, dp = read(OUT / "mams.opus.packed.jsonl"), read(P4 / "mams.jev.packed.jsonl"), read(P4 / "mams.luna.low.packed.jsonl"), read(P4 / "mams.v41flash.off.packed.jsonl")
    rows = [r for r in sentences(1000) if r["sid"] in op]
    mat = lambda src: np.array([[src[r["sid"]][k] for k in names] for r in rows])
    Y = np.array([[r["truth"][k] for k in names] for r in rows]) == 1
    PJ = mat(jp)
    J, L, D, O = PJ >= 0.5, mat(lp) >= 0.5, mat(dp) >= 0.5, mat(op) >= 0.5
    mf1 = lambda M: float(np.mean([f1(M[:, j], Y[:, j]) for j in range(len(names))]))
    print(f"\n3. Restaurant reviews ({len(rows)} sentences × {len(names)} predicates, all 21 questions packed)")
    print(f"  accuracy / F1: Jev {(J == Y).mean():.2%} / {mf1(J):.3f}, GPT 6 Luna {(L == Y).mean():.2%} / {mf1(L):.3f}, "
          f"DeepSeek V4.1 Flash {(D == Y).mean():.2%} / {mf1(D):.3f}, Opus 5.5 {(O == Y).mean():.2%} / {mf1(O):.3f}")
    mams = cascade(J, O, Y, (PJ * (1 - PJ)).sum(axis=1), rng)
    show_cascade("  cascade (upper: Opus)", mams)
    mams_luna = cascade(J, L, Y, (PJ * (1 - PJ)).sum(axis=1), rng)
    show_cascade("  the same sentences with Luna as the upper model", mams_luna)
    per_pred = {k: {"jev": float((J[:, j] == Y[:, j]).mean()), "opus": float((O[:, j] == Y[:, j]).mean())} for j, k in enumerate(names)}
    better = sum(v["opus"] > v["jev"] for v in per_pred.values())
    print(f"  Opus is more accurate than Jev on {better} of the 21 predicates")
    out["mams"] = {"n": len(rows), "acc": {"jev": float((J == Y).mean()), "luna": float((L == Y).mean()), "v41flash": float((D == Y).mean()), "opus": float((O == Y).mean())},
                   "f1": {"jev": mf1(J), "luna": mf1(L), "v41flash": mf1(D), "opus": mf1(O)}, "cascade": mams, "cascade_luna": mams_luna, "per_predicate": per_pred}

    # ---------- Product matching: positive control ----------
    em = [json.loads(line) for line in (OUT / "em.opus.jsonl").read_text().splitlines()]
    answered = {r["id"]: r for r in em if r["p"] is not None}
    full = [json.loads(line) for line in (ROOT / "out/entity_match/walmart-amazon.test.jev-1.13.0.jsonl").read_text().splitlines()]
    y = np.array([r["label"] for r in full]) == 1
    pj = np.array([r["p"] for r in full])
    J = pj >= 0.5
    unsure = (pj > 0.1) & (pj < 0.9)
    C = np.array([(answered[r["id"]]["p"] >= 0.5) if u and r["id"] in answered else j for r, u, j in zip(full, unsure, J)])
    diffs = []
    for _ in range(BOOT):
        ix = rng.integers(0, len(y), len(y))
        diffs.append(f1(C[ix], y[ix]) - f1(J[ix], y[ix]))
    us = [r for r in em if r["stratum"] == "unsure" and r["p"] is not None]
    su = [r for r in em if r["stratum"] != "unsure" and r["p"] is not None]
    ok = lambda rows, field: np.array([(r[field] >= 0.5) == bool(r["label"]) for r in rows])
    # Accuracy of Opus for everything: unsure pairs fully measured, confident pairs weighted from the two sampled strata
    weights = {s: float(((pj >= 0.9) if s == "sure_yes" else (pj <= 0.1)).mean()) for s in ("sure_yes", "sure_no")}
    opus_all = unsure.mean() * ok(us, "p").mean() + sum(w * ok([r for r in su if r["stratum"] == s], "p").mean() for s, w in weights.items())
    jev_all, casc_all = float((J == y).mean()), float((C == y).mean())
    recovered = (casc_all - jev_all) / (opus_all - jev_all)
    print(f"\n4. Positive control: Walmart-Amazon product matching (test set {len(y)} pairs, {unsure.sum()} of them unsure for Jev, {unsure.mean():.1%})")
    print(f"  on the {len(us)} unsure pairs: Jev {ok(us, 'jev_p').mean():.1%}, Opus {ok(us, 'p').mean():.1%} (paired test p = {mcnemar(ok(us, 'p'), ok(us, 'jev_p')):.4f}); "
          f"{len(su)} sampled confident pairs: Jev {ok(su, 'jev_p').mean():.1%}, Opus {ok(su, 'p').mean():.1%}")
    print(f"  F1 on the whole test set: Jev only {f1(J, y):.3f}, unsure pairs to Opus {f1(C, y):.3f}, gain {f1(C, y) - f1(J, y):+.3f} (95% interval {np.quantile(diffs, 0.025):+.3f} to {np.quantile(diffs, 0.975):+.3f})")
    print(f"  accuracy: Jev {jev_all:.2%}, cascade {casc_all:.2%}, Opus for everything (stratified estimate) {opus_all:.2%} → escalating {unsure.mean():.1%} of rows recovers {recovered:.0%} of the gap")
    out["em"] = {"pairs": len(y), "unsure": int(unsure.sum()), "unsure_share": float(unsure.mean()), "unsure_acc": {"jev": float(ok(us, "jev_p").mean()), "opus": float(ok(us, "p").mean())},
                 "sure_acc": {"jev": float(ok(su, "jev_p").mean()), "opus": float(ok(su, "p").mean())}, "f1": {"jev": float(f1(J, y)), "cascade": float(f1(C, y))},
                 "f1_gain_ci": [float(np.quantile(diffs, q)) for q in (0.025, 0.975)], "acc": {"jev": jev_all, "cascade": casc_all, "opus_all_est": float(opus_all)}, "recovered": float(recovered)}

    c = out["collect"]
    print(f"\ncost: movies ${c['movie']['cost_usd']:.2f} ({c['movie']['requests']} requests), product matching ${c['em']['cost_usd']:.2f} ({c['em']['requests']} requests), "
          f"restaurant reviews ${c['mams']['cost_usd']:.2f} ({c['mams']['requests']} requests); balance from ${c['start_balance']:.2f} to ${c['end_balance']:.2f}")
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + "\n")


if __name__ == "__main__":
    sys.exit(report())
