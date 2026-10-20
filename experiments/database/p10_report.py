"""Phase 10 analysis: the lines T and G and the readings of experiments/database/protocol/p10.txt for one model. Reads the collected answers only and calls no model.

    python experiments/database/p10_report.py kev-4b       out/database/p10/kev-4b/report.txt and results.json

The analysis runs only once every arm of the model is settled (complete, or missing for good under rules 2 and 4); before that it prints the
inventory and stops, so that no answer is looked at while arms are still being collected. With P10_REHEARSAL=1 it reads the rehearsal's
results, and the Jev column is made up from the labels (there are no Jev answers for made-up rows).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys

import numpy as np
from scipy.stats import rankdata

import p10_data as D
import p10_eval as E
from kevdb import JEV_MODEL
from paths import ROOT

BOOT, SEED = 2000, 20261015
# Phase 6's rule: accuracy at most 2 points below Jev, AUC at most 0.02 below, Brier at most 0.02 above
BARS = {"acc": -0.02, "auc": -0.02, "brier": +0.02}
GUARD_BAR = -0.02
MISSING_BAR = 0.05  # rule 6: a dataset with more than 5% of its requests unanswered is judged below
# Half the server's rounding step (4 decimals): an answer of exactly 0 or 1 is moved this far in, so its logit is finite. Changes nothing else
CLIP = 5e-5
# Ditto's F1 with each dataset's full training split (Li et al., arXiv 2004.00584, table 5), in percent
DITTO = {"abt-buy": 89.33, "walmart-amazon": 86.76, "amazon-google": 75.58, "dblp-acm": 98.99}
PRODUCTS = ("abt-buy", "amazon-google", "walmart-amazon")


# ---------- Loading ----------

def read(model: str, arm: str, name: str) -> list[dict] | None:
    path = E.out_dir(model, arm) / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else None


def made_up_p(ids: list[str], labels: np.ndarray) -> np.ndarray:
    """Rehearsal stand-in for Jev: a fixed, mostly right probability per row."""
    u = np.array([int(hashlib.sha256(f"p10.jev|{i}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF for i in ids])
    return np.round(np.where(labels == 1, 0.55 + 0.44 * u, 0.3 * u ** 3), 4)


def jev_test(dataset: str, ids: list[str], labels: np.ndarray) -> np.ndarray:
    if D.REHEARSAL:
        return made_up_p(ids, labels)
    got = {r["id"]: r for r in map(json.loads, (ROOT / "out" / "entity_match" / f"{dataset}.test.{JEV_MODEL}.jsonl").read_text().splitlines())}
    assert all(got[i]["label"] == y for i, y in zip(ids, labels)), f"{dataset}: Jev's file has other labels"
    return np.array([np.nan if got[i].get("p") is None else got[i]["p"] for i in ids], float)


def jev_e2e(dataset: str, rows: list[dict]) -> np.ndarray:
    labels = np.array([r["label"] for r in rows])
    if D.REHEARSAL:
        return made_up_p([r["id"] for r in rows], labels)
    got = {}
    for r in map(json.loads, (ROOT / "out" / "database" / "p8" / "jev" / f"{dataset}.pair.rank.sub.k10.jsonl").read_text().splitlines()):
        got[f"{r['lid']}|{r['rids'][0]}"] = None if "error" in r else r["p"][0]
    return np.array([np.nan if got.get(r["id"]) is None else got[r["id"]] for r in rows], float)


def noul(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(probability with NaN where there is no answer, label, id) of a set of single yes/no requests."""
    return np.array([np.nan if "error" in r else r["p"] for r in rows], float), np.array([r["label"] for r in rows]), [r["id"] for r in rows]


# ---------- Metrics. A request without an answer is a wrong decision and has probability 0.5 (rule 5) ----------

def decide(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(p), 1 - y, p >= 0.5).astype(int)


def filled(p: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(p), 0.5, p)


def auc(score: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUC along the last axis (ties count half)."""
    pos = (y == 1).sum(-1)
    neg = y.shape[-1] - pos
    ranks = rankdata(score, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((ranks * (y == 1)).sum(-1) - pos * (pos + 1) / 2) / (pos * neg)


def f1(d: np.ndarray, y: np.ndarray) -> np.ndarray:
    tp, fp, fn = (d * y).sum(-1), (d * (1 - y)).sum(-1), ((1 - d) * y).sum(-1)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def ece_mass(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    return float(sum(abs(p[ix].mean() - y[ix].mean()) * len(ix) for ix in np.array_split(np.argsort(p, kind="stable"), bins)) / len(p))


def stats(p: np.ndarray, y: np.ndarray, idx: np.ndarray | None = None) -> dict[str, np.ndarray | float]:
    """acc, auc, brier, f1 on the rows (or, with idx of shape (draws, n), on every resample at once)."""
    d, q = decide(p, y), filled(p)
    if idx is not None:
        d, q, y = d[idx], q[idx], y[idx]
    return {"acc": (d == y).mean(-1), "auc": auc(q, y), "brier": ((q - y) ** 2).mean(-1), "f1": f1(d, y)}


def draws(n: int, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, (BOOT, n))


def paired(pa: np.ndarray, pb: np.ndarray, y: np.ndarray) -> dict[str, tuple[float, float, float]]:
    """a - b per statistic with a 95% percentile interval from resampling rows (the same rows for both)."""
    idx = draws(len(y))
    a, b, ba, bb = stats(pa, y), stats(pb, y), stats(pa, y, idx), stats(pb, y, idx)
    return {k: (float(a[k] - b[k]), float(np.nanpercentile(ba[k] - bb[k], 2.5)), float(np.nanpercentile(ba[k] - bb[k], 97.5))) for k in a}


def readings(p: np.ndarray, y: np.ndarray) -> dict:
    """Where the errors went (on answered rows): false and missed matches, mean p on non-matching pairs, share of 0.1 < p < 0.9, equal-mass
    calibration error, and the share of all errors that sit among the 30% least confident pairs (is a cascade still possible?)."""
    q, d = filled(p), decide(p, y)
    wrong = d != y
    unsure = np.argsort(np.abs(q - 0.5), kind="stable")[: int(round(0.3 * len(q)))]
    return {"false_matches": int(((d == 1) & (y == 0)).sum()), "missed_matches": int(((d == 0) & (y == 1)).sum()), "mean_p_non_match": float(q[y == 0].mean()),
            "share_unsure": float(((q > 0.1) & (q < 0.9)).mean()), "ece_mass": ece_mass(q, y), "errors_in_least_confident_30": float(wrong[unsure].sum() / max(wrong.sum(), 1))}


def judge_t(diff: dict[str, tuple[float, float, float]], missing: float) -> tuple[bool, dict[str, bool], dict[str, bool]]:
    """Line T on one held-out dataset from the differences to Jev (point, low, high): point estimates decide; an interval containing the line is
    "not separable" from it; more than 5% unanswered is below (rule 6)."""
    crit = {"acc": diff["acc"][0] >= BARS["acc"], "auc": diff["auc"][0] >= BARS["auc"], "brier": diff["brier"][0] <= BARS["brier"]}
    separable = {k: not (diff[k][1] <= BARS[k] <= diff[k][2]) for k in BARS}
    return all(crit.values()) and missing <= MISSING_BAR, crit, separable


# ---------- Temperature ----------

def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, CLIP, 1 - CLIP)
    return np.log(q / (1 - q))


def scale(p: np.ndarray, T: float) -> np.ndarray:
    """p_T = sigmoid(logit(p_1) / T); rows without an answer stay without one."""
    return np.where(np.isnan(p), np.nan, 1 / (1 + np.exp(-logit(p) / T)))


# Kev's own temperature fit (kev/metrics.py, TEMPERATURE_FIT, used for the released checkpoints): the minimum mean negative log likelihood
# over a 121-point log grid from 0.25 to 4. The grid keeps T finite when a checkpoint separates its pool labels perfectly, where the
# unbounded maximum runs to T -> 0 (the first full rehearsal ran into exactly that)
GRID = np.exp(np.linspace(np.log(0.25), np.log(4), 121))


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """Maximum likelihood of the human labels over Kev's temperature grid; rows without an answer are left out."""
    keep = ~np.isnan(p)
    z, t = logit(p[keep]), y[keep]
    s = z[None, :] / GRID[:, None]
    return float(GRID[int(np.argmin((np.logaddexp(0, s) - t * s).mean(1)))])


def temperature(model: str, arm: str, datasets) -> tuple[float, int, int]:
    """T fitted on the pool labels of these datasets: (T, labels, positives)."""
    ps, ys = [], []
    for d in datasets:
        p, y, _ = noul(read(model, arm, f"pool.{d}"))
        ps.append(p)
        ys.append(y)
    p, y = np.concatenate(ps), np.concatenate(ys)
    return fit_temperature(p, y), int((~np.isnan(p)).sum()), int(y[~np.isnan(p)].sum())


# ---------- Guards ----------

def guard_units(name: str, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(correct, weight) per row of a guard set: g1 a review text (one judgement per labelled review row, phase 6 scored per row), g2 a judgement,
    g3 a record (its questions; choice: the answered option, noul: p >= 0.5, score: the most probable level). Unanswered counts as wrong."""
    correct, weight = [], []
    for r in rows:
        if name == "g1":
            ok = 0 if "error" in r else sum((r["p"] >= 0.5) == bool(y) for y in r["labels"])
            correct.append(ok)
            weight.append(len(r["labels"]))
        elif name == "g2":
            correct.append(0 if "error" in r else int((r["p"] >= 0.5) == bool(r["label"])))
            weight.append(1)
        else:
            ok = 0
            if "error" not in r:
                got = r.get("answers") or {next(iter(r["labels"])): {"p": r.get("p")}}
                for q, label in r["labels"].items():
                    a, kind = got[q], r["types"][q]
                    ok += (a["p"] >= 0.5) == bool(label) if kind == "noul" else a["choice"] == label if kind == "choice" else a["level"] == label
            correct.append(int(ok))
            weight.append(len(r["labels"]))
    return np.array(correct, float), np.array(weight, float)


def guard(model: str, arm: str, name: str, released: tuple[np.ndarray, np.ndarray]) -> dict:
    c, w = guard_units(name, read(model, arm, name))
    c0, w0 = released
    assert np.array_equal(w, w0), f"{arm} {name}: other rows than the released arm"
    idx = draws(len(w))
    acc, acc0 = c.sum() / w.sum(), c0.sum() / w0.sum()
    diff = (c[idx].sum(1) / w[idx].sum(1)) - (c0[idx].sum(1) / w0[idx].sum(1))
    return {"acc": float(acc), "released": float(acc0), "diff": float(acc - acc0), "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
            "judgements": int(w.sum()), "pass": bool(acc - acc0 >= GUARD_BAR)}


# ---------- End to end and option order (phase 8's subsets) ----------

def e2e_tally(rows: list[dict], p: np.ndarray, truth: set, lids: list[str], one: bool = False) -> np.ndarray:
    """Per left record (correct matches, wrong matches, true matches). True matches outside the candidates count, so recall is end to end.
    one: keep at most one match per left record, the most probable of those at p >= 0.5."""
    index = {lid: i for i, lid in enumerate(lids)}
    out = np.zeros((len(lids), 3))
    for lid, _ in truth:
        if lid in index:
            out[index[lid], 2] += 1
    best: dict[str, tuple[float, str]] = {}
    for r, x in zip(rows, p):
        if np.isnan(x) or x < 0.5:
            continue
        if one:
            if x > best.get(r["lid"], (-1, ""))[0]:
                best[r["lid"]] = (x, r["rid"])
        else:
            out[index[r["lid"]], 0 if (r["lid"], r["rid"]) in truth else 1] += 1
    for lid, (_, rid) in best.items():
        out[index[lid], 0 if (lid, rid) in truth else 1] += 1
    return out


def prf(t: np.ndarray) -> tuple:
    tp, fp, n = t[..., 0], t[..., 1], t[..., 2]
    precision, recall = tp / np.maximum(tp + fp, 1), tp / np.maximum(n, 1)
    return precision, recall, 2 * precision * recall / np.maximum(precision + recall, 1e-12)


def f1_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    idx = draws(len(a))
    diff = (prf(a[idx].sum(1))[2] - prf(b[idx].sum(1))[2]) * 100
    return float((prf(a.sum(0))[2] - prf(b.sum(0))[2]) * 100), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def order_reading(rows_by_order: dict[str, list[dict]], truth: set) -> dict:
    """Single choice on the first 100 left records, three candidate orders: accuracy (the true match chosen when there is exactly one among the
    candidates, "none" chosen when there is none), share of choices in the first three positions, and agreement of the chosen record between orders."""
    picked, out = {}, {}
    for order, rows in rows_by_order.items():
        chosen, right, scored, first3 = {}, 0, 0, 0
        for r in rows:
            if "error" in r:
                chosen[r["lid"]] = None
                continue
            c = r["answers"]["match"]["choice"]
            pos = None if c == "none" else int(c[1:]) - 1
            chosen[r["lid"]] = "none" if pos is None else r["rids"][pos]
            first3 += pos is not None and pos < 3
            hits = [rid for rid in r["rids"] if (r["lid"], rid) in truth]
            if len(hits) <= 1:
                scored += 1
                right += chosen[r["lid"]] == (hits[0] if hits else "none")
        picked[order] = chosen
        out[order] = {"accuracy": right / max(scored, 1), "scored": scored, "chosen_first_3": first3 / max(len(rows), 1), "unanswered": sum(v is None for v in chosen.values())}
    names = list(picked)
    out["agreement"] = {f"{a}/{b}": float(np.mean([picked[a][l] == picked[b][l] and picked[a][l] is not None for l in picked[a]]))
                        for i, a in enumerate(names) for b in names[i + 1:]}
    return out


# ---------- Report ----------

def report(model: str) -> int:
    lines: list[str] = []
    say = lambda s="": lines.append(s)
    inventory = io.StringIO()
    with contextlib.redirect_stdout(inventory):
        pending = E.check(model)
    title = f"Phase 10, {model}: does fine-tuning on public matching pairs bring Kev to Jev's zero-shot level on matching data it was not trained on?"
    say(title + (" (rehearsal: every row made up, the Jev column made up from the labels)" if D.REHEARSAL else ""))
    say(inventory.getvalue().rstrip())
    if pending:
        say(f"\n{pending} arm(s) still pending: no answer is analysed until every arm is settled (p10.txt, rules of collection).")
        print("\n".join(lines))
        return 1
    status = {arm: E.status(model, arm) for arm in E.ARMS}
    have = lambda arm: status[arm] == "complete"
    results: dict = {"model": model, "rehearsal": D.REHEARSAL, "status": status}
    if not have("released"):
        say("\nthe released arm is missing: no line can be judged; verdict incomplete")
        results["verdict"] = "incomplete"
        return finish(model, lines, results)

    # Temperatures
    say("\n== Temperature (maximum likelihood on the human pool labels of the datasets each checkpoint trained on; never the held-out dataset)")
    temps = {}
    for arm in E.TRAINED:
        if have(arm):
            T, n, pos = temperature(model, arm, D.ARMS[arm])
            temps[arm] = T
            say(f"   {arm:<20}T = {T:.3f}  ({n} labels, {pos} matches)" + ("  at the end of Kev's grid: the likelihood still improves beyond it" if T in (GRID[0], GRID[-1]) else ""))
    results["temperature"] = temps

    # Line T
    say("\n== Line T: loo-D with its fitted temperature on D's full test set, against Jev (on par: accuracy >= Jev - 2.0 points, AUC >= Jev - 0.02, Brier <= Jev + 0.02)")
    say(f"   {'held out':<16}{'':<10}{'acc':>8}{'AUC':>8}{'Brier':>8}{'F1':>8}   {'diff vs Jev (95% CI)':<70}missing  verdict")
    line_t, test_data = {}, {}
    for d in D.EM:
        arm = f"loo-{d}"
        if not have(arm):
            say(f"   {d:<16}arm missing ({status[arm]})")
            line_t[d] = {"on_par": False, "missing_arm": True}
            continue
        p1, y, ids = noul(read(model, arm, f"test.{d}"))
        p0 = noul(read(model, "released", f"test.{d}"))[0]
        pj = jev_test(d, ids, y)
        pt = scale(p1, temps[arm])
        test_data[d] = (pt, p1, p0, pj, y)
        diff = paired(pt, pj, y)
        missing = float(np.isnan(p1).mean())
        on_par, crit, separable = judge_t(diff, missing)
        line_t[d] = {"on_par": on_par, "criteria": crit, "separable": separable, "missing": missing, "diff": diff, "loo": {k: float(v) for k, v in stats(pt, y).items()},
                     "loo_raw": {k: float(v) for k, v in stats(p1, y).items()}, "jev": {k: float(v) for k, v in stats(pj, y).items()}}
        for label, p in (("Jev", pj), (f"{arm} T", pt), (f"{arm} T=1", p1)):
            s = stats(p, y)
            extra = ""
            if label.endswith(" T"):
                extra = "   " + "  ".join(f"{k} {diff[k][0]:+.3f} ({diff[k][1]:+.3f} to {diff[k][2]:+.3f})" + ("" if separable[k] else "*") for k in ("acc", "auc", "brier"))
                extra = f"{extra:<73}{missing:>6.1%}  " + ("on par" if on_par else "below: " + ", ".join(k for k, ok in crit.items() if not ok) + (" (over 5% unanswered)" if missing > MISSING_BAR else ""))
            say(f"   {d if label == 'Jev' else '':<16}{label:<22}{s['acc']:>7.1%}{s['auc']:>8.3f}{s['brier']:>8.3f}{s['f1']:>8.3f}{extra}")
    say("   * the interval contains the line: not separable from it")
    results["line_T"] = line_t

    # Guard G
    released = {g: guard_units(g, read(model, "released", g)) for g in E.GUARDS}
    counts = {g: f"{len(w):,} {unit}, {int(w.sum()):,} judgements" for (g, (_, w)), unit in zip(released.items(), ("review texts", "sentence-predicate pairs", "records"))}
    say(f"\n== Guard G: accuracy at most 2.0 points below the released checkpoint on g1 ({counts['g1']}), g2 ({counts['g2']}), g3 (decision-v7 development, {counts['g3']})")
    line_g = {}
    for arm in E.TRAINED:
        if not have(arm):
            continue
        line_g[arm] = {g: guard(model, arm, g, released[g]) for g in E.GUARDS}
        say(f"   {arm:<20}" + "   ".join(f"{g} {x['acc']:.1%} vs {x['released']:.1%} ({x['diff'] * 100:+.1f}, {x['ci'][0] * 100:+.1f} to {x['ci'][1] * 100:+.1f}) {'pass' if x['pass'] else 'FAIL'}"
                                        for g, x in line_g[arm].items()))
    results["line_G"] = line_g

    # Verdict
    met = [d for d in D.EM if line_t[d]["on_par"] and all(x["pass"] for x in line_g.get(f"loo-{d}", {}).values())]
    forgets = [d for d in D.EM if line_t[d]["on_par"] and d not in met]
    incomplete = [d for d in D.EM if not have(f"loo-{d}")]
    verdict = "incomplete" if incomplete else "reaches Jev's zero-shot level on matching" if len(met) == 4 else f"{len(met)} of 4"
    say(f"\n== Verdict for {model}: {verdict}")
    say(f"   on par and no forgetting: {', '.join(met) or 'none'}" + (f"; on par but forgets: {', '.join(forgets)}" if forgets else "")
        + (f"; missing arms: {', '.join(incomplete)}" if incomplete else ""))
    for d in D.EM:
        if d not in met and d not in incomplete and not line_t[d]["on_par"]:
            say(f"   {d}: below on " + ", ".join(k for k, ok in line_t[d].get("criteria", {}).items() if not ok))
    results["verdict"] = verdict
    results["met"], results["forgets"] = met, forgets

    # Improvement over the released checkpoint
    say("\n== Improvement of each loo arm over the released checkpoint on its held-out dataset (loo with its temperature; released as it ships). Improves = F1 interval above zero")
    say("   the three product folds share their question with two training datasets; dblp-acm is asked the publication question, never seen in training")
    improve = {}
    for group, members in (("same question", PRODUCTS), ("new question", ("dblp-acm",))):
        say(f"   -- {group}")
        for d in members:
            if d not in test_data:
                continue
            pt, p1, p0, pj, y = test_data[d]
            diff = paired(pt, p0, y)
            improve[d] = {"diff": diff, "improves": diff["f1"][1] > 0, "released": {k: float(v) for k, v in stats(p0, y).items()}}
            say(f"   {d:<16}released acc {improve[d]['released']['acc']:.1%} F1 {improve[d]['released']['f1']:.3f}   "
                + "  ".join(f"{k} {v[0]:+.3f} ({v[1]:+.3f} to {v[2]:+.3f})" for k, v in diff.items()) + ("   improves" if improve[d]["improves"] else "   no clear improvement"))
    results["improvement"] = improve

    # Reading: temperature alone
    say("\n== Reading: the released checkpoint with a temperature fitted the same way (on the other three datasets' pool labels) - how much of a Brier gain a temperature alone gives")
    alone = {}
    for d in test_data:
        pt, p1, p0, pj, y = test_data[d]
        T0, _, _ = temperature(model, "released", [x for x in D.EM if x != d])
        alone[d] = {"T": T0, "brier": float(stats(scale(p0, T0), y)["brier"]), "brier_as_shipped": float(stats(p0, y)["brier"]), "brier_loo": float(stats(pt, y)["brier"])}
        say(f"   {d:<16}T {T0:.3f}: Brier {alone[d]['brier_as_shipped']:.4f} as shipped, {alone[d]['brier']:.4f} with T; loo-{d} {alone[d]['brier_loo']:.4f}")
    results["released_with_temperature"] = alone

    # Where the errors went
    say("\n== Where the errors went (held-out test sets): false / missed matches, mean p on non-matching pairs, share 0.1 < p < 0.9, calibration error, errors among the 30% least confident")
    errs = {}
    for d in test_data:
        pt, p1, p0, pj, y = test_data[d]
        errs[d] = {name: readings(p, y) for name, p in (("jev", pj), ("released", p0), ("loo", pt))}
        for name, x in errs[d].items():
            say(f"   {d if name == 'jev' else '':<16}{name:<10}{x['false_matches']:>6}{x['missed_matches']:>6}{x['mean_p_non_match']:>8.3f}{x['share_unsure']:>8.1%}{x['ece_mass']:>8.3f}{x['errors_in_least_confident_30']:>8.0%}")
    results["errors"] = errs

    # The all arm
    if have("all"):
        say(f"\n== The all arm (trained on all four, T = {temps['all']:.3f}) on the four test sets, next to Jev and Ditto (full training split per dataset; not like for like)")
        every = {}
        for d in D.EM:
            p1, y, ids = noul(read(model, "all", f"test.{d}"))
            s, j = stats(scale(p1, temps["all"]), y), stats(jev_test(d, ids, y), y)
            every[d] = {"all": {k: float(v) for k, v in s.items()}, "jev": {k: float(v) for k, v in j.items()}, "ditto_f1": DITTO[d] / 100}
            say(f"   {d:<16}acc {s['acc']:.1%} AUC {s['auc']:.3f} Brier {s['brier']:.3f} F1 {s['f1']:.3f}   Jev F1 {j['f1']:.3f}   Ditto F1 {DITTO[d] / 100:.3f}")
        results["all_arm"] = every

    # End to end on phase 8's subsets
    P8 = E.p8_module()
    say("\n== End to end on phase 8's subsets (pair form, stored top-10 candidates; recall counts true matches outside the candidates)")
    e2e = {}
    for dataset, arm in E.E2E.items():
        if not have(arm):
            continue
        rows = read(model, arm, f"e2e.{dataset}")
        base = read(model, "released", f"e2e.{dataset}")
        left, _, truth, _ = P8._DATA[dataset]
        lids = P8.subset(dataset, left)
        p_arm, p_rel, p_jev = noul(rows)[0], noul(base)[0], jev_e2e(dataset, rows)
        for one in ((False, True) if dataset == "abt-buy" else (False,)):
            t = {name: e2e_tally(rows, p, truth, lids, one) for name, p in (("jev", p_jev), ("released", p_rel), (arm, p_arm))}
            key = f"{dataset}{' at most one' if one else ''}"
            e2e[key] = {name: dict(zip(("precision", "recall", "f1"), map(float, prf(x.sum(0))))) for name, x in t.items()}
            e2e[key]["f1_diff_vs_released"] = f1_diff(t[arm], t["released"])
            e2e[key]["f1_diff_vs_jev"] = f1_diff(t[arm], t["jev"])
            say(f"   {key:<26}" + "   ".join(f"{name} P {x['precision']:.1%} R {x['recall']:.1%} F1 {x['f1']:.1%}" for name, x in e2e[key].items() if isinstance(x, dict))
                + f"   F1 {arm} - released {e2e[key]['f1_diff_vs_released'][0]:+.1f} ({e2e[key]['f1_diff_vs_released'][1]:+.1f} to {e2e[key]['f1_diff_vs_released'][2]:+.1f}),"
                + f" - Jev {e2e[key]['f1_diff_vs_jev'][0]:+.1f} ({e2e[key]['f1_diff_vs_jev'][1]:+.1f} to {e2e[key]['f1_diff_vs_jev'][2]:+.1f})")
    results["end_to_end"] = e2e

    # Option order
    if have("loo-abt-buy"):
        say(f"\n== Option order: phase 8's single choice on the first {E.ORDER_ROWS} left records of the abt-buy subset, three candidate orders (a baseline; no change expected)")
        truth = P8._DATA["abt-buy"][2]
        orders = {}
        for arm in ("released", "loo-abt-buy"):
            orders[arm] = order_reading({o: read(model, arm, f"order.{o}") for o in E.ORDERS}, truth)
            x = orders[arm]
            say(f"   {arm:<14}" + "   ".join(f"{o}: right {x[o]['accuracy']:.0%} of {x[o]['scored']}, first three {x[o]['chosen_first_3']:.0%}" for o in E.ORDERS)
                + "   same choice " + ", ".join(f"{k} {v:.0%}" for k, v in x["agreement"].items()))
        results["option_order"] = orders
    return finish(model, lines, results)


def finish(model: str, lines: list[str], results: dict) -> int:
    print("\n".join(lines))
    (E.out_dir(model) / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in E.MODELS:
        sys.exit(__doc__)
    sys.exit(report(sys.argv[1]))
