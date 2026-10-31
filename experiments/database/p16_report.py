"""Phase 16 analysis: the readings of experiments/database/protocol/p16.txt. Reads the cell summaries written by p16_run.py and the cached
answers; calls no model.

    python experiments/database/p16_report.py      out/database/p16/report.txt and results.json (p16_rehearsal with P16_REHEARSAL=1)
"""

from __future__ import annotations

import json
import sys

import numpy as np

import p16_data as D
import p16_methods as M
import p16_run as R

PRIMARY = ("Q2", "Q3a", "Q3b", "Q4a", "Q4b-1", "Q4b-2", "Q4b-2t", "Q5", "Q6")
SHAPES = {"small groups": ("Q2",), "weights": ("Q3a", "Q3b"), "join reuse": ("Q4a", "Q4b-1", "Q4b-2", "Q4b-2t"), "threshold": ("Q5",),
          "rare categories": ("Q6",)}
PRIOR = ("U", "C", "G", "abae")
LINE_MEAN, LINE_P10, PAYS, BIAS_AWARE = 0.93, 0.90, 1.5, 1.2
lines: list[str] = []


def say(text: str = "") -> None:
    lines.append(text)


def load() -> dict:
    cells = {}
    for path in (D.RESULTS / "cells").rglob("*.json"):
        r = json.loads(path.read_text())
        cells[r["name"]] = r
    return cells


def curves(cells: dict, part: str, pop: str, cheap: str, query: str) -> dict:
    """design -> budget -> summary; Q5 reads the Q2 cells."""
    source = "Q2" if query == "Q5" else query
    out: dict = {}
    for r in cells.values():
        c = r["cell"]
        if (c["part"], c["pop"], c["cheap"], c["query"], c["pilot"], c["second"]) == (part, pop, cheap, source, M.PILOT, ""):
            out.setdefault(c["design"], {})[c["budget"]] = r
    return out


def metric(r: dict, query: str, truth: str = "human") -> float:
    return r["q5"]["wrong_mean"] + 0.1 if query == "Q5" else r["by_truth"][truth]["loss_mean"]


def covered(r: dict, truth: str = "human") -> bool:
    t = r["by_truth"][truth]
    return t["coverage_mean"] >= LINE_MEAN and (len(t["coverage"]) == 1 or t["coverage_p10"] >= LINE_P10)


def reading(cv: dict, query: str, primary: str, budget: int, prior: tuple) -> dict:
    """The primary reading of p16.txt for one query."""
    target = metric(cv[primary][budget], query)
    rho = {}
    for design, by_budget in cv.items():
        grid = sorted(by_budget)
        rho[design] = M.labels_needed(grid, [metric(by_budget[b], query) for b in grid], target) / budget
    ok = {d: covered(cv[d][budget]) for d in cv if budget in cv[d]}
    prior_ok = [d for d in prior if d in cv and ok.get(d)]
    if not ok[primary]:
        verdict = "not reliable"
    elif not prior_ok:
        verdict = "gain in validity"
    elif min(rho[d] for d in prior_ok) >= PAYS:
        verdict = "pays off"
    else:
        verdict = "does not pay off"
    return {"query": query, "primary": primary, "budget": budget, "target": target, "rho": rho, "coverage_ok": ok, "verdict": verdict,
            "best_prior": min(prior_ok, key=lambda d: rho[d]) if prior_ok else None}


def describe(cv: dict, query: str, budget: int, read: dict) -> None:
    say(f"  {'design':8s} {'loss':>9s} {'labels needed':>14s} {'coverage mean':>14s} {'p10':>6s} {'labels':>7s}" +
        ("  wrong movies  decided  any wrong decided" if query == "Q5" else ""))
    for d in sorted(cv, key=lambda d: (d not in ("QP", "D"), d)):
        if budget not in cv[d]:
            continue
        r = cv[d][budget]
        t = r["by_truth"]["human"]
        extra = f"  {r['q5']['wrong_mean']:12.2f} {r['q5']['decided_mean']:8.1f} {r['q5']['any_wrong_decided']:17.3f}" if query == "Q5" else ""
        say(f"  {d:8s} {metric(r, query):9.4f} {read['rho'][d]:13.2f}x {t['coverage_mean']:14.3f} {t['coverage_p10']:6.3f} "
            f"{r['labels_mean']:7.0f}{extra}")


def primary_readings(cells: dict, results: dict) -> None:
    say("PRIMARY READINGS (D1 with Jev, D2 with Jev; labels needed = budget to reach the primary design's loss at B*, as a multiple of B*)")
    results["primary"] = {}
    for query in ("Q1",) + PRIMARY:
        pop = "D2" if query == "Q6" else "D1"
        cv = curves(cells, "primary", pop, "jev", query)
        budget = R.budgets_of("Q2" if query == "Q5" else query, pop)[1]
        primary = "D" if query == "Q5" else "QP"
        read = reading(cv, query, primary, budget, PRIOR)
        read["bias_aware"] = bias_aware(read, cv, budget) if query != "Q5" else None
        results["primary"][query] = read
        tag = "control" if query == "Q1" else read["verdict"]
        say(f"{query} (B* = {budget}): {tag}; best prior design meeting the coverage line: {read['best_prior']}"
            + (f"; pilot vs confidence: {read['bias_aware']}" if read["bias_aware"] else ""))
        describe(cv, query, budget, read)
    say()
    say("PER SHAPE")
    results["shapes"] = {}
    for shape, queries in SHAPES.items():
        verdicts = [results["primary"][q]["verdict"] for q in queries]
        results["shapes"][shape] = verdicts
        say(f"  {shape:16s} pays off on {sum(v == 'pays off' for v in verdicts)} of {len(verdicts)}; gain in validity on "
            f"{sum(v == 'gain in validity' for v in verdicts)}; " + ", ".join(f"{q}: {v}" for q, v in zip(queries, verdicts)))
    say()


def bias_aware(read: dict, cv: dict, budget: int) -> str:
    if "QC" not in cv:
        return ""
    if read["rho"]["QC"] >= BIAS_AWARE or (read["coverage_ok"]["QP"] and not read["coverage_ok"]["QC"]):
        return f"pilot pays off (QC needs {read['rho']['QC']:.2f}x)"
    return f"pilot does not pay off (QC needs {read['rho']['QC']:.2f}x)"


def secondary(cells: dict, results: dict) -> None:
    say("S1 KEV-4B AS THE CHEAP MODEL (and Jev on the same rows); reading as primary with prior designs U, C, G")
    results["S1"] = {}
    for pop, queries in (("D1-5k", ("Q1", "Q2", "Q3b", "Q4a", "Q4b-2")), ("D2-1k", ("Q6",))):
        for cheap in ("jev", "kev-4b"):
            for query in queries:
                cv = curves(cells, "S1", pop, cheap, query)
                budget = R.budgets_of(query, pop)[1]
                read = reading(cv, query, "QP", budget, ("U", "C", "G"))
                results["S1"][f"{pop}.{cheap}.{query}"] = read
                say(f"  {pop:6s} {cheap:7s} {query:6s} B*={budget:5d} {read['verdict']:18s} "
                    + " ".join(f"{d}:{read['rho'][d]:.2f}x{'' if read['coverage_ok'].get(d) else '(cov)'}" for d in ("U", "C", "G", "QU", "QC")))
    say()
    say("S2 PILOT SHARE (loss at B* relative to the 30% pilot)")
    results["S2"] = {}
    for query in R.D1_QUERIES + ("Q6",):
        pop = "D2" if query == "Q6" else "D1"
        budget = R.budgets_of(query, pop)[1]
        base = curves(cells, "primary", pop, "jev", query)
        row = {}
        for design in ("QP", "G"):
            for share in (0.15, 0.5):
                name = R.Cell("S2", pop, "jev", query, design, budget, pilot=share).name
                if name in cells:
                    row[f"{design}@{share:g}"] = cells[name]["by_truth"]["human"]["loss_mean"] / base[design][budget]["by_truth"]["human"]["loss_mean"]
        results["S2"][query] = row
        say(f"  {query:6s} " + "  ".join(f"{k}: {v:.2f}" for k, v in row.items()))
    say()
    say("S3 PER PUBLICATION (Q2b)")
    cv = curves(cells, "S3", "D1", "jev", "Q2b")
    budget = R.budgets_of("Q2b", "D1")[1]
    read = reading(cv, "Q2b", "QP", budget, PRIOR)
    results["S3"] = read
    say(f"  B* = {budget}: {read['verdict']}")
    describe(cv, "Q2b", budget, read)
    say()
    say("S4 A SECOND MODEL AS THE REFERENCE (its yes/no answers replace the human labels)")
    say("  population second   query design budget  coverage of its own answer  coverage of the human truth  outputs below 80%  line")
    results["S4"] = {}
    for r in sorted((r for r in cells.values() if r["cell"]["part"] == "S4"), key=lambda r: r["name"]):
        c = r["cell"]
        own, human = r["by_truth"]["second"], r["by_truth"]["human"]
        below = float(np.mean(np.array(human["coverage"]) < 0.80))
        largest = c["budget"] == R.budgets_of(c["query"], c["pop"])[0][-1] and c["design"] == "QP"  # the line is read on QP
        unreliable = largest and (below >= 1 / 3 if len(human["coverage"]) > 1 else human["coverage_mean"] < 0.80)
        results["S4"][r["name"]] = {"own": own["coverage_mean"], "human": human["coverage_mean"], "below80": below, "unreliable": unreliable}
        say(f"  {c['pop']:9s} {c['second']:15s} {c['query']:5s} {c['design']:4s} {c['budget']:6d} {own['coverage_mean']:27.3f} "
            f"{human['coverage_mean']:28.3f} {below:18.2f}  {'unreliable' if unreliable else ('reliable' if largest else '')}")
    say()
    say("S5 THREE TIERS (cost in second-model calls; human label = r calls; worth adding if QP's cost / three-tier cost >= 1.2 at r = 100)")
    results["S5"] = {}
    for pop, second, queries in (("D1", "luna", ("Q1", "Q2")), ("D1-5k", "kev-9b", ("Q1", "Q2")), ("D1-5k", "deepseek", ("Q1", "Q2")),
                                 ("D2-1k", "luna", ("Q6",)), ("D2-1k", "deepseek", ("Q6",))):
        for query in queries:
            part = "primary" if pop in ("D1", "D2") else "S1"
            qp = curves(cells, part, pop, "jev", query).get("QP", {})
            budget = R.budgets_of(query, pop)[1]
            if budget not in qp:
                continue
            target = metric(qp[budget], query)
            for m in (3.0, 10.0):
                runs = {r["cell"]["budget"]: r for r in cells.values() if r["cell"]["part"] == "S5" and r["cell"]["pop"] == pop
                        and r["cell"]["second"] == second and r["cell"]["query"] == query and r["cell"]["m"] == m}
                if not runs:
                    continue
                grid = sorted(runs)
                need = M.labels_needed(grid, [metric(runs[b], query) for b in grid], target)
                calls = float(np.interp(need, grid, [runs[b]["second_calls_mean"] for b in grid],
                                        left=runs[grid[0]]["second_calls_mean"] * need / grid[0],
                                        right=runs[grid[-1]]["second_calls_mean"] * need / grid[-1]))
                ratios = {price: price * budget / (price * need + calls) for price in (10, 100, 1000)}
                worth = ratios[100] >= 1.2
                results["S5"][f"{pop}.{second}.{query}.m{m:g}"] = {"human_needed": need, "calls": calls, "ratios": ratios, "worth": worth}
                say(f"  {pop:6s} {second:9s} {query:3s} m={m:<3g} human labels needed {need:7.0f} (QP: {budget}), calls {calls:7.0f}; "
                    f"cost ratio at r=10/100/1000: {ratios[10]:.2f}/{ratios[100]:.2f}/{ratios[1000]:.2f} -> "
                    f"{'worth adding' if worth else 'not worth adding'}")
    say()


def propagation(results: dict) -> None:
    say("PROPAGATION (no sampling): model-only answers; error per output against the human truth. Group queries: share of the squared")
    say("error the population's false-positive/negative rates predict, and the standardised remainder; scalars: error those rates predict")
    say("  population answer          query  outputs  median|err|  max|err|  share>h  predicted by global rates  |z|>1.96  var(z)")
    results["propagation"] = {}
    sets = (("D1", "jev", ("luna",)), ("D1-5k", "jev", ("kev-9b", "deepseek.yesno", "v4pro", "qwen")), ("D1-5k", "kev-4b", ()),
            ("D1-1.85k", "jev", ("opus",)), ("D2", "jev", ()), ("D2-1k", "jev", ("luna", "deepseek")), ("D2-1k", "kev-4b", ()),
            ("D2-546", "jev", ("opus",)))
    for pop_name, cheap, seconds in sets:
        pop, qs = R.population(pop_name, cheap)
        answers = {f"{cheap} count": (pop.p >= 0.5).astype(float), f"{cheap} mean p": pop.p}
        answers.update({s: (pop.second[s] >= 0.5).astype(float) for s in seconds})
        for qname, q in qs.items():
            if pop_name not in ("D1", "D2") and qname not in ("Q1", "Q2", "Q3b", "Q6"):
                continue
            theta = q.truth(pop.y)
            for label, yhat in answers.items():
                e = q.truth(yhat) - theta
                row = {"median": float(np.median(np.abs(e))), "max": float(np.abs(e).max()), "over_h": float(np.mean(np.abs(e) > q.h))}
                row.update(predicted=np.nan, z196=np.nan, varz=np.nan)
                if not label.endswith("mean p") and qname != "Q6":
                    y = pop.y
                    fpr = (yhat * (1 - y)).sum(0) / np.maximum((1 - y).sum(0), 1)
                    fnr = ((1 - yhat) * y).sum(0) / np.maximum(y.sum(0), 1)
                    eg = D.apply(q.A, fpr * (1 - y) - fnr * y)
                    if q.K == 1:  # scalar: the error the global rates predict (shown signed, next to the signed error)
                        row.update(error=float(e[0]), predicted=float(eg[0]))
                    else:
                        var = D.apply(q.A.multiply(q.A).tocsr(), fpr * (1 - fpr) * (1 - y) + fnr * (1 - fnr) * y)
                        d = e - eg
                        z = d[var > 0] / np.sqrt(var[var > 0])
                        row.update(predicted=float(1 - (d**2).sum() / (e**2).sum()) if (e**2).sum() > 0 else np.nan,
                                   z196=float(np.mean(np.abs(z) > 1.96)), varz=float(z.var()))
                results["propagation"][f"{pop_name}.{label}.{qname}"] = row
                predicted = (f"{row['error']:+.4f} vs {row['predicted']:+.4f}" if "error" in row else f"{row['predicted']:.2f}")
                say(f"  {pop_name:9s} {label:18s} {qname:6s} {q.K:7d} {row['median']:12.4f} {row['max']:9.4f} {row['over_h']:8.2f} "
                    f"{predicted:>26s} {row['z196']:9.2f} {row['varz']:7.2f}")
    say()
    say("CASCADES (the least confident 5/10/30% of rows answered by the second model; no human labels)")
    results["cascades"] = {}
    for pop_name, seconds in (("D1", ("luna",)), ("D1-5k", ("kev-9b", "deepseek.yesno")), ("D1-1.85k", ("opus",)),
                              ("D2-1k", ("luna", "deepseek")), ("D2-546", ("opus",))):
        pop, qs = R.population(pop_name, "jev")
        for s in seconds:
            for qname in (("Q6",) if pop_name.startswith("D2") else ("Q1", "Q2", "Q3b")):
                q = qs[qname]
                theta = q.truth(pop.y)
                out = []
                for share in (0.05, 0.10, 0.30):
                    yhat = (pop.p >= 0.5).astype(float)
                    for c in range(pop.C):
                        routed = np.argsort(np.abs(pop.p[:, c] - 0.5), kind="stable")[: int(round(share * pop.N))]
                        yhat[routed, c] = (pop.second[s][routed, c] >= 0.5)
                    e = q.truth(yhat) - theta
                    out.append({"share": share, "median": float(np.median(np.abs(e))), "over_h": float(np.mean(np.abs(e) > q.h)),
                                "mean_error": float(e.mean())})
                results["cascades"][f"{pop_name}.{s}.{qname}"] = out
                say(f"  {pop_name:9s} {s:15s} {qname:4s} " + "  ".join(
                    f"{o['share']:.0%}: median|err| {o['median']:.4f}, share>h {o['over_h']:.2f}, mean err {o['mean_error']:+.4f}" for o in out))
    say()


def heterogeneity(r: np.ndarray, member: list, rng: np.random.Generator, shuffles: int = 1000) -> dict:
    """Between-group spread of residuals: Q = sum_g n_g (mean_g - mean)^2 (rows counted in each of their groups), permutation p-value,
    random-effects SD (DerSimonian-Laird), intraclass correlation (one-way ANOVA, single-membership groupings only)."""
    rows = np.concatenate([np.full(len(g), i) for i, g in enumerate(member)])
    codes = np.concatenate([np.asarray(g, np.int64) for g in member])
    G = codes.max() + 1
    n = np.bincount(codes, minlength=G)

    def stat(values):
        s = np.bincount(codes, weights=values[rows], minlength=G)
        m = s[n > 0] / n[n > 0]
        return float((n[n > 0] * (m - values.mean()) ** 2).sum())

    observed = stat(r)
    perm = np.array([stat(rng.permutation(r)) for _ in range(shuffles)])
    s = np.bincount(codes, weights=r[rows], minlength=G)
    keep = n >= 2
    means = s[keep] / n[keep]
    within = np.bincount(codes, weights=r[rows] ** 2, minlength=G)[keep] - n[keep] * means**2
    s2 = within.sum() / max(n[keep].sum() - keep.sum(), 1)
    w = n[keep] / s2
    mw = (w * means).sum() / w.sum()
    qdl = (w * (means - mw) ** 2).sum()
    tau2 = max(0.0, (qdl - (keep.sum() - 1)) / (w.sum() - (w**2).sum() / w.sum()))
    out = {"groups": int((n > 0).sum()), "p_value": float((1 + (perm >= observed).sum()) / (1 + shuffles)), "tau": float(np.sqrt(tau2))}
    if len(rows) == len(r):
        msb = (n[keep] * (means - r.mean()) ** 2).sum() / max(keep.sum() - 1, 1)
        n0 = (n[keep].sum() - (n[keep] ** 2).sum() / n[keep].sum()) / max(keep.sum() - 1, 1)
        out["icc"] = float((msb - s2) / (msb + (n0 - 1) * s2))
    return out


def bias(results: dict) -> None:
    say("BIAS BY GROUP (residuals after one population-wide calibration; heterogeneous if p < .01)")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([D.SEED, 7])))
    results["bias"] = {}
    pop, qs = R.population("D1", "jev")
    alpha, beta = M.calibrate(pop.p, pop.y, np.ones(pop.N))
    r = (pop.y - (alpha + beta * pop.p))[:, 0]
    genres = sorted({g for gs in pop.cols["genres"] for g in gs})
    gi = {g: k for k, g in enumerate(genres)}
    groupings = {"movie": D._codes(pop.cols["movie"]), "critic": D._codes(pop.cols["critic"]), "publication": D._codes(pop.cols["publication"]),
                 "genre": [[gi[g] for g in gs] for gs in pop.cols["genres"]], "top critic": pop.cols["top"],
                 "length tertile": pop.cols["length"], "score bin": pop.bins[:, 0]}
    for name, codes in groupings.items():
        member = codes if isinstance(codes, list) else [[c] for c in codes]
        h = heterogeneity(r, member, rng)
        results["bias"][f"D1.{name}"] = h
        say(f"  D1 {name:15s} groups {h['groups']:5d}  permutation p {h['p_value']:.4f}  random-effects SD {h['tau']:.4f}"
            + (f"  ICC {h['icc']:.4f}" if "icc" in h else "") + ("  heterogeneous" if h["p_value"] < 0.01 else ""))
    wrong = ((pop.p >= 0.5) != (pop.y == 1))[:, 0]
    confident = ((pop.p <= 0.1) | (pop.p >= 0.9))[:, 0]
    results["bias"]["D1.confident_errors"] = float(confident[wrong].mean())
    say(f"  D1 share of Jev's errors with p <= .1 or >= .9: {confident[wrong].mean():.3f} (of rows: {confident.mean():.3f})")
    for g in genres:
        rows = np.array([g in gs for gs in pop.cols["genres"]])
        results["bias"][f"D1.confident_errors.{g}"] = float(confident[rows & wrong].mean()) if (rows & wrong).any() else np.nan
    pop2, _ = R.population("D2", "jev")
    alpha, beta = M.calibrate(pop2.p, pop2.y, np.ones(pop2.N))
    for c, name in enumerate(pop2.predicates):
        r2 = pop2.y[:, c] - (alpha[c] + beta[c] * pop2.p[:, c])
        h = heterogeneity(r2, [[t] for t in np.searchsorted(np.quantile(pop2.cols["length"], [1 / 3, 2 / 3]), pop2.cols["length"], side="right")],
                          rng)
        wrong2 = (pop2.p[:, c] >= 0.5) != (pop2.y[:, c] == 1)
        conf2 = (pop2.p[:, c] <= 0.1) | (pop2.p[:, c] >= 0.9)
        row = {"count_bias": float((pop2.p[:, c] >= 0.5).mean() - pop2.y[:, c].mean()),
               "confident_errors": float(conf2[wrong2].mean()) if wrong2.any() else np.nan, "length": h}
        results["bias"][f"D2.{name}"] = row
        say(f"  D2 {name:18s} count bias {row['count_bias']:+.3f}  confident share of errors {row['confident_errors']:.3f}  "
            f"length tertiles p {h['p_value']:.4f}")
    say()
    say("SHARED ERRORS (when Jev is wrong, how often the second model is wrong too, against its overall error rate)")
    results["shared"] = {}
    for pop_name, seconds in (("D1", ("luna",)), ("D1-5k", ("kev-9b", "deepseek.yesno", "v4pro", "qwen", "kev-4b")), ("D1-1.85k", ("opus",)),
                              ("D2-1k", ("luna", "deepseek", "kev-4b")), ("D2-546", ("opus",))):
        pop, _ = R.population(pop_name, "jev")
        jev_wrong = (pop.p >= 0.5) != (pop.y == 1)
        for s in seconds:
            s_wrong = (pop.second[s] >= 0.5) != (pop.y == 1)
            row = {"given": float(s_wrong[jev_wrong].mean()), "marginal": float(s_wrong.mean())}
            results["shared"][f"{pop_name}.{s}"] = row
            say(f"  {pop_name:9s} {s:15s} P(wrong | Jev wrong) {row['given']:.3f}   P(wrong) {row['marginal']:.3f}")
    say()


def main() -> None:
    cells = load()
    expected = {c.name for c in R.cells()}
    missing = expected - set(cells)
    say(f"Phase 16 report ({'REHEARSAL, simulated labels' if D.REHEARSAL else 'formal run'}): {len(cells)} cells read, "
        f"{len(missing)} of {len(expected)} missing")
    say()
    results: dict = {"missing": sorted(missing)}
    if missing:
        say("Missing cells; the readings below cover only what exists.")
    primary_readings(cells, results)
    secondary(cells, results)
    propagation(results)
    bias(results)
    D.RESULTS.mkdir(parents=True, exist_ok=True)
    (D.RESULTS / "report.txt").write_text("\n".join(lines) + "\n")
    (D.RESULTS / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
