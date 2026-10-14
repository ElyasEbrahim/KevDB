"""Phase 3: semantic aggregation. Estimate "what share of reviews is positive" against human labels.

Primary analysis and success criteria were written before the run, see experiments/database/protocol/p3.txt. This only replays the judgements collected in phase 2 and calls no model.

Population: the distinct reviews in the 16,000-row version of the movie data (15,388). Truth: scoreSentiment = POSITIVE.

Without labels:
    Jev threshold   share with p >= 0.5          Jev probability   mean of p          LLM full   share the LLM judges positive
With n human labels (simple random sampling without replacement):
    sample only     share positive among the n
    difference      population mean of p + mean of (label - p) over the n                ← primary analysis
    regression      the same, with p scaled by a coefficient estimated from the n (more robust when p is not calibrated)
    stratified      5 strata by p, more samples from the uncertain strata

Intervals are always estimate ± 1.96 × standard error, with the finite-population correction. Each setting is drawn 2,000 times.

    python experiments/database/p3_aggregates.py
"""

from __future__ import annotations

import json

import duckdb
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression

from paths import DATA, OUT as RESULTS

P2, OUT = RESULTS / "p2", RESULTS / "p3"
BIG = f"{DATA}/sembench/files/movie/data/sf_16000/Reviews.csv"
REPS, SEED, Z = 2000, 20261007, 1.96
BUDGETS = (50, 100, 200, 500, 1000)
OPEN = ("v41flash.off", "qwen38flash.off", "v4pro.off")


def load() -> dict[str, np.ndarray]:
    """One row per distinct review: truth, movie, and each source's judgement. The open models were only asked the sampled 5,000 texts; the rest are NaN."""
    con = duckdb.connect()
    con.execute(f"""CREATE TABLE r AS SELECT DISTINCT id, reviewId, isTopCritic, (scoreSentiment = 'POSITIVE')::INT AS y,
                    substr(sha256(reviewText), 1, 16) AS text FROM read_csv('{BIG}')""")
    cols = {"p": "jev.positive", "pc": "jev.clearly", "llm": "luna.low.positive"}
    select, joins = ["r.id", "r.y", "r.isTopCritic::INT AS top", "r.text"], []
    for alias, name in cols.items():
        joins.append(f"JOIN read_json_auto('{P2}/{name}.jsonl') AS {alias} ON {alias}.text = r.text")
        select.append(f"{alias}.p AS {alias}")
    for i, name in enumerate(OPEN):
        joins.append(f"LEFT JOIN read_json_auto('{P2}/{name}.positive.jsonl') AS o{i} ON o{i}.text = r.text")
        select += [f'o{i}.p_true AS "{name}"', f'o{i}.p AS "{name}.hard"']
    df = con.execute(f"SELECT {', '.join(select)} FROM r {' '.join(joins)} ORDER BY r.id, r.reviewId").fetchdf()
    assert len(df) == con.execute("SELECT count(*) FROM r").fetchone()[0] and not df[["p", "pc", "llm"]].isna().any().any()
    return {c: df[c].to_numpy() for c in df.columns}


def auc(score: np.ndarray, y: np.ndarray) -> float:
    ranks, pos = rankdata(score), y == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    which = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(p[which == b].mean() - y[which == b].mean()) * (which == b).mean() for b in range(bins) if (which == b).any())


def simulate(y: np.ndarray, proxies: dict[str, np.ndarray], n: int, rng: np.random.Generator, strata_on: str | None = None) -> dict[str, dict]:
    """Draw REPS samples of n labels; return each method's error, interval coverage and interval half-width."""
    N, truth = len(y), y.mean()
    fpc = 1 - n / N
    est = {k: np.empty(REPS) for k in ["sample"] + [f"{kind}:{name}" for name in proxies for kind in ("diff", "reg")]}
    se = {k: np.empty(REPS) for k in est}
    means = {name: f.mean() for name, f in proxies.items()}
    for r in range(REPS):
        s = rng.choice(N, n, replace=False)
        ys = y[s]
        est["sample"][r], se["sample"][r] = ys.mean(), np.sqrt(ys.var(ddof=1) / n * fpc)
        for name, f in proxies.items():
            fs = f[s]
            d = ys - fs
            est[f"diff:{name}"][r], se[f"diff:{name}"][r] = means[name] + d.mean(), np.sqrt(d.var(ddof=1) / n * fpc)
            vf = fs.var(ddof=1)
            b = np.cov(ys, fs, ddof=1)[0, 1] / vf if vf > 0 else 0.0
            resid = ys - ys.mean() - b * (fs - fs.mean())
            est[f"reg:{name}"][r] = ys.mean() + b * (means[name] - fs.mean())
            se[f"reg:{name}"][r] = np.sqrt((resid**2).sum() / (n - 2) / n * fpc)
    if strata_on:
        # The stratified sample uses its own random stream, so it does not affect the samples drawn by the methods above
        est["strat"], se["strat"] = stratified(y, proxies[strata_on], n, np.random.default_rng(SEED + n))
    out = {}
    base = ((est["sample"] - truth) ** 2).mean()
    for k in est:
        err = est[k] - truth
        out[k] = {"rmse_pp": 100 * np.sqrt((err**2).mean()), "bias_pp": 100 * err.mean(), "coverage": (np.abs(err) <= Z * se[k]).mean(),
                  "half_width_pp": 100 * Z * se[k].mean(), "efficiency": base / (err**2).mean()}
    return out


def stratified(y: np.ndarray, p: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """5 strata by p; each stratum gets samples proportional to stratum size × sqrt(p̄(1-p̄)) (at least 2)."""
    which = np.digitize(p, [0.1, 0.3, 0.7, 0.9])
    layers = [np.flatnonzero(which == h) for h in range(5)]
    layers = [ix for ix in layers if len(ix)]
    weight = np.array([len(ix) for ix in layers]) / len(y)
    spread = np.array([np.sqrt(max(p[ix].mean() * (1 - p[ix].mean()), 1e-4)) for ix in layers])
    share = weight * spread / (weight * spread).sum()
    alloc = np.maximum(2, np.floor(n * share).astype(int))
    for h in np.argsort(-(n * share - np.floor(n * share))):  # remaining slots go to the largest fractional parts, so the total is exactly n
        if alloc.sum() >= n:
            break
        alloc[h] += 1
    assert alloc.sum() == n and all(k <= len(ix) for k, ix in zip(alloc, layers))
    est, se = np.empty(REPS), np.empty(REPS)
    for r in range(REPS):
        m = v = 0.0
        for ix, w, k in zip(layers, weight, alloc):
            ys = y[rng.choice(ix, k, replace=False)]
            m += w * ys.mean()
            v += w**2 * ys.var(ddof=1) / k * (1 - k / len(ix))
        est[r], se[r] = m, np.sqrt(v)
    return est, se


def show(title: str, res: dict[int, dict[str, dict]], keys: dict[str, str]) -> None:
    print(f"\n{title}")
    print(f"{'method':22s}" + "".join(f"{'n=' + str(n):>26s}" for n in res))
    print(f"{'':22s}" + "".join(f"{'error / coverage / labels saved':>26s}" for _ in res))
    for key, label in keys.items():
        print(f"{label:22s}" + "".join(
            f"{r[key]['rmse_pp']:>10.2f} / {r[key]['coverage']:>5.1%} / {r[key]['efficiency']:>4.1f}x" if key in r else f"{'—':>26s}" for r in res.values()))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = load()
    y, p, pc, llm = t["y"], t["p"], t["pc"], t["llm"]
    N, truth = len(y), y.mean()
    results: dict = {"N": N, "truth": truth}
    print(f"population {N} reviews, {len(set(t['id']))} movies, {truth:.2%} labelled positive by humans")

    # Is Jev consistent with itself?
    con = duckdb.connect()
    a, b = zip(*con.execute(f"SELECT a.p, b.p FROM read_json_auto('{P2}/jev.positive.jsonl') a JOIN read_json_auto('{P2}/jev.repeat.positive.jsonl') b USING (text)").fetchall())
    a, b = np.array(a), np.array(b)
    results["repeat"] = {"n": len(a), "same_decision": ((a >= 0.5) == (b >= 0.5)).mean(), "mean_abs_diff": np.abs(a - b).mean(), "max_abs_diff": np.abs(a - b).max()}
    print(f"Jev asked the same {len(a)} reviews twice: same decision {results['repeat']['same_decision']:.2%}, mean probability difference {results['repeat']['mean_abs_diff']:.4f}, max {results['repeat']['max_abs_diff']:.2f}")

    hard = (p >= 0.5).astype(float)
    nolabel = {"Jev threshold (positive)": hard.mean(), "Jev probability (positive)": p.mean(), "LLM full": llm.mean(),
               "Jev threshold (clearly positive)": (pc >= 0.5).mean(), "Jev probability (clearly positive)": pc.mean()}
    print("\nwithout labels: estimate, and difference from the truth (points)")
    for name, v in nolabel.items():
        print(f"  {name:20s}{v:8.2%}{100 * (v - truth):+8.2f}")
    results["no_labels"] = {k: {"estimate": v, "error_pp": 100 * (v - truth)} for k, v in nolabel.items()}
    print(f"  per-row agreement with human labels: Jev {((p >= 0.5) == y).mean():.2%}, LLM {(llm == y).mean():.2%}; AUC: Jev positive {auc(p, y):.3f}, Jev clearly positive {auc(pc, y):.3f}")

    rng = np.random.default_rng(SEED)
    proxies = {"jev": p, "jev_hard": hard, "llm": llm.astype(float), "clearly": pc}
    keys = {"sample": "sample only", "diff:jev": "difference (primary)", "reg:jev": "regression", "strat": "stratified",
            "diff:jev_hard": "difference, Jev threshold", "diff:llm": "difference, LLM full", "reg:llm": "regression, LLM full",
            "diff:clearly": "difference, \"clearly positive\"", "reg:clearly": "regression, \"clearly positive\""}
    whole = {n: simulate(y, proxies, n, rng, strata_on="jev") for n in BUDGETS}
    show("population: with n labels, error (RMSE, points) / 95% interval coverage / labels saved relative to sample only", whole, keys)
    results["whole"] = whole
    primary = whole[200]["diff:jev"]
    verdict = primary["efficiency"] >= 3 and primary["coverage"] >= 0.90
    print(f"\nprimary analysis (n=200): saves {primary['efficiency']:.2f}x the labels (target ≥3), interval coverage {primary['coverage']:.1%} (target ≥90%) → {'met' if verdict else 'not met'}")
    results["primary"] = {**primary, "passed": bool(verdict)}

    # Subsets selected by relational conditions: top critic or not; and each of the 27 movies with at least 200 reviews separately
    subsets = {"top critics": t["top"] == 1, "other critics": t["top"] == 0}
    sub = {name: simulate(y[m], {"jev": p[m]}, 200, rng) for name, m in subsets.items()}
    ids, counts = np.unique(t["id"], return_counts=True)
    big = [i for i, c in zip(ids, counts) if c >= 200]
    per_movie = [simulate(y[t["id"] == i], {"jev": p[t["id"] == i]}, 50, rng) for i in big]
    avg = lambda key, field: float(np.mean([r[key][field] for r in per_movie]))
    print("\nsubsets (difference vs sample only)")
    for name, r in sub.items():
        print(f"  {name:15s}{subsets[name].sum():5d} reviews, n=200: saves {r['diff:jev']['efficiency']:.2f}x the labels, coverage {r['diff:jev']['coverage']:.1%} (sample only {r['sample']['coverage']:.1%})")
    print(f"  {len(big)} large movies with n=50 each, averaged: saves {avg('diff:jev', 'efficiency'):.2f}x (regression {avg('reg:jev', 'efficiency'):.2f}), "
          f"coverage {avg('diff:jev', 'coverage'):.1%} (sample only {avg('sample', 'coverage'):.1%}, regression {avg('reg:jev', 'coverage'):.1%})")
    results["subsets"] = {"top_critic": sub, "movies_n50_mean": {k: {f: avg(k, f) for f in ("efficiency", "coverage", "rmse_pp")} for k in ("sample", "diff:jev", "reg:jev")}}

    # Grouped by movie (movies with at least 30 reviews): each group's share of positive reviews
    groups = [(i, t["id"] == i) for i, c in zip(ids, counts) if c >= 30]
    true_g = np.array([y[m].mean() for _, m in groups])
    size_g = np.array([m.sum() for _, m in groups])
    logit = lambda q: np.log(np.clip(q, 0.005, 0.995) / (1 - np.clip(q, 0.005, 0.995)))

    def by_group(values: np.ndarray) -> dict[str, float]:
        g = np.array([values[m].mean() for _, m in groups])
        return {"mae_pp": 100 * np.abs(g - true_g).mean(), "p90_pp": 100 * np.quantile(np.abs(g - true_g), 0.9), "spearman": spearmanr(g, true_g).correlation}

    grouped = {"Jev threshold": by_group(hard), "Jev probability": by_group(p), "LLM full": by_group(llm.astype(float))}
    for n in (200, 500):
        runs = []
        for _ in range(200):
            s = rng.choice(N, n, replace=False)
            model = LogisticRegression(C=1e6).fit(logit(p[s]).reshape(-1, 1), y[s])
            runs.append(by_group(model.predict_proba(logit(p).reshape(-1, 1))[:, 1]))
        grouped[f"Jev probability + {n} population labels for calibration"] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    # Reference: spread the same number of labels across the groups and use only each group's own labels
    for n in (200, 500):
        k = max(1, n // len(groups))
        runs = [np.abs(np.array([y[np.flatnonzero(m)[rng.choice(m.sum(), min(k, m.sum()), replace=False)]].mean() for _, m in groups]) - true_g) for _ in range(200)]
        grouped[f"sample only, {k} per group ({k * len(groups)} in total)"] = {"mae_pp": 100 * float(np.mean(runs)), "p90_pp": 100 * float(np.mean([np.quantile(r, 0.9) for r in runs])), "spearman": float("nan")}
    print(f"\ngrouped by movie ({len(groups)} movies, {size_g.min()}-{size_g.max()} reviews each): share positive per group, mean absolute error / 90th percentile error (points) / rank correlation")
    for name, r in grouped.items():
        print(f"  {name:32s}{r['mae_pp']:7.2f}{r['p90_pp']:8.2f}{r['spearman']:8.3f}")
    results["grouped"] = grouped

    # Without human labels: use the LLM as the labeller; the target is "the share a full LLM pass would give"
    target = llm.astype(float)
    as_oracle = {n: simulate(target, {"jev": p}, n, rng) for n in (200, 500)}
    print(f"\nthe LLM as labeller (target: the full LLM result {target.mean():.2%}, itself {100 * (target.mean() - truth):+.2f} points from the human truth)")
    for n, r in as_oracle.items():
        print(f"  {n} LLM calls: difference-estimator error {r['diff:jev']['rmse_pp']:.2f} points (sample only {r['sample']['rmse_pp']:.2f}), {r['diff:jev']['efficiency']:.1f}x fewer calls, coverage {r['diff:jev']['coverage']:.1%}")
    results["llm_as_oracle"] = as_oracle

    # Other sources as the "full judgement", on the reviews behind the 5,000 texts the open models were asked (the attribution check of phase 5)
    m = ~np.isnan(t[OPEN[0]])
    ys = y[m]
    sources = {"Jev probability": p[m], "Jev threshold": hard[m], "LLM (Luna) yes or no": llm[m].astype(float)}
    for name in OPEN:
        sources[f"{name} probability"] = t[name][m]
        sources[f"{name} yes or no"] = t[f"{name}.hard"][m]
    runs = simulate(ys, sources, 200, rng)
    cost = {name: json.loads((P2 / f"{name}.positive.run.json").read_text()) for name in ("jev", "luna.low") + OPEN}
    print(f"\nother sources as the full judgement ({m.sum()} reviews, n=200): per-row agreement / AUC / calibration error / labels saved (difference) / labels saved (regression)")
    table = {}
    for name, f in sources.items():
        soft = len(np.unique(f)) > 2
        table[name] = {"accuracy": ((f >= 0.5) == ys).mean(), "auc": auc(f, ys) if soft else float("nan"), "ece": ece(f, ys) if soft else float("nan"),
                       "diff": runs[f"diff:{name}"]["efficiency"], "reg": runs[f"reg:{name}"]["efficiency"], "share_extreme": ((f < 0.02) | (f > 0.98)).mean()}
        r = table[name]
        print(f"  {name:26s}{r['accuracy']:8.2%}{r['auc']:8.3f}{r['ece']:8.3f}{r['diff']:7.2f}x{r['reg']:7.2f}x   share of probabilities near 0 or 1: {r['share_extreme']:.0%}")
    print("  cost per 1,000 rows / median latency: " + "; ".join(f"{k} ${1000 * v['cost_usd'] / v['rows']:.3f} / {v['latency_p50']:.2f}s" for k, v in cost.items()))
    results["sources"] = {"rows": int(m.sum()), "table": table, "cost": {k: {"per_1000": 1000 * v["cost_usd"] / v["rows"], "latency_p50": v["latency_p50"]} for k, v in cost.items()}}

    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
