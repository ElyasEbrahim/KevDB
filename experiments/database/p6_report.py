"""Phase 6 analysis: can open, Jev-compatible models replace Jev as the default backend? Reads the collected judgements only and calls no model.

Questions, metrics and pass criteria are in experiments/database/protocol/p6.txt, all written before collection. Jev's and DeepSeek's judgements come from the outputs of earlier phases.

    python experiments/database/p6_report.py | tee out/database/p6/report.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata

from kevdb.cache import Cache, cache_key
from kevdb.client import USD_PER_INPUT_TOKEN

from kevdb import JEV_MODEL
from p3_aggregates import ece, load
from p4_collect import PREDICATES, sentences
from p5_report import SHARES, concentration
from p6_collect import EM, JEV, JEV_NAMESPACE, N_MAMS, SOURCES, requests
from paths import OUT as RESULTS, ROOT

OUT, P2, P4 = RESULTS / "p6", RESULTS / "p2", RESULTS / "p4"
BOOT, SEED, DRAWS, LABELS = 2000, 20261009, 200, 200
# Criteria fixed beforehand: difference from Jev; a plus sign means "not lower than", a minus sign "not higher than"
BARS = {"acc": ("accuracy", -0.02, +1), "auc": ("AUC", -0.02, +1), "brier": ("Brier", +0.02, -1)}
ECE_BAR, MISSING_BAR, SPEED_BAR = 0.02, 0.05, 0.10
NAMES = {"jev": "Jev 1.13.0 (cached)", JEV: "Jev 1.13.0 (asked today)", "v41flash.off": "DeepSeek V4.1 Flash logprob", "kev-4b": "Kev-4B", "kev-9b": "Kev-9B",
         "laya": "Laya English (0-shot)", "laya-multilingual": "Laya multilingual (0-shot)", "clef-flash": "Clef-flash", "no-skill": "No-skill (base rate)"}
# The four datasets used for the verdicts and tables; movie.repeat is only for the ask-again check
TABLES = ("movie", "mams", "mams.packed", "em")
TITLES = {"movie": "Movie reviews \"positive\"", "mams": "Restaurant reviews, 21 predicates one by one", "mams.packed": "Restaurant reviews, 21 questions packed", "em": "Product matching Walmart-Amazon"}
# The context window (tokens) each model was validated on, from their own documentation
WINDOW = {"kev-4b": 8192, "kev-9b": 8192, "laya": 512, "laya-multilingual": 1024, "clef-flash": 16384, JEV: 32768}
# laya uses one window per question and reports the token count summed over questions, so comparing it with the window is meaningless: its own truncation flag tells whether it overflowed
PER_QUESTION_WINDOW = ("laya", "laya-multilingual")
STATS = ("acc", "auc", "brier", "ece_mass")
# The field used as the row id in each dataset's collected results
KEY = {"movie": "text", "mams": "sid", "mams.packed": "sid", "em": "id"}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# ---------- Loading: each dataset is a row × question matrix with NaN where there is no answer; the row ids are returned too ----------

def movie(sources: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray], list]:
    t = load()
    keep = np.isin(t["text"], [r["text"] for r in requests("movie")[0]])
    soft, hard = t["v41flash.off"][keep].astype(float), t["v41flash.off.hard"][keep].astype(float)
    P = {"jev": t["p"][keep].astype(float), "v41flash.off": np.where(np.isnan(soft), hard, soft)}
    for s in sources:
        got = {r["text"]: r["p"] for r in read(OUT / f"movie.{s}.jsonl")}
        P[s] = np.array([np.nan if got.get(x) is None else got[x] for x in t["text"][keep]])
    return t["y"][keep][:, None], {k: v[:, None] for k, v in P.items()}, list(t["text"][keep])


def mams(sources: list[str], packed: bool = False) -> tuple[np.ndarray, dict[str, np.ndarray], list]:
    rows, names = sentences(N_MAMS), list(PREDICATES)

    def matrix(path: Path, field: str = "p", fallback: str | None = None) -> np.ndarray:
        got = {r["sid"]: r for r in read(path)}
        out = np.full((len(rows), len(names)), np.nan)
        for i, r in enumerate(rows):
            rec = got.get(r["sid"]) or {}
            first, second = rec.get(field) or {}, (rec.get(fallback) or {}) if fallback else {}
            for j, k in enumerate(names):
                v = first.get(k)
                v = second.get(k) if v is None else v
                if v is not None:
                    out[i, j] = v
        return out

    tag = "mams.packed" if packed else "mams"
    P = {"jev": matrix(P4 / ("mams.jev.packed.jsonl" if packed else "mams.jev.jsonl"))}
    if not packed:  # output probabilities are only available when asking one question at a time; the missing cells use its yes or no
        P["v41flash.off"] = matrix(P4 / "mams.v41flash.off.jsonl", "p_true", "p")
    for s in sources:
        P[s] = matrix(OUT / f"{tag}.{s}.jsonl")
    return np.array([[r["truth"][k] for k in names] for r in rows]), P, [r["sid"] for r in rows]


def em(sources: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray], list]:
    full = read(ROOT / f"out/entity_match/{EM}.test.{JEV_MODEL}.jsonl")
    P = {"jev": np.array([r["p"] for r in full], float)}
    for s in sources:
        got = {r["id"]: r["p"] for r in read(OUT / f"em.{s}.jsonl")}
        P[s] = np.array([np.nan if got.get(r["id"]) is None else got[r["id"]] for r in full])
    return np.array([r["label"] for r in full])[:, None], {k: v[:, None] for k, v in P.items()}, [r["id"] for r in full]


LOADERS = {"movie": movie, "mams": mams, "mams.packed": lambda sources: mams(sources, packed=True), "em": em}


# ---------- Metrics ----------

def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos = y == 1
    if pos.all() or not pos.any():
        return np.nan
    return (rankdata(score)[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def ece_mass(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Equal-mass binning: after sorting by probability, each bin has the same number of rows. With equal-width bins almost every row lands in the two end bins and the middle ones hardly count."""
    return sum(abs(p[ix].mean() - y[ix].mean()) * len(ix) for ix in np.array_split(np.argsort(p, kind="stable"), bins)) / len(p)


def filled(P: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(P), 0.5, P)


def correct(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Cells without an answer count as wrong."""
    return ((filled(P) >= 0.5) == (Y == 1)) & ~np.isnan(P)


def four(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """The four numbers used for the verdicts, computed per question and then averaged over questions."""
    Q, C = filled(P), correct(P, Y)
    per = [(C[:, j].mean(), auc(Q[:, j], Y[:, j]), ((Q[:, j] - Y[:, j]) ** 2).mean(), ece_mass(Q[:, j], Y[:, j])) for j in range(Y.shape[1])]
    return np.nanmean(np.array(per), axis=0)


def describe(P: np.ndarray, Y: np.ndarray) -> dict:
    Q, C = filled(P), correct(P, Y)
    out = dict(zip(STATS, map(float, four(P, Y))))
    out["ece"] = float(np.mean([ece(Q[:, j], Y[:, j]) for j in range(Y.shape[1])]))
    conc = [list(concentration(~C[:, j], np.argsort(np.abs(Q[:, j] - 0.5), kind="stable")).values()) for j in range(Y.shape[1]) if not C[:, j].all()]
    out["concentration"] = dict(zip(map(str, SHARES), map(float, np.mean(conc, axis=0)))) if conc else {}
    # The middle band and the reliability table pool all cells: a single predicate may have only a few rows in the middle band
    q, y, c = Q.ravel(), Y.ravel() == 1, C.ravel()
    mid = (q >= 0.1) & (q <= 0.9)
    out["missing"], out["cells"] = int(np.isnan(P).sum()), int(P.size)
    out["middle"] = {"n": int(mid.sum()), "share": float(mid.mean()), "mean_p": float(q[mid].mean()) if mid.any() else None,
                     "positive": float(y[mid].mean()) if mid.any() else None, "accuracy": float(c[mid].mean()) if mid.any() else None}
    which = np.minimum((q * 10).astype(int), 9)
    out["reliability"] = [{"n": int((which == b).sum()), "mean_p": float(q[which == b].mean()), "positive": float(y[which == b].mean())} if (which == b).any()
                          else {"n": 0} for b in range(10)]
    return out


def bootstrap(P: dict[str, np.ndarray], Y: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Each source's four numbers on the same bootstrap resamples, so differences between sources are paired."""
    index = rng.integers(0, len(Y), (BOOT, len(Y)))
    return {s: np.array([four(p[ix], Y[ix]) for ix in index]) for s, p in P.items()}


def platt(p: np.ndarray, y: np.ndarray):
    """Fit a + b·logit(p) on human labels. Targets are pulled in slightly as Platt suggests, so the fit does not diverge when all labels are of one class."""
    x, n1 = logit(np.clip(p, 1e-3, 1 - 1e-3)), (y == 1).sum()
    target = np.where(y == 1, (n1 + 1) / (n1 + 2), 1 / (len(y) - n1 + 2))
    a, b = minimize(lambda w: (np.logaddexp(0, w[0] + w[1] * x) - target * (w[0] + w[1] * x)).sum(), [0.0, 1.0], method="BFGS").x
    return lambda q: expit(a + b * logit(np.clip(q, 1e-3, 1 - 1e-3)))


def recalibrated(P: np.ndarray, Y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fit on LABELS human-labelled rows (one parameter set shared by all predicates), compute the four numbers on the remaining rows, repeat DRAWS times and average."""
    Q, out = filled(P), []
    for _ in range(DRAWS):
        fit = np.zeros(len(Y), bool)
        fit[rng.choice(len(Y), LABELS, replace=False)] = True
        out.append(four(platt(Q[fit].ravel(), Y[fit].ravel())(Q[~fit]), Y[~fit]))
    return np.mean(out, axis=0)


# ---------- Throughput and cost ----------

def jev_speed() -> dict[str, float]:
    """Requests per second Jev completed when it ran these datasets cold earlier (concurrency 16, official API)."""
    run = lambda path: json.loads(path.read_text().splitlines()[0])
    movie_run, single, packed = run(P2 / "jev.positive.run.json"), run(P4 / "mams.jev.run.json"), run(P4 / "mams.jev.packed.run.json")
    pairs = run(ROOT / f"out/entity_match/{EM}.test.{JEV_MODEL}.runs.jsonl")
    return {"movie": movie_run["rows"] / movie_run["seconds"], "mams": single["requests"] / single["seconds"],
            "mams.packed": packed["requests"] / packed["seconds"], "em": pairs["pairs"] / pairs["seconds"]}


def jev_cost_per_1000(dataset: str) -> float:
    cache = Cache(ROOT / ".jevcache.sqlite")
    ids, reqs = requests(dataset)
    tokens = sum(cache.get(cache_key(JEV_NAMESPACE[dataset] + JEV_MODEL, state, questions))["usage"]["input_tokens"] for state, questions in reqs)
    return tokens * USD_PER_INPUT_TOKEN / len(ids) * 1000


def over_window(source: str, dataset: str) -> set:
    """Rows that exceed this model's window: those the model itself reports as truncated, or (for models with one window per request) requests whose token count exceeds the window."""
    rows = read(OUT / f"{dataset}.{source}.jsonl")
    shared = source not in PER_QUESTION_WINDOW
    return {r[KEY[dataset]] for r in rows if r.get("truncated") or (shared and r.get("request_tokens", 0) > WINDOW[source])}


def runs(source: str, dataset: str) -> dict | None:
    path = OUT / f"{dataset}.{source}.runs.jsonl"
    if not path.exists():
        return None
    log = read(path)
    cold = [r for r in log if r["fresh"]]
    seconds = sum(r["seconds"] for r in cold)
    return {"requests_per_second": sum(r["fresh"] for r in cold) / seconds if seconds else None, "concurrency": log[-1]["concurrency"],
            "cost_per_1000": log[-1]["cost_usd"] / log[-1]["rows"] * 1000, "errors": log[-1]["errors"], "requests": log[-1]["requests"], "passes": len(log)}


# ---------- Report ----------

def report() -> None:
    sources = [s for s in SOURCES if any((OUT / f"{d}.{s}.jsonl").exists() for d in TABLES)]
    out: dict = {"sources": sources, "datasets": {}, "verdict": {}}
    speed = jev_speed()
    print("Can open, Jev-compatible models replace Jev as the default backend? (criteria in experiments/database/protocol/p6.txt)")
    print("Sources with collected judgements: " + (", ".join(NAMES[s] for s in sources) or "none yet"))

    for dataset in TABLES:
        have = [s for s in sources if (OUT / f"{dataset}.{s}.jsonl").exists()]
        Y, P, keys = LOADERS[dataset](have)
        rng = np.random.default_rng([SEED, TABLES.index(dataset)])
        boot = bootstrap(P, Y, rng)
        d = out["datasets"][dataset] = {"rows": int(len(Y)), "questions": int(Y.shape[1]), "positive": float(Y.mean()), "sources": {}}
        print(f"\n===== {TITLES[dataset]}: {len(Y)} rows × {Y.shape[1]} questions, {Y.mean():.1%} labelled true by humans =====")
        print(f"{'source':30s}{'no ans':>7s}{'accuracy':>9s}{'AUC':>8s}{'Brier':>8s}{'ECE width':>10s}{'ECE mass':>9s}   middle band 0.1-0.9: cells (share), mean probability / actually true / accuracy      errors within the least certain 10/20/30/50%")
        for s, p in P.items():
            m = d["sources"][s] = describe(p, Y)
            mid, conc = m["middle"], m["concentration"]
            middle = f"{mid['n']:>6d} ({mid['share']:.1%}), {mid['mean_p']:.2f} / {mid['positive']:.2f} / {mid['accuracy']:.1%}" if mid["n"] else "     0"
            print(f"{NAMES[s]:30s}{m['missing']:>8d}{m['acc']:>10.1%}{m['auc']:>9.3f}{m['brier']:>9.3f}{m['ece']:>9.3f}{m['ece_mass']:>10.3f}   {middle:52s}"
                  + " / ".join(f"{conc[str(b)]:.0%}" for b in SHARES if conc))
        # Reference: ignoring everything and answering each question with its base rate, what do the metrics give?
        m = d["no_skill"] = describe(np.tile(Y.mean(axis=0), (len(Y), 1)), Y)
        print(f"{NAMES['no-skill']:30s}{'':>8s}{m['acc']:>10.1%}{m['auc']:>9.3f}{m['brier']:>9.3f}{m['ece']:>9.3f}{m['ece_mass']:>10.3f}")
        # When requests exceed a model's validated window: on the rows that fit only, what do the model and Jev score?
        for s in have:
            fits = np.array([k not in over_window(s, dataset) for k in keys])
            if not fits.any():
                print(f"  every request of {NAMES[s]} exceeds its window ({WINDOW[s]} tokens)")
            elif not fits.all():
                a, b = four(P[s][fits], Y[fits]), four(P["jev"][fits], Y[fits])
                d["sources"][s]["within_window"] = {"rows": int(fits.sum()), "own": dict(zip(STATS, map(float, a))), "jev": dict(zip(STATS, map(float, b)))}
                print(f"  {NAMES[s]}: {(~fits).sum()} rows exceed its window ({WINDOW[s]} tokens). On the {fits.sum()} rows that fit: accuracy {a[0]:.1%} (Jev {b[0]:.1%}), "
                      f"AUC {a[1]:.3f} ({b[1]:.3f}), Brier {a[2]:.3f} ({b[2]:.3f})")
        print("  reliability table (per bin: cells, mean probability → actually true)")
        for s in P:
            cells = [f"{b['n']}:{b['mean_p']:.2f}→{b['positive']:.2f}" if b["n"] else "0" for b in d["sources"][s]["reliability"]]
            print(f"  {NAMES[s]:28s}" + "  ".join(cells))

        # Verdict against Jev for each source, by the criteria fixed beforehand
        if dataset == "mams.packed":
            continue  # packing is a secondary analysis and takes no part in the verdict
        if len(P) > 1:
            print("  difference from Jev (paired on the same rows, 95% bootstrap interval in brackets) and verdict")
        for s in P:
            if s == "jev":
                continue
            m, jev = d["sources"][s], d["sources"]["jev"]
            diff = boot[s] - boot["jev"]
            v = {"missing_share": m["missing"] / m["cells"]}
            line = []
            for i, key in enumerate(STATS):
                gap = m[key] - jev[key]
                lo, hi = np.nanquantile(diff[:, i], [0.025, 0.975])
                label, bar, sign = BARS.get(key, ("ECE mass", ECE_BAR, -1))
                ok = bool(gap >= bar) if sign > 0 else bool(gap <= bar)
                v[key] = {"gap": float(gap), "ci": [float(lo), float(hi)], "bar": bar, "pass": ok, "separable": not (lo <= bar <= hi)}
                unit = 100 if key == "acc" else 1
                shown = f"{gap * unit:+.1f} pts [{lo * unit:+.1f}, {hi * unit:+.1f}]" if key == "acc" else f"{gap:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                line.append(f"{label} {shown} {'pass' if ok else 'fail'}{'' if v[key]['separable'] else ' (not separable from the bar)'}")
            v["served"] = v["missing_share"] <= MISSING_BAR
            v["matches_jev"] = v["served"] and all(v[k]["pass"] for k in BARS)
            v["calibrated_like_jev"] = v["ece_mass"]["pass"]
            out["verdict"].setdefault(s, {})[dataset] = v
            note = ""
            if s == JEV:
                # Control group: the same model asked again a few days later. A difference between another source and Jev only means something if it exceeds this
                both = ~np.isnan(P[s])
                v["agreement"] = float(((P[s] >= 0.5) == (P["jev"] >= 0.5))[both].mean())
                v["mean_abs_difference"] = float(np.abs(P[s] - P["jev"])[both].mean())
                note = f"; the two passes agree on {v['agreement']:.2%}, mean probability difference {v['mean_abs_difference']:.3f}"
            elif JEV in P:
                # Robustness: does the verdict change with today's re-asked Jev as the reference?
                fresh = d["sources"][JEV]
                v["matches_jev_fresh"] = v["served"] and all((m[k] - fresh[k] >= bar) if sign > 0 else (m[k] - fresh[k] <= bar) for k, (_, bar, sign) in BARS.items())
                note = "" if v["matches_jev_fresh"] == v["matches_jev"] else f"; with today's re-asked Jev as reference it would be \"{'on par' if v['matches_jev_fresh'] else 'behind'}\""
            print(f"  {NAMES[s]:28s}" + "; ".join(line[:3]) + f" → {'on par with Jev' if v['matches_jev'] else 'behind Jev'}"
                  + ("" if v["served"] else f" ({v['missing_share']:.1%} of requests unanswered, more than 5%)") + note)
            print(f"  {'':28s}calibration: {line[3]}")

        # After recalibration with human labels (secondary)
        rng = np.random.default_rng([SEED, 100 + TABLES.index(dataset)])
        print(f"  after recalibration with {LABELS} human-labelled rows, on the remaining rows (mean of {DRAWS} draws):")
        for s, p in P.items():
            after = recalibrated(p, Y, rng)
            d["sources"][s]["recalibrated"] = dict(zip(STATS, map(float, after)))
            print(f"  {NAMES[s]:28s}accuracy {after[0]:.1%}, Brier {after[2]:.3f}, ECE mass {after[3]:.3f}")

    # Packed vs one by one (secondary)
    if any((OUT / f"mams.packed.{s}.jsonl").exists() and (OUT / f"mams.{s}.jsonl").exists() for s in sources):
        print("\n===== Restaurant reviews: 21 questions packed vs one by one (secondary) =====")
        Y, single, _ = mams([s for s in sources if (OUT / f"mams.{s}.jsonl").exists()])
        _, packed, _ = mams([s for s in sources if (OUT / f"mams.packed.{s}.jsonl").exists()], packed=True)
        out["packing"] = {}
        for s in packed:
            if s in single:
                agree = float(((filled(single[s]) >= 0.5) == (filled(packed[s]) >= 0.5)).mean())
                out["packing"][s] = {"single": float(correct(single[s], Y).mean()), "packed": float(correct(packed[s], Y).mean()), "agreement": agree}
                print(f"  {NAMES[s]:28s}one by one {out['packing'][s]['single']:.1%}, packed {out['packing'][s]['packed']:.1%}, the two forms agree on {agree:.1%}")

    # The same source asked the same reviews again (secondary): does it answer the same way twice?
    repeated = [s for s in sources if (OUT / f"movie.repeat.{s}.jsonl").exists() and (OUT / f"movie.{s}.jsonl").exists()]
    if repeated:
        print("\n===== The first 200 movie reviews asked again after all other data (secondary) =====")
        out["repeat"] = {}
        for s in repeated:
            first = {r["text"]: r["p"] for r in read(OUT / f"movie.{s}.jsonl")}
            pairs = np.array([(first[r["text"]], r["p"]) for r in read(OUT / f"movie.repeat.{s}.jsonl") if r["p"] is not None and first.get(r["text"]) is not None])
            gap = np.abs(pairs[:, 0] - pairs[:, 1])
            out["repeat"][s] = {"rows": len(pairs), "agreement": float(((pairs[:, 0] >= 0.5) == (pairs[:, 1] >= 0.5)).mean()), "mean_abs_difference": float(gap.mean()), "max_abs_difference": float(gap.max())}
            print(f"  {NAMES[s]:28s}{len(pairs)} reviews, the two passes agree on {out['repeat'][s]['agreement']:.2%}, mean probability difference {gap.mean():.4f}, max {gap.max():.4f}")

    # Throughput, cost, context window
    print("\n===== Batch throughput and cost (one cold pass; local models are free, hardware and electricity not counted) =====")
    print(f"{'source':30s}{'dataset':14s}{'conc':>4s}{'requests/s':>11s}{'of Jev':>8s}{'per 1k rows':>12s}{'no ans':>7s}{'longest request tokens / window':>33s}")
    out["speed"] = {"jev": speed}
    for dataset in TABLES:
        print(f"{'Jev 1.13.0 (earlier cold runs)':30s}{dataset:14s}{16:>6d}{speed[dataset]:>14.1f}{'':>12s}{'$' + format(jev_cost_per_1000(dataset), '.3f'):>14s}")
    for s in sources:
        for dataset in TABLES:
            r = runs(s, dataset)
            if not r:
                continue
            longest, over = max(row.get("request_tokens", 0) for row in read(OUT / f"{dataset}.{s}.jsonl")), len(over_window(s, dataset))
            ratio = r["requests_per_second"] / speed[dataset] if r["requests_per_second"] else None
            out["speed"].setdefault(s, {})[dataset] = {**r, "ratio_to_jev": ratio, "longest_request_tokens": longest, "window": WINDOW[s], "rows_over_window": over}
            if dataset != "mams.packed":
                out["verdict"][s][dataset]["practical"] = None if s == JEV or ratio is None else bool(ratio >= SPEED_BAR)
            print(f"{NAMES[s]:30s}{dataset:14s}{r['concurrency']:>6d}{r['requests_per_second'] or 0:>14.1f}{'' if ratio is None else format(ratio, '.0%'):>12s}"
                  f"{'$' + format(r['cost_per_1000'], '.3f'):>14s}{r['errors']:>9d}{longest:>14d} / {WINDOW[s]}{f'  {over} rows exceed the window' if over else ''}")

    # Verdict
    print("\n===== By the criteria fixed beforehand: can it replace Jev as the default backend? =====")
    primary = [d for d in TABLES if d != "mams.packed"]
    for s in ["v41flash.off"] + sources:
        v = out["verdict"].get(s, {})
        done = [d for d in primary if d in v]
        if s in ("v41flash.off", JEV):
            print(f"  {NAMES[s]} ({'reference, two datasets only' if s != JEV else 'control group'}): " + ", ".join(f"{TITLES[d]} {'on par' if v[d]['matches_jev'] else 'behind'}" for d in done))
            continue
        if len(done) < len(primary):
            print(f"  {NAMES[s]}: still missing {', '.join(TITLES[d] for d in primary if d not in v)}, no conclusion yet")
            continue
        ok = all(v[d]["matches_jev"] for d in primary)
        v["can_replace_jev"] = ok
        practical = [v[d].get("practical") for d in primary]
        speed_note = "" if practical[0] is None else ("; throughput on this machine is enough for batch use" if all(practical) else "; throughput on this machine is below 10% of Jev, not practical for batch use")
        flips = [TITLES[d] for d in primary if v[d].get("matches_jev_fresh", v[d]["matches_jev"]) != v[d]["matches_jev"]]
        robust = "; with today's re-asked Jev as the reference, the verdict differs on: " + ", ".join(flips) if flips else ""
        print(f"  {NAMES[s]}: {'yes' if ok else 'no'} (" + ", ".join(f"{TITLES[d]} {'on par' if v[d]['matches_jev'] else 'behind'}" for d in primary) + f"){speed_note}{robust}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + "\n")


if __name__ == "__main__":
    sys.exit(report())
