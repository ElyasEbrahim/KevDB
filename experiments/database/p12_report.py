"""Phase 12 analysis: the lines and readings of experiments/database/protocol/p12.txt. Reads collected answers only and calls no model.

    python experiments/database/p12_report.py      out/database/p12/kev-4b/report.txt and results.json

Per dataset, on its test pairs (both records on the test side): Jev's zero-shot answers, the starting points at n = 0 (phase 10's
loo-<dataset> with phase 10's temperature, the released checkpoint as it ships) and every trained run with the temperature fitted on its own
held-out 20%. Metrics, temperature fit, bootstrap, line T and guard are phase 10's (p10_report.py), unchanged. The analysis runs only once
every run is settled; before that it prints the inventory and stops.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys

import numpy as np

import p10_report as R
import p12_data as D12
import p12_eval as P
from paths import OUT as RESULTS, ROOT

MODEL = P.MODEL
BASE = RESULTS / ("p12_rehearsal" if P.REHEARSAL else "p12")
STATS = ("acc", "auc", "brier", "f1")


def read(base, arm: str, name: str) -> list[dict]:
    path = base / MODEL / arm / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def jev(dataset: str, ids: list[str], y: np.ndarray) -> np.ndarray:
    """Jev's cached answers (2026-10-05) to the test pairs, from its test-split and validation-sample files."""
    if P.REHEARSAL:
        return R.made_up_p(ids, y)
    got = {}
    for split in ("test", "valid"):
        for r in map(json.loads, (ROOT / "out" / "entity_match" / f"{dataset}.{split}.{D12.JEV_MODEL}.jsonl").read_text().splitlines()):
            got[r["id"]] = r
    assert all(got[i]["label"] == label for i, label in zip(ids, y)), f"{dataset}: Jev's files have other labels"
    return np.array([np.nan if got[i].get("p") is None else got[i]["p"] for i in ids], float)


def phase_10_temperature(dataset: str) -> float:
    """loo-<dataset>'s temperature as phase 10 fitted it: on the pool labels of the three datasets it trained on."""
    ps, ys = [], []
    for d in D12.EM:
        if d != dataset:
            p, y, _ = R.noul(read(P.P10_OUT, f"loo-{dataset}", f"pool.{d}"))
            ps.append(p)
            ys.append(y)
    return R.fit_temperature(np.concatenate(ps), np.concatenate(ys))


def released_temperature() -> float:
    """The temperature the released checkpoint ships with, as its server reported it when its answers were collected here."""
    state = json.loads((BASE / MODEL / "released" / "eval.json").read_text())
    return float(next(a["card"]["temperature"] for a in state["attempts"] if a.get("card")))


def intervals(p: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
    point, boot = R.stats(p, y), R.stats(p, y, R.draws(len(y)))
    return {k: [float(point[k]), float(np.nanpercentile(boot[k], 2.5)), float(np.nanpercentile(boot[k], 97.5))] for k in STATS}


def guard(rows: list[dict], g: str, ref: tuple[np.ndarray, np.ndarray]) -> dict:
    c, w = R.guard_units(g, rows)
    c0, w0 = ref
    assert np.array_equal(w, w0), f"{g}: other rows than the released checkpoint's"
    idx = R.draws(len(w))
    diff = (c[idx].sum(1) / w[idx].sum(1)) - (c0[idx].sum(1) / w0[idx].sum(1))
    d = c.sum() / w.sum() - c0.sum() / w0.sum()
    return {"diff": float(d), "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))], "pass": bool(d >= R.GUARD_BAR)}


def score(p_raw: np.ndarray, T: float, y: np.ndarray, p_jev: np.ndarray) -> dict:
    p = R.scale(p_raw, T)
    vs_jev = R.paired(p, p_jev, y)
    missing = float(np.isnan(p_raw).mean())
    on_par, crit, separable = R.judge_t(vs_jev, missing)
    return {"p": p, "T": T, "stats": intervals(p, y), "errors": R.readings(p, y), "vs_jev": vs_jev, "on_par": bool(on_par), "criteria": crit,
            "separable": separable, "missing": missing}


def n_star(meets: dict[int, bool]) -> tuple[int | None, str, bool]:
    """Line n*: the smallest of 0 and the budgets at which the main run is on par with Jev and passes the guard ({budget: both met} of the
    settled runs). Undetermined if a run below it is missing. Returns (n*, how it reads, whether every larger budget meets it too)."""
    budgets = [0, *D12.BUDGETS]
    reached = [n for n in budgets if meets.get(n)]
    first = reached[0] if reached else None
    if any(n not in meets for n in budgets if first is None or n < first):
        return None, "undetermined: runs missing", False
    if first is None:
        return None, f"more than {D12.BUDGETS[-1]:,}", False
    return first, f"{first:,}", all(meets.get(n, False) for n in budgets if n >= first)


def fmt(x: dict) -> str:
    s, e = x["stats"], x["errors"]
    return (f"acc {s['acc'][0]:.1%} ({s['acc'][1]:.1%}-{s['acc'][2]:.1%})  AUC {s['auc'][0]:.3f}  Brier {s['brier'][0]:.3f}  "
            f"F1 {s['f1'][0]:.3f} ({s['f1'][1]:.3f}-{s['f1'][2]:.3f})  false {e['false_matches']:>3} missed {e['missed_matches']:>3}")


def report() -> int:
    P.configure()
    lines: list[str] = []
    say = lines.append
    inventory = io.StringIO()
    with contextlib.redirect_stdout(inventory):
        pending = P.E.check(MODEL)
    say("Phase 12, Kev-4B: with how many of their own labels does it reach Jev's zero-shot level on a matching dataset it has not seen?"
        + (" (rehearsal: every row made up, the Jev column made up from the labels)" if P.REHEARSAL else ""))
    say(inventory.getvalue().rstrip())
    if pending:
        say(f"\n{pending} run(s) still pending: no answer is analysed until every run is settled.")
        print("\n".join(lines))
        return 1
    status = {arm: P.E.status(MODEL, arm) for arm in P.ARMS}
    results: dict = {"model": MODEL, "rehearsal": P.REHEARSAL, "status": status, "datasets": {}}
    g_released = {g: R.guard_units(g, read(P.P10_OUT, "released", g)) for g in P.E.GUARDS}
    T_released = released_temperature()
    say(f"\nLine n*: the smallest budget at which the run is on par with Jev on the test pairs (phase 6's rule: accuracy at most 2.0 points below, "
        f"AUC at most 0.02 below, Brier at most 0.02 above) and no guard set is more than 2.0 points below the released checkpoint. "
        f"Main runs start from phase 10's loo-<dataset>; reference runs from the released checkpoint (ships with T = {T_released:.3f}).")

    for d in D12.EM:
        out: dict = {"runs": {}}
        if status["released"] != "complete" or status[D12.arm(d, "loo", 0)] != "complete":
            say(f"\n== {d}: a starting point is missing ({status['released']}; {status[D12.arm(d, 'loo', 0)]}); no line")
            results["datasets"][d] = {"status": "starting point missing"}
            continue
        rel0 = R.noul(read(BASE, "released", f"side.{d}"))
        y, ids = rel0[1], rel0[2]
        p_jev = jev(d, ids, y)
        T_loo = phase_10_temperature(d)
        jev_score = {"stats": intervals(p_jev, y), "errors": R.readings(p_jev, y)}
        say(f"\n== {d}: {len(y)} test pairs, {int(y.sum())} matches")
        say(f"   {'Jev, zero-shot':<34}{fmt(jev_score)}")
        runs: list[tuple[str, str, int, dict, dict, bool]] = []  # (arm, start, n, score, guard, guard passes)

        main0 = R.noul(read(BASE, D12.arm(d, "loo", 0), f"side.{d}"))
        assert main0[2] == ids, f"{d}: other rows than the released checkpoint's"
        g0 = {g: guard(read(P.P10_OUT, f"loo-{d}", g), g, g_released[g]) for g in P.E.GUARDS}
        runs.append((D12.arm(d, "loo", 0), "loo", 0, score(main0[0], T_loo, y, p_jev), g0, all(x["pass"] for x in g0.values())))
        runs.append(("released", "rel", 0, score(rel0[0], 1.0, y, p_jev), {}, True))  # as it ships: its own temperature is applied by the server
        for arm in P.TRAINED:
            dataset, start, n = D12.parse(arm)
            if dataset != d:
                continue
            if status[arm] != "complete":
                out["runs"][arm] = {"status": status[arm]}
                say(f"   {arm:<34}{status[arm]}")
                continue
            got = R.noul(read(BASE, arm, f"side.{d}"))
            assert got[2] == ids, f"{arm}: other rows than the released checkpoint's"
            sl = R.noul(read(BASE, arm, f"slice.{d}.{n}"))
            T = R.fit_temperature(sl[0], sl[1])
            x = score(got[0], T, y, p_jev)
            x["slice"] = {"labels": int((~np.isnan(sl[0])).sum()), "matches": int(sl[1][~np.isnan(sl[0])].sum())}
            # Reading: the same answers with the starting point's temperature instead of the one fitted on the run's own 20%
            x["with_start_T"] = score(got[0], T_loo if start == "loo" else T_released, y, p_jev)
            g = {name: guard(read(BASE, arm, name), name, g_released[name]) for name in P.E.GUARDS}
            runs.append((arm, start, n, x, g, all(v["pass"] for v in g.values())))

        for arm, start, n, x, g, g_ok in runs:
            label = f"{'main' if start == 'loo' else 'reference'} n = {n}" + (f", T = {x['T']:.3f}" if arm != "released" else ", as shipped")
            say(f"   {label:<34}{fmt(x)}")
            crit = ", ".join(k for k, ok in x["criteria"].items() if not ok)
            say(f"   {'':<34}vs Jev " + "  ".join(f"{k} {v[0]:+.3f} ({v[1]:+.3f} to {v[2]:+.3f})" for k, v in x["vs_jev"].items() if k != "f1")
                + f"; {'on par' if x['on_par'] else 'below: ' + crit}"
                + (f"; guard {'passes' if g_ok else 'FAILS'} (" + ", ".join(f"{k} {v['diff'] * 100:+.1f}" for k, v in g.items()) + ")" if g else ""))
            if "slice" in x:
                w = x["with_start_T"]
                say(f"   {'':<34}temperature from {x['slice']['labels']} held-out labels ({x['slice']['matches']} matches); with the starting point's "
                    f"temperature instead: Brier {w['stats']['brier'][0]:.3f}, {'on par' if w['on_par'] else 'not on par'}")
            out["runs"][arm] = {"start": start, "n": n, "T": x["T"], "stats": x["stats"], "errors": x["errors"], "vs_jev": x["vs_jev"], "on_par": x["on_par"],
                                "criteria": x["criteria"], "separable": x["separable"], "missing": x["missing"], "guard": g, "guard_passes": g_ok,
                                **({"slice": x["slice"], "with_start_T": {"T": x["with_start_T"]["T"], "stats": x["with_start_T"]["stats"],
                                                                          "on_par": x["with_start_T"]["on_par"]}} if "slice" in x else {})}

        main = {n: (x, g_ok) for arm, start, n, x, g, g_ok in runs if start == "loo"}
        out["n_star"], out["n_star_reading"], out["holds_at_every_larger_budget"] = n_star({n: x["on_par"] and g_ok for n, (x, g_ok) in main.items()})
        say(f"   n* = {out['n_star_reading']}" + ("" if out["n_star"] is None else f"; on par at every larger budget too: {'yes' if out['holds_at_every_larger_budget'] else 'no'}"))
        # Reading: does starting from phase 10's checkpoint save labels? Main minus reference at the same budget, paired on the same pairs
        ref = {n: x for arm, start, n, x, g, g_ok in runs if start == "rel"}
        out["main_vs_reference"] = {}
        for n in [0, *D12.REFERENCE_BUDGETS]:
            if n in main and n in ref:
                diff = R.paired(main[n][0]["p"], ref[n]["p"], y)
                out["main_vs_reference"][n] = diff
                say(f"   main minus reference at n = {n}: F1 {diff['f1'][0]:+.3f} ({diff['f1'][1]:+.3f} to {diff['f1'][2]:+.3f}), "
                    f"Brier {diff['brier'][0]:+.3f} ({diff['brier'][1]:+.3f} to {diff['brier'][2]:+.3f})")
        out["jev"] = jev_score
        out["T_loo_phase_10"] = T_loo
        results["datasets"][d] = out

    say("\n== Reading (rules written before the run, p12.txt)")
    for d, x in results["datasets"].items():
        if "n_star_reading" in x:
            say(f"   {d:<16}n* = {x['n_star_reading']}")
    results["T_released"] = T_released
    print("\n".join(lines))
    (BASE / MODEL / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(report())
