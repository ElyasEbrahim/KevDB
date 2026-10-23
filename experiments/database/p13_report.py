"""Phase 13 analysis: the reading of experiments/database/protocol/p13.txt. Reads collected answers only and calls no model.

    python experiments/database/p13_report.py      out/database/p13/report.txt and results.json

For every close arm of phase 10 and every seed (1 is phase 10's own checkpoint, 2 and 3 were trained here): line T against Jev's zero-shot
answers on the held-out dataset and guard G against the released checkpoint, judged exactly as phase 10 judged them (p10_report.py:
temperature on the pools of the three training datasets, phase 6's rule, 2,000 paired bootstrap resamples). The analysis runs only once
every run of both models is settled; before that it prints the inventories and stops.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys

import numpy as np

import p13_eval as P  # first: it switches phase 10's modules to their rehearsal rows when P13_REHEARSAL=1
import p10_data as D10  # noqa: E402
import p10_report as R  # noqa: E402

BASE = P.OUT
STATS = ("acc", "auc", "brier", "f1")


def read(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def run_dir(model: str, arm: str, seed: int):
    return P.P10_OUT / model / arm if seed == 1 else BASE / model / f"{arm}.seed{seed}"


def guard(rows: list[dict], g: str, ref: tuple[np.ndarray, np.ndarray]) -> dict:
    c, w = R.guard_units(g, rows)
    c0, w0 = ref
    assert np.array_equal(w, w0), f"{g}: other rows than the released checkpoint's"
    d = c.sum() / w.sum() - c0.sum() / w0.sum()
    return {"diff": float(d), "pass": bool(d >= R.GUARD_BAR)}


def judge(model: str, arm: str, seed: int, g_released: dict) -> dict:
    """One seed of one arm, as phase 10 judged its arms."""
    held = arm.removeprefix("loo-")
    where = run_dir(model, arm, seed)
    p_raw, y, ids = R.noul(read(where / f"test.{held}.jsonl"))
    ps, ys = zip(*[R.noul(read(where / f"pool.{d}.jsonl"))[:2] for d in D10.ARMS[arm]])
    T = R.fit_temperature(np.concatenate(ps), np.concatenate(ys))
    p = R.scale(p_raw, T)
    vs_jev = R.paired(p, R.jev_test(held, ids, y), y)
    missing = float(np.isnan(p_raw).mean())
    on_par, crit, _ = R.judge_t(vs_jev, missing)
    g = {name: guard(read(where / f"{name}.jsonl"), name, g_released[name]) for name in P.E.GUARDS}
    stats = {k: float(v) for k, v in R.stats(p, y).items()}
    return {"seed": seed, "T": T, "stats": stats, "vs_jev": vs_jev, "T_on_par": bool(on_par), "criteria": crit, "guard": g,
            "G": all(x["pass"] for x in g.values()), "on_par": bool(on_par and all(x["pass"] for x in g.values())), "missing": missing,
            "errors": R.readings(p, y)}


def verdict(x: dict) -> str:
    if x["on_par"]:
        return "on par"
    failed = [k for k, ok in x["criteria"].items() if not ok] + [f"guard {g}" for g, v in x["guard"].items() if not v["pass"]]
    return "below: " + ", ".join(failed)


def stable(outcomes: list[bool]) -> bool:
    return len(set(outcomes)) == 1


def report() -> int:
    lines: list[str] = []
    say = lines.append
    say("Phase 13: are phase 10's close verdicts stable when only the training seed changes?"
        + (" (rehearsal: every row made up, the Jev column made up from the labels)" if P.REHEARSAL else ""))
    pending = 0
    for model in P.MODELS:
        P.configure(model)
        inventory = io.StringIO()
        with contextlib.redirect_stdout(inventory):
            pending += P.E.check(model)
        say(inventory.getvalue().rstrip())
    if pending:
        say(f"\n{pending} run(s) still pending: no answer is analysed until every run is settled.")
        print("\n".join(lines))
        return 1
    results: dict = {"rehearsal": P.REHEARSAL, "arms": {}}
    for model in P.MODELS:
        P.configure(model)
        g_released = {g: R.guard_units(g, read(P.P10_OUT / model / "released" / f"{g}.jsonl")) for g in P.E.GUARDS}
        for arm in P.CLOSE[model]:
            key = f"{model} {arm}"
            status = {s: P.E.status(model, f"{arm}.seed{s}") for s in P.SEEDS}
            seeds = [judge(model, arm, 1, g_released)] + [judge(model, arm, s, g_released) for s in P.SEEDS if status[s] == "complete"]
            outcomes = [x["on_par"] for x in seeds]
            rule = not (model == "kev-4b" and arm == "loo-amazon-google")
            spread = {k: [min(x["stats"][k] for x in seeds), max(x["stats"][k] for x in seeds)] for k in STATS}
            spread.update({f"guard {g}": [min(x["guard"][g]["diff"] for x in seeds), max(x["guard"][g]["diff"] for x in seeds)] for g in P.E.GUARDS})
            results["arms"][key] = {"by_rule": rule, "seeds": seeds, "missing_seeds": [s for s in P.SEEDS if status[s] != "complete"],
                                    "stable": stable(outcomes) and len(seeds) == 1 + len(P.SEEDS), "on_par_count": sum(outcomes), "spread": spread}
            say(f"\n== {key}{'' if rule else ' (outside the rule: Kev-4B nearest verdict, reported apart)'}")
            for x in seeds:
                s, e = x["stats"], x["errors"]
                say(f"   seed {x['seed']}{' (phase 10)' if x['seed'] == 1 else '          '}  T {x['T']:.3f}  acc {s['acc']:.1%}  AUC {s['auc']:.3f}  Brier {s['brier']:.3f}  "
                    f"F1 {s['f1']:.3f}  false {e['false_matches']:>4} missed {e['missed_matches']:>4}   vs Jev "
                    + "  ".join(f"{k} {x['vs_jev'][k][0]:+.3f}" for k in ("acc", "auc", "brier"))
                    + "   guard " + " ".join(f"{g} {v['diff'] * 100:+.1f}" for g, v in x["guard"].items()) + f"   {verdict(x)}")
            for s in P.SEEDS:
                if status[s] != "complete":
                    say(f"   seed {s}: {status[s]}")
            r = results["arms"][key]
            say(f"   on par in {r['on_par_count']} of {len(seeds)} seeds; " + ("stable" if r["stable"] else "NOT stable to the seed" if len(seeds) == 3 else "undetermined: a seed is missing")
                + "; spread " + ", ".join(f"{k} {lo:.3f}-{hi:.3f}" for k, (lo, hi) in spread.items() if k in STATS)
                + ", " + ", ".join(f"{k} {lo * 100:+.1f} to {hi * 100:+.1f}" for k, (lo, hi) in spread.items() if k.startswith("guard")))
    say("\n== Reading (rules written before the run, p13.txt)")
    for key, r in results["arms"].items():
        outcomes = " | ".join(f"seed {x['seed']} {verdict(x)}" for x in r["seeds"])
        say(f"   {key:<28} {'stable' if r['stable'] else 'not stable' if not r['missing_seeds'] else 'undetermined'} ({outcomes})"
            + ("" if r["by_rule"] else "  [outside the rule]"))
    print("\n".join(lines))
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(report())
