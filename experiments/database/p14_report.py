"""Phase 14 analysis: the reading of experiments/database/protocol/p14.txt. Reads collected answers only and calls no model.

    python experiments/database/p14_report.py      out/database/p14/report.txt and results.json

For each of phase 13's three Kev-0.8B arms, four checkpoints: phase 10's seed 1 (trained without the padding launcher), phase 14's seed 1
(the same seed, through the launcher), and phase 13's seeds 2 and 3 (through the launcher). Each is judged exactly as phase 10 judged its
arms (p10_report.py). The question is the movie-review guard g1: does padded seed 1 drop as phase 10's seed 1 did, or pass as seeds 2 and
3 did? The analysis runs only once every run is settled; before that it prints the inventory and stops.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys

import numpy as np

import p14_eval as P  # first: it switches phase 10's modules to their rehearsal rows when P14_REHEARSAL=1
import p10_data as D10  # noqa: E402
import p10_report as R  # noqa: E402

MODEL = "kev-0.8b"
STATS = ("acc", "auc", "brier", "f1")
# (label, folder of the run's answers) per condition, in the order they are reported
CONDITIONS = (("seed 1, phase 10 (no padding)", lambda arm: P.P10_OUT / MODEL / arm),
              ("seed 1, padded (phase 14)", lambda arm: P.OUT / MODEL / f"{arm}.seed1"),
              ("seed 2, padded (phase 13)", lambda arm: P.P13_OUT / MODEL / f"{arm}.seed2"),
              ("seed 3, padded (phase 13)", lambda arm: P.P13_OUT / MODEL / f"{arm}.seed3"))


def read(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def judge(where, arm: str, g_released: dict) -> dict:
    """One checkpoint, as phase 10 judged its arms (and phase 13 its seeds)."""
    held = arm.removeprefix("loo-")
    p_raw, y, ids = R.noul(read(where / f"test.{held}.jsonl"))
    ps, ys = zip(*[R.noul(read(where / f"pool.{d}.jsonl"))[:2] for d in D10.ARMS[arm]])
    T = R.fit_temperature(np.concatenate(ps), np.concatenate(ys))
    p = R.scale(p_raw, T)
    vs_jev = R.paired(p, R.jev_test(held, ids, y), y)
    on_par, crit, _ = R.judge_t(vs_jev, float(np.isnan(p_raw).mean()))
    guard = {}
    for g in P.E.GUARDS:
        c, w = R.guard_units(g, read(where / f"{g}.jsonl"))
        c0, w0 = g_released[g]
        assert np.array_equal(w, w0), f"{g}: other rows than the released checkpoint's"
        d = c.sum() / w.sum() - c0.sum() / w0.sum()
        guard[g] = {"diff": float(d), "pass": bool(d >= R.GUARD_BAR)}
    return {"T": T, "stats": {k: float(v) for k, v in R.stats(p, y).items()}, "vs_jev": vs_jev, "T_on_par": bool(on_par), "criteria": crit,
            "guard": guard, "G": all(x["pass"] for x in guard.values()), "on_par": bool(on_par and all(x["pass"] for x in guard.values()))}


def g1_difference(where_a, where_b) -> tuple[float, float, float]:
    """g1 accuracy of checkpoint a minus checkpoint b, with a 95% interval from 2,000 paired resamples of the review texts."""
    ca, wa = R.guard_units("g1", read(where_a / "g1.jsonl"))
    cb, wb = R.guard_units("g1", read(where_b / "g1.jsonl"))
    assert np.array_equal(wa, wb)
    idx = R.draws(len(wa))
    boot = ca[idx].sum(1) / wa[idx].sum(1) - cb[idx].sum(1) / wb[idx].sum(1)
    return float(ca.sum() / wa.sum() - cb.sum() / wb.sum()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def pooled_g1_difference(pairs: list[tuple]) -> tuple[float, float, float]:
    """Mean over arms of (g1 accuracy of a minus b), with a 95% interval from 2,000 paired resamples of the review texts, drawn once and
    applied to every arm (all arms are asked the same 1,000 texts)."""
    loaded = []
    for where_a, where_b in pairs:
        ca, wa = R.guard_units("g1", read(where_a / "g1.jsonl"))
        cb, wb = R.guard_units("g1", read(where_b / "g1.jsonl"))
        assert np.array_equal(wa, wb)
        loaded.append((ca, cb, wa))
    idx = R.draws(len(loaded[0][2]))
    point = np.mean([ca.sum() / w.sum() - cb.sum() / w.sum() for ca, cb, w in loaded])
    boot = np.mean([ca[idx].sum(1) / w[idx].sum(1) - cb[idx].sum(1) / w[idx].sum(1) for ca, cb, w in loaded], axis=0)
    return float(point), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def reading(diff: tuple[float, float, float]) -> str:
    """The pre-registered reading from the pooled interval of padded seed 1 minus phase 10's seed 1 on g1."""
    if diff[1] > 0:
        return "B: padding moves seed 1 up on movie reviews, so phase 10's drop came with its unpadded runs (the padding or their dropout draws)"
    if diff[1] <= 0 <= diff[2]:
        return "A: padded seed 1 cannot be told apart from phase 10's seed 1, so the drop belongs to seed 1 (its replay sample or record order)"
    return "undetermined: padded seed 1 is lower still"


def verdict(x: dict) -> str:
    if x["on_par"]:
        return "on par"
    return "below: " + ", ".join([k for k, ok in x["criteria"].items() if not ok] + [f"guard {g}" for g, v in x["guard"].items() if not v["pass"]])


def report() -> int:
    P.configure(MODEL)
    lines: list[str] = []
    say = lines.append
    inventory = io.StringIO()
    with contextlib.redirect_stdout(inventory):
        pending = P.E.check(MODEL)
    say("Phase 14: is Kev-0.8B's drop on movie reviews in phase 10 a property of seed 1 or of training without padding?"
        + (" (rehearsal: every row made up, the Jev column made up from the labels)" if P.REHEARSAL else ""))
    say(inventory.getvalue().rstrip())
    if pending:
        say(f"\n{pending} run(s) still pending: no answer is analysed until every run is settled.")
        print("\n".join(lines))
        return 1
    g_released = {g: R.guard_units(g, read(P.P10_OUT / MODEL / "released" / f"{g}.jsonl")) for g in P.E.GUARDS}
    results: dict = {"rehearsal": P.REHEARSAL, "model": MODEL, "arms": {}}
    fails = 0
    for arm in P.CLOSE[MODEL]:
        status = P.E.status(MODEL, f"{arm}.seed1")
        say(f"\n== {arm}")
        if status != "complete":
            say(f"   padded seed 1: {status}")
            results["arms"][arm] = {"status": status}
            continue
        judged = {label: judge(where(arm), arm, g_released) for label, where in CONDITIONS}
        for label, x in judged.items():
            s = x["stats"]
            say(f"   {label:<31} T {x['T']:.3f}  acc {s['acc']:.1%}  AUC {s['auc']:.3f}  Brier {s['brier']:.3f}  F1 {s['f1']:.3f}   guard "
                + " ".join(f"{g} {v['diff'] * 100:+.1f}" for g, v in x["guard"].items()) + f"   {verdict(x)}")
        padded, unpadded = judged[CONDITIONS[1][0]], judged[CONDITIONS[0][0]]
        diff = g1_difference(CONDITIONS[1][1](arm), CONDITIONS[0][1](arm))
        fails += not padded["guard"]["g1"]["pass"]
        say(f"   g1 padded seed 1 minus phase 10's seed 1: {diff[0] * 100:+.1f} points ({diff[1] * 100:+.1f} to {diff[2] * 100:+.1f}); "
            f"padded seed 1 {'fails' if not padded['guard']['g1']['pass'] else 'passes'} the g1 guard; "
            f"verdict {'unchanged' if padded['on_par'] == unpadded['on_par'] else 'changed'} against phase 10 ({verdict(unpadded)} -> {verdict(padded)})")
        results["arms"][arm] = {"status": "complete", "conditions": judged, "g1_padded_minus_phase10": diff,
                                "padded_g1_passes": padded["guard"]["g1"]["pass"]}
    done = [a for a, x in results["arms"].items() if x.get("status") == "complete"]
    results["fails_g1"] = fails
    if len(done) == len(P.CLOSE[MODEL]):
        main = pooled_g1_difference([(CONDITIONS[1][1](a), CONDITIONS[0][1](a)) for a in done])
        reference = pooled_g1_difference([(CONDITIONS[i][1](a), CONDITIONS[0][1](a)) for a in done for i in (2, 3)])
        results.update({"g1_padded_seed1_minus_phase10": main, "g1_seeds23_minus_phase10": reference, "reading": reading(main)})
        say(f"\ng1, mean over the {len(done)} arms (95% interval, review texts resampled):")
        say(f"   padded seed 1 minus phase 10's seed 1:   {main[0] * 100:+.1f} points ({main[1] * 100:+.1f} to {main[2] * 100:+.1f})")
        say(f"   seeds 2 and 3 minus phase 10's seed 1:   {reference[0] * 100:+.1f} points ({reference[1] * 100:+.1f} to {reference[2] * 100:+.1f})  [reference]")
    else:
        results["reading"] = "undetermined: a run is missing"
    say(f"\n== Reading (rules written before the run, p14.txt)\n   {results['reading']}\n   padded seed 1 fails the g1 guard on {fails} of {len(done)} arms")
    print("\n".join(lines))
    P.OUT.mkdir(parents=True, exist_ok=True)
    (P.OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(report())
