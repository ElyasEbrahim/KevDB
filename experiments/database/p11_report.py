"""Phase 11 analysis: the lines and readings of experiments/database/protocol/p11.txt. Reads collected answers only and calls no model.

    python experiments/database/p11_report.py      out/database/p11/kev-4b/report.txt and results.json

Every arm is compared on dblp-acm's full test split with the released Kev-4B and the control (phase 10's loo-dblp-acm), whose answers to the
same rows were collected in phase 10, and with Jev's zero-shot answers. Metrics, temperature fit, bootstrap and guard are phase 10's
(p10_report.py), unchanged. The analysis runs only once every arm is settled; before that it prints the inventory and stops.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys

import numpy as np

import p11_eval as P
import p10_report as R
import p11_data as D11
from paths import OUT as RESULTS

MODEL = "kev-4b"
P10 = RESULTS / ("p10_rehearsal" if P.REHEARSAL else "p10")
TEST = "test.dblp-acm"


def read(base, arm: str, name: str) -> list[dict]:
    path = base / MODEL / arm / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def pooled(base, arm: str, names: list[str]) -> tuple[float, int, int]:
    ps, ys = [], []
    for name in names:
        p, y, _ = R.noul(read(base, arm, name))
        ps.append(p)
        ys.append(y)
    p, y = np.concatenate(ps), np.concatenate(ys)
    keep = ~np.isnan(p)
    return R.fit_temperature(p, y), int(keep.sum()), int(y[keep].sum())


def report() -> int:
    P.configure()
    lines: list[str] = []
    say = lines.append
    inventory = io.StringIO()
    with contextlib.redirect_stdout(inventory):
        pending = P.E.check(MODEL)
    say("Phase 11, Kev-4B: why did fine-tuning not improve dblp-acm in phase 10? Three added ingredients against the control"
        + (" (rehearsal: every row made up, the Jev column made up from the labels)" if P.REHEARSAL else ""))
    say(inventory.getvalue().rstrip())
    if pending:
        say(f"\n{pending} arm(s) still pending: no answer is analysed until every arm is settled.")
        print("\n".join(lines))
        return 1
    status = {arm: P.E.status(MODEL, arm) for arm in P.ARMS}
    results: dict = {"model": MODEL, "rehearsal": P.REHEARSAL, "status": status, "arms": {}}

    released = R.noul(read(P10, "released", TEST))
    control = R.noul(read(P10, "loo-dblp-acm", TEST))
    y, ids = released[1], released[2]
    assert control[2] == ids, "the control's rows differ from the released checkpoint's"
    T_control, _, _ = pooled(P10, "loo-dblp-acm", [f"pool.{d}" for d in D11.PRODUCTS])
    p_control = R.scale(control[0], T_control)
    p_jev = R.jev_test("dblp-acm", ids, y)
    say(f"\nReferences on dblp-acm's test split ({len(y):,} pairs, {int(y.sum())} matches):")
    for label, p in (("Jev, zero-shot", p_jev), ("Kev-4B released", released[0]), (f"control loo-dblp-acm, T = {T_control:.3f}", p_control)):
        s, e = R.stats(p, y), R.readings(p, y)
        say(f"   {label:<36}acc {s['acc']:.1%}  AUC {s['auc']:.3f}  Brier {s['brier']:.3f}  F1 {s['f1']:.3f}   false {e['false_matches']:>4}  missed {e['missed_matches']:>4}")
    g_released = {g: R.guard_units(g, read(P10, "released", g)) for g in P.E.GUARDS}

    say("\nLines: P1 F1 above the released checkpoint (interval of the difference above zero); P2 F1 above the control (same); "
        "T on par with Jev (phase 6's rule); G no guard set more than 2 points below the released checkpoint.")
    for arm in P.ARMS:
        if status[arm] != "complete":
            say(f"\n== {arm}: {status[arm]}")
            results["arms"][arm] = {"status": status[arm]}
            continue
        base = RESULTS / ("p11_rehearsal" if P.REHEARSAL else "p11")
        got = R.noul(read(base, arm, TEST))
        assert got[2] == ids, f"{arm}: other rows than the released checkpoint"
        pools = [n for n in P.sets(arm) if n.startswith("pool")]
        T, n_labels, n_pos = pooled(base, arm, pools)
        p = R.scale(got[0], T)
        s, e = R.stats(p, y), R.readings(p, y)
        vs_released, vs_control, vs_jev = R.paired(p, released[0], y), R.paired(p, p_control, y), R.paired(p, p_jev, y)
        missing = float(np.isnan(got[0]).mean())
        on_par, crit, separable = R.judge_t(vs_jev, missing)
        guard = {}
        for g in P.E.GUARDS:
            c, w = R.guard_units(g, read(base, arm, g))
            c0, w0 = g_released[g]
            assert np.array_equal(w, w0)
            idx = R.draws(len(w))
            diff = (c[idx].sum(1) / w[idx].sum(1)) - (c0[idx].sum(1) / w0[idx].sum(1))
            d = c.sum() / w.sum() - c0.sum() / w0.sum()
            guard[g] = {"diff": float(d), "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))], "pass": bool(d >= R.GUARD_BAR)}
        p1, p2 = vs_released["f1"][1] > 0, vs_control["f1"][1] > 0
        g_ok = all(x["pass"] for x in guard.values())
        results["arms"][arm] = {"status": "complete", "T": T, "labels": n_labels, "positives": n_pos, "stats": {k: float(v) for k, v in s.items()},
                                "errors": e, "vs_released": vs_released, "vs_control": vs_control, "vs_jev": vs_jev, "P1": bool(p1), "P2": bool(p2),
                                "T_on_par": bool(on_par), "T_criteria": crit, "T_separable": separable, "guard": guard, "G": bool(g_ok), "missing": missing}
        say(f"\n== {arm} (trained on {', '.join(D11.ARMS[arm])}; T = {T:.3f} from {n_labels} pool labels, {n_pos} matches)")
        say(f"   acc {s['acc']:.1%}  AUC {s['auc']:.3f}  Brier {s['brier']:.3f}  F1 {s['f1']:.3f}   false {e['false_matches']:>4}  missed {e['missed_matches']:>4}"
            f"   share 0.1<p<0.9 {e['share_unsure']:.1%}")
        for label, d in (("vs released", vs_released), ("vs control", vs_control), ("vs Jev", vs_jev)):
            say(f"   {label:<12}" + "  ".join(f"{k} {v[0]:+.3f} ({v[1]:+.3f} to {v[2]:+.3f})" for k, v in d.items()))
        say(f"   P1 {'met' if p1 else 'not met'}; P2 {'met' if p2 else 'not met'}; T {'on par' if on_par else 'below: ' + ', '.join(k for k, ok in crit.items() if not ok)}; "
            f"G {'met' if g_ok else 'not met'} (" + ", ".join(f"{g} {x['diff'] * 100:+.1f}" for g, x in guard.items()) + ")")
        if arm != "pubs":
            P8 = P.E.p8_module()
            rows_arm = read(base, arm, "e2e.dblp-scholar")
            rows_rel = read(P10, "released", "e2e.dblp-scholar")
            left, _, truth, _ = P8._DATA["dblp-scholar"]
            lids = P8.subset("dblp-scholar", left)
            assert [r["id"] for r in rows_arm] == [r["id"] for r in rows_rel]
            t = {arm: R.e2e_tally(rows_arm, R.noul(rows_arm)[0], truth, lids),
                 "released": R.e2e_tally(rows_rel, R.noul(rows_rel)[0], truth, lids),
                 "jev": R.e2e_tally(rows_arm, R.jev_e2e("dblp-scholar", rows_arm), truth, lids)}
            f1 = {name: float(R.prf(x.sum(0))[2]) for name, x in t.items()}
            results["arms"][arm]["e2e_dblp_scholar"] = {"f1": f1, "vs_released": R.f1_diff(t[arm], t["released"]), "vs_jev": R.f1_diff(t[arm], t["jev"])}
            say(f"   end to end on phase 8's dblp-scholar subset: F1 {f1[arm]:.1%} (released {f1['released']:.1%}, Jev {f1['jev']:.1%})")

    done = {a: x for a, x in results["arms"].items() if x.get("status") == "complete"}
    say("\n== Reading (rules written before the run, p11.txt)")
    for arm in P.ARMS:
        if arm in done:
            x = done[arm]
            say(f"   {arm:<8}{'helps' if x['P2'] else 'does not help'} beyond the control (F1 {x['vs_control']['f1'][0]:+.3f}); "
                f"{'improves' if x['P1'] else 'does not improve'} on the released checkpoint; {'on par with Jev' if x['T_on_par'] else 'not on par with Jev'}"
                f"{'' if x['G'] else '; forgets'}")
    results["control_T"] = T_control
    print("\n".join(lines))
    out = RESULTS / ("p11_rehearsal" if P.REHEARSAL else "p11") / MODEL / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(report())
