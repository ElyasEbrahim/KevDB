"""Phase 16 figures for the README (PNG) and the report (PDF). Reads out/database/p16 and the cached answers only; calls no model.

    python experiments/database/p16_figures.py      experiments/database/report/figures/p16_*.en.{png,pdf}
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import p16_data as D
import p16_run as R
from paths import HERE

FIG = HERE / "report" / "figures"
# Colours as in report/make_figures.py (dataviz reference palette, light surface; blue and orange pass its validation script)
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#c3c2b7"
INK, SECONDARY, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#f0efec"
QUERIES = (("Q1", "Share of positive reviews (control)"), ("Q2", "Share per movie (172 groups)"),
           ("Q3a", "Mean of the movie shares"), ("Q3b", "Box-office-weighted mean"), ("Q4a", "Share per genre, through a join"),
           ("Q4b-1", "Join reusing reviews, fan-out σ = 1*"), ("Q4b-2", "Join reusing reviews, fan-out σ = 2*"),
           ("Q4b-2t", "The same, top critics reused 5× more*"), ("Q5", "Fresh or not at 60% (threshold)"),
           ("Q6", "21 restaurant predicates, many rare"))


def style() -> None:
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial"], "font.size": 9, "pdf.fonttype": 42, "axes.unicode_minus": False,
                         "text.color": INK, "axes.labelcolor": SECONDARY, "xtick.color": MUTED, "ytick.color": SECONDARY,
                         "axes.edgecolor": GRAY, "axes.linewidth": 0.6})


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.en.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def clean(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.tick_params(length=0)


def fig_propagation() -> None:
    """Per movie: error of the share of positive reviews when Jev's yes/no answers are counted, against the movie's size."""
    pop, qs = R.population("D1", "jev")
    q = qs["Q2"]
    truth, counted = q.truth(pop.y), q.truth((pop.p >= 0.5).astype(float))
    err = 100 * (counted - truth)
    n = np.asarray((q.A > 0).sum(axis=0)).ravel()
    overall = 100 * (qs["Q1"].truth((pop.p >= 0.5).astype(float)) - qs["Q1"].truth(pop.y))[0]
    off = np.abs(err) > 10
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.axhspan(-10, 10, color=BAND, zorder=0, linewidth=0)
    ax.axhline(0, color=GRAY, linewidth=0.8, zorder=1)
    ax.axhline(overall, color=SECONDARY, linewidth=1.0, zorder=1)
    ax.scatter(n[~off], err[~off], s=16, color=GRAY, edgecolor="white", linewidth=0.6, zorder=2)
    ax.scatter(n[off], err[off], s=20, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 200, 500], ["5", "10", "20", "50", "100", "200", "500"])
    ax.set_xlabel("Reviews of the movie (log scale)")
    ax.set_ylabel("Error of the movie's share, points")
    ax.set_ylim(-45, 25)
    side = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.01, overall, f"whole table {overall:+.1f}", ha="left", va="center", color=SECONDARY, fontsize=8, transform=side)
    ax.text(1.01, 10, "±10 band", ha="left", va="center", color=MUTED, fontsize=8, transform=side)
    ax.text(0.99, 0.04, f"blue: {off.sum()} of {len(err)} movies off by more than 10 points", ha="right", va="bottom", color=INK,
            fontsize=8.5, transform=ax.transAxes)
    ax.set_title("Jev is right on 91% of the reviews, yet a quarter of the movies are off by more than 10 points",
                 loc="left", fontsize=10, color=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    clean(ax)
    save(fig, "p16_propagation")


def fig_allocation(results: dict) -> None:
    """Labels each design needs to reach the query-aware design's loss at the primary budget, as a multiple of that budget."""
    rows = [(q, label) for q, label in QUERIES]
    designs = (("U", "uniform sample", GRAY, "o"), ("C", "model confidence", ORANGE, "o"),
               ("G", "group key only (the best earlier design)", BLUE, "o"))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(rows))[::-1]
    ax.axvline(1.0, color=SECONDARY, linewidth=1.0, zorder=1)
    ax.axvline(1.5, color=MUTED, linewidth=0.8, zorder=1)
    for yi, (q, _) in zip(y, rows):
        values = [results["primary"][q]["rho"][d] for d, *_ in designs]
        ax.plot([min(values + [1.0]), max(values)], [yi, yi], color=GRID, linewidth=1.0, zorder=1)
        for (d, label, colour, marker), v in zip(designs, values):
            ax.scatter(v, yi, s=34, color=colour, marker=marker, edgecolor="white", linewidth=0.8, zorder=3 if d == "G" else 2,
                       label=label if yi == y[0] else None)
        g = results["primary"][q]["rho"]["G"]
        left = 1.3 <= g <= 1.5  # keep the label off the pass-mark line
        ax.text(g / 1.05 if left else g * 1.05, yi + 0.3, f"{g:.2f}×", fontsize=7.5, color=INK, ha="right" if left else "left")
    ax.set_xscale("log")
    ticks = [1, 1.5, 2, 5, 10, 20, 40]
    ax.set_xticks(ticks, [f"{t:g}×" for t in ticks])
    ax.set_xlim(0.8, 45)
    ax.set_yticks(y, [label for _, label in rows])
    ax.set_ylim(-1.3, len(rows) - 0.5)
    ax.text(0.97, -1.0, "query-aware design", ha="right", va="center", color=SECONDARY, fontsize=7.5)
    ax.text(1.55, -1.0, "pass mark", ha="left", va="center", color=MUTED, fontsize=7.5)
    ax.set_xlabel("Labels needed for the same error, relative to the query-aware design (log scale)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, frameon=False, fontsize=8, handletextpad=0.2, columnspacing=1.2)
    ax.set_title("Choosing rows from the query pays off when a join reuses rows many times;\nfor groups, knowing the group key is enough",
                 loc="left", fontsize=10, color=INK, pad=22)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    clean(ax, keep=("bottom",))
    save(fig, "p16_allocation")


def fig_second_models(results: dict) -> None:
    """Left: how often a second model is wrong, overall and on the reviews Jev gets wrong. Right: a second model as the reference."""
    shared = results["shared"]
    models = (("D1-1.85k.opus", "Opus 5.5"), ("D1-5k.kev-9b", "Kev-9B"), ("D1-5k.deepseek.yesno", "DeepSeek V4.1 Flash"),
              ("D1-5k.kev-4b", "Kev-4B"), ("D1-5k.v4pro", "DeepSeek V4 Pro"), ("D1.luna", "GPT 6 Luna"), ("D1-5k.qwen", "Qwen 3.8 Flash"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"wspace": 0.75, "width_ratios": [1.15, 1]})
    y = np.arange(len(models))[::-1]
    for yi, (key, _) in zip(y, models):
        lo, hi = 100 * shared[key]["marginal"], 100 * shared[key]["given"]
        a.plot([lo, hi], [yi, yi], color=GRID, linewidth=1.6, zorder=1)
        a.scatter(lo, yi, s=30, color=GRAY, edgecolor="white", linewidth=0.8, zorder=2, label="on all reviews" if yi == y[0] else None)
        a.scatter(hi, yi, s=34, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3,
                  label="on the reviews Jev gets wrong" if yi == y[0] else None)
        a.text(hi + 2.5, yi, f"{hi:.0f}%", va="center", fontsize=7.5, color=INK)
    a.set_yticks(y, [label for _, label in models])
    a.set_xlim(0, 100)
    a.set_xlabel("Share of reviews the model gets wrong, %")
    a.set_title("When Jev is wrong, the others\nare wrong too", loc="left", fontsize=9.5, color=INK)
    a.legend(loc="upper center", frameon=False, fontsize=7.5, ncol=2, bbox_to_anchor=(0.4, -0.22), handletextpad=0.2)
    a.xaxis.grid(True, color=GRID, linewidth=0.6)
    a.set_axisbelow(True)
    clean(a, keep=("bottom",))
    refs = (("S4/D1.jev/Q1/QP.luna.B2000", "GPT 6 Luna"), ("S4/D1-5k.jev/Q1/QP.deepseek.yesno.B650", "DeepSeek V4.1 Flash"),
            ("S4/D1-5k.jev/Q1/QP.kev-9b.B650", "Kev-9B"), ("S4/D1-1.85k.jev/Q1/QP.opus.B241", "Opus 5.5"))
    s4 = results["S4"]
    yb = np.arange(len(refs))[::-1]
    h = 0.34
    for yi, (key, _) in zip(yb, refs):
        own, human = 100 * s4[key]["own"], 100 * s4[key]["human"]
        b.barh(yi + h / 2 + 0.02, own, height=h, color=GRAY, linewidth=0, label="its own answer" if yi == yb[0] else None)
        b.barh(yi - h / 2 - 0.02, max(human, 0.6), height=h, color=BLUE, linewidth=0, label="the human truth" if yi == yb[0] else None)
        b.text(own + 2, yi + h / 2 + 0.02, f"{own:.0f}%", va="center", fontsize=7.5, color=INK)
        b.text(max(human, 0.6) + 2, yi - h / 2 - 0.02, f"{human:.1f}%", va="center", fontsize=7.5, color=INK)
    b.set_yticks(yb, [label for _, label in refs])
    b.set_xlim(0, 112)
    b.set_xticks([0, 50, 95])
    b.set_xlabel("95% intervals that cover it, %")
    b.set_title("With one as the reference, its intervals\ncover its own share, not the truth", loc="left", fontsize=9.5, color=INK)
    b.legend(loc="upper center", frameon=False, fontsize=7.5, ncol=2, bbox_to_anchor=(0.45, -0.22), handlelength=1.0)
    b.xaxis.grid(True, color=GRID, linewidth=0.6)
    b.set_axisbelow(True)
    clean(b, keep=("bottom",))
    save(fig, "p16_second_models")


def main() -> None:
    style()
    results = json.loads((D.RESULTS / "results.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    fig_propagation()
    fig_allocation(results)
    fig_second_models(results)
    print("\n".join(str(p) for p in sorted(FIG.glob("p16_*.en.png"))))


if __name__ == "__main__":
    sys.exit(main())
