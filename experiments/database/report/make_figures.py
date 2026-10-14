"""The two figures of the report (PDF vector graphics for typesetting, PNG for checking). Reads results under out/ only and calls no model.

    python experiments/database/report/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from p3_aggregates import load  # noqa: E402
from p4_collect import PREDICATES, sentences  # noqa: E402
from paths import OUT, ROOT  # noqa: E402

FIG = HERE / "figures"
# Colours come from the dataviz reference palette (light surface); the blue and orange were checked with its validation script
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#c3c2b7"
INK, SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
FONTS = {"en": ["Helvetica Neue", "Arial"]}

TEXT = {
    "en": {
        "cost": "Total cost, US$", "time": "Total time, seconds",
        "systems": ["Jev (this work)", "GPT 6 Luna (this work)", "BigQuery (published)", "LOTUS (published)", "Palimpzest (published)"],
        "x": "Share of rows escalated, least-sure first", "y": "Share of Jev's errors covered",
        "matching": "Record matching (6 data sets)", "movie": "Movie reviews", "mams": "Restaurant reviews\n(mean of 21 predicates)", "random": "rows picked at random",
        "other": "Sentiment and aspect predicates (2 data sets)",
    },
}


def style(lang: str) -> None:
    plt.rcParams.update({"font.family": FONTS[lang], "font.size": 9, "pdf.fonttype": 42, "axes.unicode_minus": False,
                         "text.color": INK, "axes.labelcolor": SECONDARY, "xtick.color": MUTED, "ytick.color": SECONDARY,
                         "axes.edgecolor": GRAY, "axes.linewidth": 0.6})


def save(fig: plt.Figure, name: str, lang: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{lang}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_cost_time(lang: str) -> None:
    """Total cost and time of the ten queries. One panel per quantity, no shared axis. The focus is Jev; the rest are grey references."""
    t = TEXT[lang]
    rows = json.loads((OUT / "p1/scores.json").read_text())
    mid = lambda xs: sorted(xs)[len(xs) // 2]
    total = lambda mode, field: sum(mid([r[field] for r in rows if r["mode"] == mode and r["query"] == f"Q{i}"]) for i in range(1, 11))
    cost = [total("jev", "cost"), total("llm", "cost"), 5.4740, 5.9404, 8.2258]
    secs = [total("jev", "seconds"), total("llm", "seconds"), 401.3, 1544.1, 1399.8]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2), gridspec_kw={"wspace": 0.12})
    for ax, values, label, fmt in ((axes[0], cost, t["cost"], "{:.2f}"), (axes[1], secs, t["time"], "{:,.0f}")):
        y = np.arange(len(values))[::-1]
        ax.barh(y, values, height=0.5, color=[BLUE] + [GRAY] * 4, linewidth=0)
        for yi, v in zip(y, values):
            ax.text(v + max(values) * 0.015, yi, fmt.format(v), va="center", ha="left", color=INK, fontsize=8.5)
        ax.set_xlim(0, max(values) * 1.16)
        ax.set_yticks(y, t["systems"] if ax is axes[0] else [""] * len(y))
        ax.set_xlabel(label)
        ax.tick_params(length=0)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
    save(fig, "cost_time", lang)


def curve(wrong: np.ndarray, unsure_first: np.ndarray, grid: np.ndarray) -> np.ndarray:
    cum = np.concatenate([[0], np.cumsum(wrong[unsure_first])]) / max(wrong.sum(), 1)
    return cum[np.round(grid * len(wrong)).astype(int)]


def fig_concentration(lang: str) -> None:
    """How concentrated Jev's errors are among its least certain rows. Two task types in two colours (colour follows task type); the diagonal is picking rows at random."""
    t = TEXT[lang]
    grid = np.linspace(0, 0.6, 121)
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.plot(grid, grid, color=GRAY, linewidth=1)
    ax.text(0.585, 0.545, t["random"], color=MUTED, fontsize=8, ha="right", va="top", rotation=24)
    for path in ("entity_match/abt-buy.test", "entity_match/walmart-amazon.test", "entity_match/amazon-google.test", "entity_match/dblp-acm.test", "table_match/abt-buy", "table_match/dblp-scholar"):
        rows = [json.loads(line) for line in (ROOT / f"out/{path}.jev-1.13.0.jsonl").read_text().splitlines()]
        p, y = np.array([r["p"] for r in rows]), np.array([r["label"] for r in rows])
        ax.plot(grid, curve((p >= 0.5) != (y == 1), np.argsort(np.abs(p - 0.5), kind="stable"), grid), color=BLUE, linewidth=1.3, alpha=0.75)
    data = load()
    movie = curve((data["p"] >= 0.5) != (data["y"] == 1), np.argsort(np.abs(data["p"] - 0.5), kind="stable"), grid)
    rows = sentences()
    single = {r["sid"]: r["p"] for r in map(json.loads, (OUT / "p4/mams.jev.jsonl").read_text().splitlines())}
    per = []
    for k in PREDICATES:
        p, y = np.array([single[r["sid"]][k] for r in rows]), np.array([r["truth"][k] for r in rows])
        per.append(curve((p >= 0.5) != (y == 1), np.argsort(np.abs(p - 0.5), kind="stable"), grid))
    mams = np.mean(per, axis=0)
    ax.plot(grid, movie, color=ORANGE, linewidth=2)
    ax.plot(grid, mams, color=ORANGE, linewidth=2)
    # Each of the three groups of lines gets a direct label at its right end (in the text colour, not the line colour); the legend stays as the key from colour to task type
    for y_at, text in ((1.035, t["matching"]), (movie[-1] - 0.012, t["movie"]), (mams[-1] - 0.03, t["mams"])):
        ax.text(0.612, y_at, text, color=INK, fontsize=8.5, ha="left", va="center", clip_on=False, linespacing=1.15)
    ax.plot([], [], color=BLUE, linewidth=1.6, label=t["matching"])
    ax.plot([], [], color=ORANGE, linewidth=2, label=t["other"])
    ax.legend(loc="lower right", frameon=False, fontsize=8, handlelength=1.6, labelcolor=SECONDARY)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ["0", "10%", "20%", "30%", "40%", "50%", "60%"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(t["x"])
    ax.set_ylabel(t["y"])
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, "error_concentration", lang)


if __name__ == "__main__":
    for lang in ("en",):
        style(lang)
        fig_cost_time(lang)
        fig_concentration(lang)
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))
