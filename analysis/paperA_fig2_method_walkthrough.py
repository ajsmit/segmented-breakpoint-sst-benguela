#!/usr/bin/env python3
"""Create the illustrative five-panel breakpoint-method walk-through.

Outputs
-------
results/figures/fig2_method_walkthrough_v2.pdf
results/figures/fig2_method_walkthrough_v2.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperA_common as P


OUT_PDF = P.PAPERA_FIGURES / "fig2_method_walkthrough_v2.pdf"
OUT_PNG = P.PAPERA_FIGURES / "fig2_method_walkthrough_v2.png"
COOL = "#2860a5"
WARM = "#c85a37"
ACCEPT = "#23784b"
REJECT = "#aaaaaa"

DISTANCE = np.arange(0, 501, 25)
SST = np.array(
    [
        13.05, 13.44, 14.12, 14.50, 15.18, 15.55, 16.22,
        16.60, 16.90, 16.82, 17.02, 16.95, 17.15, 17.08,
        17.25, 17.20, 17.40, 17.32, 17.50, 17.45, 17.62,
    ]
)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, weight="bold", fontsize=11)


def main() -> None:
    fig = plt.figure(figsize=(13, 6.8), layout="constrained")
    grid = fig.add_gridspec(2, 6, height_ratios=[1, 1.08])
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 0:3]),
        fig.add_subplot(grid[1, 3:6]),
    ]

    # (a) One transect in an illustrative SST field.
    ax = axes[0]
    x = np.linspace(0, 500, 220)
    y = np.linspace(0, 1, 80)
    field = 13.0 + 4.6 * (1 - np.exp(-x / 150))[None, :] + 0.15 * np.sin(4 * np.pi * y)[:, None]
    ax.imshow(field, extent=[0, 500, 0, 1], origin="lower", aspect="auto", cmap="coolwarm")
    coast_y = 0.12 + 0.05 * np.sin(np.linspace(0, 3 * np.pi, 220))
    ax.fill_between(x, 0, coast_y, color="0.75", zorder=3)
    ax.plot(x, coast_y, color="0.35", lw=0.8, zorder=4)
    transect_y = 0.55
    ax.add_patch(FancyArrowPatch((8, transect_y), (492, transect_y), arrowstyle="-|>", mutation_scale=10, lw=1.2, color="black"))
    sample_x = np.linspace(25, 475, 13)
    ax.scatter(sample_x, np.full_like(sample_x, transect_y), s=12, color="black", zorder=5)
    ax.text(250, 0.61, "one transect", ha="center", va="bottom", fontsize=8)
    ax.set(xlim=(0, 500), ylim=(0, 1), xticks=[0, 250, 500], yticks=[], xlabel="distance from coast (km)")
    ax.set_title("One day, one transect", fontsize=10)
    panel_label(ax, "a")

    # (b) Sampled cross-shore temperatures.
    ax = axes[1]
    ax.scatter(DISTANCE, SST, s=15, color="0.25")
    ax.set(xlim=(-15, 515), ylim=(12.6, 18.1), xticks=[0, 250, 500], yticks=[13, 15, 17],
           xlabel="distance from coast (km)", ylabel=r"SST ($^\circ$C)")
    ax.set_title("Temperatures sampled", fontsize=10)
    panel_label(ax, "b")

    # (c) Candidate-grid search.
    ax = axes[2]
    candidates = np.arange(20, 481, 10)
    rss = 0.42 + ((candidates - 180) / 155) ** 2 + 0.06 * np.sin(candidates / 42)
    ax.axvspan(0, 10, color="0.9")
    ax.axvspan(400, 500, color="0.9")
    ax.plot(candidates, rss, color="0.2", lw=1.2)
    best = int(np.argmin(rss))
    ax.scatter(candidates[best], rss[best], s=26, color=WARM, zorder=3)
    ax.annotate("best fit", (candidates[best], rss[best]), xytext=(105, 0.72),
                textcoords="data", fontsize=8, color=WARM,
                arrowprops={"arrowstyle": "->", "color": WARM, "lw": 0.8})
    ax.text(450, 1.7, "excluded", rotation=90, ha="center", va="center", color="0.45", fontsize=7)
    ax.set(xlim=(0, 500), ylim=(0, 4.5), xticks=[0, 250, 500], yticks=[],
           xlabel="candidate breakpoint (km)", ylabel="residual sum of squares")
    ax.set_title("Every position is tried", fontsize=10)
    panel_label(ax, "c")

    # (d) Accepted fit and derived quantities.
    ax = axes[3]
    xb = 180
    inshore_x = np.array([0, xb])
    inshore_fit = np.array([13.0, 16.78])
    offshore_x = np.array([xb, 500])
    offshore_fit = np.array([16.78, 17.58])
    baseline = 17.18
    ax.fill_between(inshore_x, inshore_fit, baseline, color=COOL, alpha=0.20)
    ax.axhline(baseline, color=WARM, ls="--", lw=1.0)
    ax.plot(inshore_x, inshore_fit, color="black", lw=1.3)
    ax.plot(offshore_x, offshore_fit, color="black", lw=1.3)
    ax.scatter(DISTANCE, SST, s=10, color="0.4", zorder=3)
    ax.axvline(xb, color="0.2", ls=":", lw=0.9)
    for d0 in (50, 100):
        ax.axvline(d0, color="0.55", ls="-.", lw=0.7)
    ax.text(24, 15.6, r"$\bar{D}_\mathrm{in}$", color=COOL, fontsize=9)
    ax.text(488, baseline + 0.08, "fitted offshore baseline", color=WARM, ha="right", fontsize=8)
    ax.text(205, 15.25, r"slope $\beta_\mathrm{in}$", fontsize=8)
    ax.set(xlim=(-15, 515), ylim=(12.6, 18.25), xticks=[50, 100, 180, 300, 500],
           xticklabels=["50", "100", r"$x_b$", "300", "500"], yticks=[13, 15, 17],
           xlabel="distance from coast (km)", ylabel=r"SST ($^\circ$C)")
    ax.set_title("Accepted fit and fixed references", fontsize=10)
    panel_label(ax, "d")

    # (e) Study-sector daily accepted-profile frequency.
    ax = axes[4]
    ax.add_patch(Rectangle((0, 0), 10, 0.8, color="0.85"))
    accepted = [0.5, 1.3, 2.1, 2.9, 4.3, 5.1, 5.9, 7.3, 8.1, 9.5]
    rejected = [3.6, 6.6, 8.8]
    for xpos in accepted:
        ax.add_patch(FancyArrowPatch((xpos, 0.8), (xpos, 3.8), arrowstyle="-|>", mutation_scale=8, lw=1.2, color=ACCEPT))
    for xpos in rejected:
        ax.add_patch(FancyArrowPatch((xpos, 0.8), (xpos, 3.1), arrowstyle="-|>", mutation_scale=8, lw=1.1, color=REJECT, linestyle="--"))
    ax.text(0.35, 4.45, "accepted: 10", color=ACCEPT, fontsize=9)
    ax.text(3.45, 4.45, "rejected: 3", color="0.45", fontsize=9)
    ax.text(0.35, 5.1, r"$p_\mathrm{accept}=10/13=0.77\geq0.5$", fontsize=10)
    ax.text(0.35, 5.55, "the day meets the SST criterion", fontsize=9)
    ax.text(0.35, 0.35, "land", color="0.35", fontsize=8)
    ax.set(xlim=(0, 10), ylim=(0, 6), xticks=[], yticks=[])
    ax.set_title("The study sector on that day", fontsize=10)
    panel_label(ax, "e")

    for ax in axes:
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"[output] {OUT_PDF}")
    print(f"[output] {OUT_PNG}")


if __name__ == "__main__":
    main()
