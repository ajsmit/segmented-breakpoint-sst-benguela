#!/usr/bin/env python3
"""Create the revised Figure S3 with an axis matched to observed kappa."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperA_common as P


INPUT = P.PAPERA_TABLES / "table_s5_paired_classifier_ar1.csv"
OUT_PDF = P.PAPERA_FIGURES / "fig_s3_kappa_wind_marginal_v2.pdf"
OUT_PNG = P.PAPERA_FIGURES / "fig_s3_kappa_wind_marginal_v2.png"

CELL_COLOURS = {
    "Cape Frio": "#1b9e77",
    "Walvis Bay": "#d95f02",
    "Luderitz": "#7570b3",
    "Cape Columbine": "#e7298a",
    "Cape Peninsula": "#66a61e",
}
PRODUCT_MARKERS = {"OSTIA": "o", "OSPO": "s", "AVHRR": "^", "MW_OI": "x"}
COVERAGE_ARTEFACTS = {("Cape Frio", "AVHRR"), ("Cape Peninsula", "AVHRR")}


def main() -> None:
    data = pd.read_csv(INPUT)
    observed_max = float(data["cohen_kappa"].max())

    fig, ax = plt.subplots(figsize=(8.2, 5.4), layout="constrained")
    ax.axvspan(0.85, 1.0, color="0.94", zorder=0)
    ax.axhline(0, color="0.35", lw=0.7, zorder=0)

    for (cell, product), subset in data.groupby(["cell", "product"], sort=False):
        colour = CELL_COLOURS[cell]
        marker = PRODUCT_MARKERS[product]
        open_marker = (cell, product) in COVERAGE_ARTEFACTS
        kwargs = {"s": 58 if open_marker else 38, "marker": marker, "linewidths": 1.0, "zorder": 2}
        if marker == "x":
            kwargs.update(color=colour)
        elif open_marker:
            kwargs.update(facecolors="white", edgecolors=colour)
        else:
            kwargs.update(facecolors=colour, edgecolors="white")
        ax.scatter(subset["prevalence_wind"], subset["cohen_kappa"], **kwargs)

    ax.text(0.985, 0.137, "high positive-transport\nfrequency", ha="right", va="top", fontsize=9)
    ax.text(0.56, 0.145, f"observed maximum = {observed_max:.2f}", ha="center", va="top", fontsize=9)
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.08, 0.15),
           xlabel="Positive offshore Ekman-transport frequency", ylabel=r"Cohen's $\kappa$")
    ax.grid(alpha=0.25, lw=0.5)

    product_handles = [
        plt.Line2D([0], [0], marker=marker, color="0.2", lw=0, markersize=6, label=product)
        for product, marker in PRODUCT_MARKERS.items()
    ]
    cell_handles = [
        plt.Line2D([0], [0], marker="o", color=colour, markerfacecolor=colour,
                   lw=0, markersize=6, label=cell)
        for cell, colour in CELL_COLOURS.items()
    ]
    product_legend = ax.legend(handles=product_handles, title="Product", frameon=False,
                               loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.add_artist(product_legend)
    ax.legend(handles=cell_handles, title="Study sector", frameon=False,
              loc="center left", bbox_to_anchor=(1.01, 0.5))

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"[data] maximum Cohen's kappa = {observed_max:.6f}")
    print(f"[output] {OUT_PDF}")
    print(f"[output] {OUT_PNG}")


if __name__ == "__main__":
    main()
