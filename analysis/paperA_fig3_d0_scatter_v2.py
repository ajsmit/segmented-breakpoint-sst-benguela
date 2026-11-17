#!/usr/bin/env python3
"""Revised inshore-drawdown versus fixed-distance scatter figure.

The point order is shuffled deterministically, markers are smaller than in the
submitted figure, and one light density contour is drawn for each study sector.

Outputs
-------
results/figures/fig3_uibr_vs_uisst_scatter_v2.pdf
results/figures/fig3_uibr_vs_uisst_scatter_v2.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperA_common as P


OUT_PDF = P.PAPERA_FIGURES / "fig3_uibr_vs_uisst_scatter_v2.pdf"
OUT_PNG = P.PAPERA_FIGURES / "fig3_uibr_vs_uisst_scatter_v2.png"
AR1_PROFILES = P.ROOT / "data/outputs/uidx/paperA_ar1/fixed_distance_profiles.parquet"

D0_VALUES = [50, 100, 200, 500]
CELL_COLOURS = {
    "Cape Frio": "#9b2c2c",
    "Walvis Bay": "#1f3b5b",
    "Luderitz": "#2b8c4a",
    "Cape Columbine": "#8b4a8e",
    "Cape Peninsula": "#d4731f",
}
PLOT_LIMITS = (-2.0, 10.0)


def build_intermediate() -> pd.DataFrame:
    columns = ["cell", "product", "date", "D_in"] + [
        f"ui_sst_d{distance}" for distance in D0_VALUES
    ]
    profiles = pd.read_parquet(AR1_PROFILES, columns=columns)
    profiles = profiles[profiles["product"] == "OSTIA"].drop(columns="product")
    return (
        profiles.groupby(["cell", "date"])
        .agg(
            **{
                f"ui_d{distance}": (f"ui_sst_d{distance}", "median")
                for distance in D0_VALUES
            },
            D_in_med=("D_in", "median"),
        )
        .reset_index()
        .dropna(subset=["D_in_med"])
    )


def smooth_histogram(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a lightly smoothed 2-D histogram without a SciPy dependency."""
    edges = np.linspace(PLOT_LIMITS[0], PLOT_LIMITS[1], 55)
    density, x_edges, y_edges = np.histogram2d(x, y, bins=[edges, edges])
    kernel_x = np.arange(-4, 5, dtype=float)
    kernel = np.exp(-0.5 * (kernel_x / 1.5) ** 2)
    kernel /= kernel.sum()
    density = np.apply_along_axis(lambda values: np.convolve(values, kernel, mode="same"), 0, density)
    density = np.apply_along_axis(lambda values: np.convolve(values, kernel, mode="same"), 1, density)
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_centres, y_centres, density.T


def add_density_contour(ax: plt.Axes, x: np.ndarray, y: np.ndarray, colour: str) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return
    x_centres, y_centres, density = smooth_histogram(x[mask], y[mask])
    positive = density[density > 0]
    if positive.size == 0:
        return
    level = float(np.quantile(positive, 0.72))
    if not 0 < level < float(density.max()):
        return
    ax.contour(x_centres, y_centres, density, levels=[level], colors=[colour],
               linewidths=0.65, alpha=0.75, zorder=2)


def main() -> None:
    cell_day = build_intermediate()
    print(f"[loaded] {len(cell_day):,} OSTIA (cell, date) median pairs")

    fig, axes = plt.subplots(1, len(D0_VALUES), figsize=(13, 3.5), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.19, top=0.76, wspace=0.12)

    for panel_number, (ax, distance) in enumerate(zip(axes, D0_VALUES)):
        x_column = f"ui_d{distance}"
        plotted = cell_day[["cell", x_column, "D_in_med"]].dropna()
        plotted = plotted.sample(frac=1, random_state=20260820 + panel_number)
        point_colours = plotted["cell"].map(CELL_COLOURS)
        ax.scatter(plotted[x_column], plotted["D_in_med"], s=2.2, alpha=0.25,
                   c=point_colours, edgecolors="none", zorder=1)

        for cell, colour in CELL_COLOURS.items():
            subset = cell_day[cell_day["cell"] == cell]
            add_density_contour(
                ax,
                subset[x_column].to_numpy(dtype=float),
                subset["D_in_med"].to_numpy(dtype=float),
                colour,
            )

        x = cell_day[x_column].to_numpy(dtype=float)
        y = cell_day["D_in_med"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        correlation = float(np.corrcoef(x[mask], y[mask])[0, 1]) if mask.sum() >= 3 else np.nan
        ax.text(
            0.04,
            0.94,
            fr"$r = {correlation:+.2f}$",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#888", "alpha": 0.88},
            zorder=4,
        )
        ax.plot(PLOT_LIMITS, PLOT_LIMITS, color="#999999", lw=0.6, zorder=0)
        ax.set_xlim(*PLOT_LIMITS)
        ax.set_ylim(*PLOT_LIMITS)
        ax.set_title(fr"$d_0 = {distance}$ km", fontsize=10, pad=6)
        ax.set_xlabel(fr"$T({distance})-T(0)$ ($^\circ$C)")
        ax.grid(alpha=0.25, lw=0.4)

    axes[0].set_ylabel(r"$\bar{D}_\mathrm{in}$ ($^\circ$C)")
    legend_handles = [
        plt.Line2D([0], [0], marker="o", markersize=5, lw=0,
                   markerfacecolor=colour, markeredgecolor="none", label=cell)
        for cell, colour in CELL_COLOURS.items()
    ]
    fig.legend(handles=legend_handles, ncol=5, loc="upper center",
               bbox_to_anchor=(0.53, 0.98), frameon=False, fontsize=9)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"[output] {OUT_PDF}")
    print(f"[output] {OUT_PNG}")


if __name__ == "__main__":
    main()
