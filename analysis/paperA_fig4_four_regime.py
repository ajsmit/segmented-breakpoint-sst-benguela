#!/usr/bin/env python3
"""paperA_fig4_four_regime.py -- rebuild Figure 4 from the AR(1)-corrected
four-regime fractions.

Five study sectors x two fine-resolution products x four seasons stacked-bar grid.

Outputs: results/figures/fig4_four_regime_stacked_bar.{pdf,png}

Run from the release repository root:
    python3 analysis/paperA_fig4_four_regime.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperA_common as P

SRC = P.PAPERA_TABLES / "fig4_four_regime_stacked_bar.csv"
OUT_PDF = P.PAPERA_FIGURES / "fig4_four_regime_stacked_bar.pdf"
OUT_PNG = P.PAPERA_FIGURES / "fig4_four_regime_stacked_bar.png"

REGIME_COLOURS = {
    "concordant_active": "#3a7d3f",  # green
    "concordant_quiet":  "#dad6c8",  # warm beige
    "sst_only":          "#c97f3a",  # ochre
    "wind_only":         "#3b6b8c",  # steel blue
}
REGIME_LABELS = {
    "concordant_active": "both criteria met",
    "concordant_quiet":  "neither criterion met",
    "sst_only":          "SST criterion only",
    "wind_only":         "wind criterion only",
}
REGIME_ORDER = ["concordant_active", "sst_only", "wind_only", "concordant_quiet"]

CELLS_ORDERED   = ["Cape Frio", "Walvis Bay", "Luderitz", "Cape Columbine", "Cape Peninsula"]
PRODUCTS_ORDERED = ["OSTIA", "OSPO"]
SEASONS_ORDERED  = ["DJF", "MAM", "JJA", "SON"]


def main() -> None:
    df = pd.read_csv(SRC)
    n_cells = len(CELLS_ORDERED)
    n_seasons = len(SEASONS_ORDERED)
    fig, axes = plt.subplots(n_seasons, n_cells,
                              figsize=(2.7 * n_cells, 2.0 * n_seasons),
                              constrained_layout=False,
                              sharex=True, sharey=True)
    for j, cell in enumerate(CELLS_ORDERED):
        for i, season in enumerate(SEASONS_ORDERED):
            ax = axes[i, j]
            sub = df[(df["cell"] == cell) & (df["season"] == season)]
            for product_idx, product in enumerate(PRODUCTS_ORDERED):
                row = sub[sub["product"] == product]
                if row.empty:
                    continue
                r = row.iloc[0]
                bottom = 0.0
                for regime in REGIME_ORDER:
                    val = float(r[regime])
                    ax.bar(product_idx, val, bottom=bottom, width=0.78,
                           color=REGIME_COLOURS[regime], edgecolor="#222", lw=0.4)
                    bottom += val
            ax.set_xticks(range(len(PRODUCTS_ORDERED)))
            ax.set_xticklabels([p.replace("_", "\n") for p in PRODUCTS_ORDERED],
                               fontsize=7)
            ax.set_ylim(0, 1.0)
            if j == 0:
                ax.set_ylabel(f"{season}\nfraction of days", fontsize=8)
            if i == 0:
                ax.set_title(cell, fontsize=10)
            ax.grid(axis="y", alpha=0.3, lw=0.4)
    legend_handles = [Patch(facecolor=REGIME_COLOURS[r], edgecolor="#222",
                            label=REGIME_LABELS[r]) for r in REGIME_ORDER]
    fig.legend(handles=legend_handles, ncol=4, loc="upper center",
                bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.7, w_pad=0.4)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"[output] {OUT_PDF}")
    print(f"[output] {OUT_PNG}")

    # Verification: sums per row equal 1.0 to floating-point
    sums = df[["concordant_active", "concordant_quiet", "sst_only", "wind_only"]].sum(axis=1)
    max_err = float(np.max(np.abs(sums - 1.0)))
    print(f"\n[verify] max |row sum - 1.0|: {max_err:.2e}")
    # CSV is written with 4-decimal rounding; allow up to 2e-4 absorption.
    assert max_err < 1e-3, "regime fractions do not sum to 1 within CSV rounding"


if __name__ == "__main__":
    main()
