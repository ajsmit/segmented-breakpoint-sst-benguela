#!/usr/bin/env python3
"""paperA_sensitivity_figures.py -- Stage 4 Task 4.2: produce per-axis
small-multiples figures and tables.

For each of the 6 sensitivity axes, draw a 3-row × 1-column figure
showing the three headline statistics as a function of the axis value,
with one line per (cell, product) restricted to OSTIA / OSPO (the two
fine-resolution products that the manuscript's findings rest on; AVHRR /
MW_OI added as faint background lines).

Baseline value is marked with a vertical guide line on each panel.

For axis 5 (pixel_rule) and axis 6 (p_active_threshold), the categorical
nature gives a 2-tick or 3-tick discrete x-axis.

Outputs (per axis):
  results/figures/fig_s_sens_<axis>.pdf
  results/figures/fig_s_sens_<axis>.png
  results/tables/table_s_sens_<axis>.csv

Run from the release repository root:
    python3 analysis/paperA_sensitivity_figures.py
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
import paperA_sensitivity_grid as G


SRC_PARQUET = P.ROOT / "data/outputs/uidx/paperA_sensitivity/grid_results.parquet"

CELL_COLOURS = {
    "Cape Frio":       "#9b2c2c",
    "Walvis Bay":      "#1f3b5b",
    "Luderitz":        "#2b8c4a",
    "Cape Columbine":  "#8b4a8e",
    "Cape Peninsula":  "#d4731f",
}
PRODUCT_STYLE = {
    "OSTIA": dict(ls="-",  lw=1.6, alpha=1.00),
    "OSPO":  dict(ls="--", lw=1.4, alpha=0.90),
    "AVHRR": dict(ls=":",  lw=1.0, alpha=0.50),
    "MW_OI": dict(ls="-.", lw=1.0, alpha=0.40),
}
HEADLINE_LABELS = {
    "p_active":         r"$p_\mathrm{active}$",
    "r_Ain_UIsst_d50":  r"$r\,(\bar D_\mathrm{in}, UI_\mathrm{SST}^{(50)})$",
    "P_d0_lt_xb_d50":   r"$P(d_0\!<\!x_b)$  at  $d_0\!=\!50$ km",
}
HEADLINE_YLIMS = {
    "p_active":         (-0.05, 1.05),
    "r_Ain_UIsst_d50":  (-1.0, 1.0),
    "P_d0_lt_xb_d50":   (-0.05, 1.05),
}
AXIS_XLABELS = {
    "delta_aic_bar":      r"$\Delta\mathrm{AIC}$ bar",
    "slope_ratio":        r"slope ratio $|\beta_\mathrm{in}|/|\beta_\mathrm{off}|$",
    "min_offshore_km":    "min offshore length (km)",
    "edge_buffer_km":     "edge buffer (km)",
    "pixel_rule":         "pixel-thinning rule",
    "p_active_threshold": r"$p_\mathrm{active}$ threshold",
}


def draw_axis_panel(axis: str, df: pd.DataFrame) -> None:
    values = G.AXIS_DEFS[axis]
    baseline = G.AXIS_BASELINE[axis]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax_i, metric in enumerate(["p_active", "r_Ain_UIsst_d50", "P_d0_lt_xb_d50"]):
        ax = axes[ax_i]
        sub = df[df["axis"] == axis].copy()
        if axis in ("pixel_rule",):
            # Categorical x
            x_positions = {v: i for i, v in enumerate(values)}
            sub["x_pos"] = sub["axis_value"].map(x_positions)
            for (cell, product), g in sub.groupby(["cell", "product"]):
                if product not in ("OSTIA", "OSPO", "AVHRR", "MW_OI"):
                    continue
                g = g.sort_values("x_pos")
                ax.plot(g["x_pos"], g[metric],
                        color=CELL_COLOURS[cell],
                        marker="o", ms=4,
                        **PRODUCT_STYLE[product])
            ax.set_xticks(list(x_positions.values()))
            ax.set_xticklabels(list(x_positions.keys()), rotation=15)
        else:
            for (cell, product), g in sub.groupby(["cell", "product"]):
                if product not in ("OSTIA", "OSPO", "AVHRR", "MW_OI"):
                    continue
                g = g.copy()
                g["axis_value_float"] = g["axis_value"].astype(float)
                g = g.sort_values("axis_value_float")
                ax.plot(g["axis_value_float"], g[metric],
                        color=CELL_COLOURS[cell],
                        marker="o", ms=4,
                        **PRODUCT_STYLE[product])
            ax.axvline(float(baseline), color="#888", lw=0.7, ls="-")
            ax.text(float(baseline), ax.get_ylim()[1] * 0.93, " baseline",
                    fontsize=7, color="#666")
        ax.set_ylim(*HEADLINE_YLIMS[metric])
        if metric == "r_Ain_UIsst_d50":
            ax.axhline(0.0, color="#999", lw=0.5, ls="-")
        ax.set_xlabel(AXIS_XLABELS[axis])
        ax.set_ylabel(HEADLINE_LABELS[metric])
        ax.grid(alpha=0.3, lw=0.4)
    # Common legend
    legend_handles = []
    legend_labels  = []
    for cell, c in CELL_COLOURS.items():
        legend_handles.append(plt.Line2D([0], [0], color=c, lw=1.6, marker="o", ms=4))
        legend_labels.append(cell)
    for product, st in PRODUCT_STYLE.items():
        legend_handles.append(plt.Line2D([0], [0], color="black", **st))
        legend_labels.append(product.replace("_", " "))
    fig.legend(legend_handles, legend_labels,
               ncol=9, loc="upper center", bbox_to_anchor=(0.5, 1.07),
               fontsize=8, frameon=False)
    fig.suptitle(f"Sensitivity axis: {axis}", fontsize=11, y=1.13)
    out_pdf = P.PAPERA_ROOT / f"figures/fig_s_sens_{axis}.pdf"
    out_png = P.PAPERA_ROOT / f"figures/fig_s_sens_{axis}.png"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out_pdf.name}")
    # Per-axis CSV (wide format: rows = (cell, product), cols = axis_values × metrics)
    sub_csv = df[df["axis"] == axis].copy()
    out_csv = P.PAPERA_ROOT / f"tables/table_s_sens_{axis}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub_csv.sort_values(["cell", "product", "axis_value"]).to_csv(out_csv, index=False)
    print(f"  [csv] {out_csv.name}")


def main() -> None:
    df = pd.read_parquet(SRC_PARQUET)
    print(f"[loaded] {len(df):,} grid rows from {SRC_PARQUET}\n")
    for axis in G.AXIS_DEFS:
        draw_axis_panel(axis, df)
    print(f"\n[done] {len(G.AXIS_DEFS)} per-axis figures + tables in "
          f"{P.PAPERA_ROOT}/figures/ and {P.PAPERA_ROOT}/tables/")


if __name__ == "__main__":
    main()
