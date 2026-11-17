#!/usr/bin/env python3
"""paperA_fig3_xb_cdf.py -- per-cell ECDFs of x_b on the AR(1)-corrected
OSTIA and OSPO accepted-profile set, with d_0 reference distances marked.

5 panels (one per cell). OSTIA and OSPO ECDFs as solid + dashed lines.
Vertical reference lines at d_0 in {50, 100, 200, 500} km.
Outputs: results/figures/fig3_xb_cdf_literature_d0.{pdf,png}

Run from the release repository root:
    python3 analysis/paperA_fig3_xb_cdf.py
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

AR1_PROFILES = P.ROOT / "data/outputs/uidx/paperA_ar1/fixed_distance_profiles.parquet"
OUT_PDF = P.PAPERA_FIGURES / "fig3_xb_cdf_literature_d0.pdf"
OUT_PNG = P.PAPERA_FIGURES / "fig3_xb_cdf_literature_d0.png"

CELLS_ORDERED = ["Cape Frio", "Walvis Bay", "Luderitz", "Cape Columbine", "Cape Peninsula"]
D0_VALUES = [50, 100, 200, 500]
PRODUCT_STYLE = {"OSTIA": dict(ls="-",  lw=1.6, color="#1f3b5b"),
                 "OSPO":  dict(ls="--", lw=1.4, color="#9b2c2c")}


def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(x[np.isfinite(x)])
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def main() -> None:
    df = pd.read_parquet(AR1_PROFILES, columns=["cell", "product", "x_b_km"])
    df = df[df["product"].isin(PRODUCT_STYLE) & np.isfinite(df["x_b_km"])]
    fig, axes = plt.subplots(1, len(CELLS_ORDERED),
                              figsize=(3.0 * len(CELLS_ORDERED), 3.5),
                              constrained_layout=True, sharey=True)
    for ax, cell in zip(axes, CELLS_ORDERED):
        for product, style in PRODUCT_STYLE.items():
            sub = df[(df["cell"] == cell) & (df["product"] == product)]
            if sub.empty:
                continue
            xs, ys = ecdf(sub["x_b_km"].to_numpy(dtype=float))
            ax.plot(xs, ys, label=product, **style)
        for d0 in D0_VALUES:
            ax.axvline(d0, color="#888", lw=0.5, ls="-")
            ax.text(d0, 0.05, f"{d0}", fontsize=7, color="#666",
                    ha="center", rotation=0,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor="none", alpha=0.85))
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 1.02)
        ax.set_title(cell, fontsize=10)
        ax.set_xlabel(r"$x_b$ (km coast-distance)")
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel(r"empirical CDF of $x_b$")
    axes[-1].legend(fontsize=9, loc="lower right")
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"[output] {OUT_PDF}")
    print(f"[output] {OUT_PNG}")

    # Verification: CDFs reach 1.0 within cell window; d_0=50/100/200 on the
    # rising part for most cells; d_0=500 at the top.
    print("\n[verify] per-cell P(x_b > 50, 100, 200, 500 km) on OSTIA:")
    for cell in CELLS_ORDERED:
        sub = df[(df["cell"] == cell) & (df["product"] == "OSTIA")]
        if sub.empty:
            continue
        x = sub["x_b_km"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        print(f"  {cell:18s}: " + ", ".join(
            f"P(x_b > {d0})={float(np.mean(x > d0)):.3f}" for d0 in D0_VALUES))


if __name__ == "__main__":
    main()
