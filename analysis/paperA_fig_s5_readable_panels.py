#!/usr/bin/env python3
"""Draw readable per-study-sector panels for Supplementary Figure S5.

The compact profile parquet and its manifest are produced from the canonical
external-volume data by ``paperA_export_fig_s5_profiles.py`` on ocean-worker.
Each output contains one accepted and one rejected OSTIA profile for DJF and
JJA. The fitted segments, candidate breakpoint, and the criterion responsible
for rejection are shown explicitly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
PAPERA = ROOT / "results"
PROFILES = ROOT / "data" / "outputs" / "uidx" / "diagnostics" / (
    "paperA_fig_s5_profiles.parquet"
)
MANIFEST = ROOT / "data" / "outputs" / "uidx" / "diagnostics" / (
    "paperA_fig_s5_manifest.csv"
)
OUT_DIR = PAPERA / "figures"

CELLS = (
    ("Cape Frio", "cape_frio"),
    ("Walvis Bay", "walvis_bay"),
    ("Luderitz", "luderitz"),
    ("Cape Columbine", "cape_columbine"),
    ("Cape Peninsula", "cape_peninsula"),
)
SEASONS = ("DJF", "JJA")
STATUSES = ("accepted", "rejected")

OBSERVED = "#3f4a59"
INSHORE = "#d95f02"
OFFSHORE = "#1b9e77"
BREAKPOINT = "#6a3d9a"


def fitted_segments(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_min = 0.0
    x_max = float(row["line_length_km"])
    xb = float(row["x_b_km"])
    alpha = float(row["alpha_inshore"]) - 273.15
    beta_in = float(row["beta_inshore"])
    beta_off = float(row["beta_offshore"])

    x_in = np.linspace(x_min, xb, 160)
    y_in = alpha + beta_in * x_in
    x_off = np.linspace(xb, x_max, 240)
    y_off = alpha + beta_in * xb + beta_off * (x_off - xb)
    return x_in, y_in, x_off, y_off


def panel_annotation(row: pd.Series) -> str:
    ratio = abs(float(row["beta_inshore"]) / float(row["beta_offshore"]))
    delta_aic = float(row["delta_aic"])
    if row["profile_status"] == "accepted":
        return rf"Accepted: $\Delta$AIC = {delta_aic:.1f}; slope ratio = {ratio:.2f}"
    return f"Rejected: slope ratio = {ratio:.2f} < 2"


def draw_cell(profiles: pd.DataFrame, manifest: pd.DataFrame, cell: str, slug: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.3), sharex=True, sharey=True)
    display_cell = "Lüderitz" if cell == "Luderitz" else cell
    fig.suptitle(display_cell, fontsize=15, fontweight="bold", y=0.975)

    cell_profiles = profiles[profiles["cell"] == cell]
    finite = cell_profiles["sst_c"].dropna()
    y_pad = max(0.35, 0.08 * (finite.max() - finite.min()))
    y_limits = (finite.min() - y_pad, finite.max() + y_pad)

    for row_index, season in enumerate(SEASONS):
        for col_index, status in enumerate(STATUSES):
            ax = axes[row_index, col_index]
            key = manifest[
                (manifest["cell"] == cell)
                & (manifest["season"] == season)
                & (manifest["profile_status"] == status)
            ].iloc[0]
            profile = cell_profiles[
                (cell_profiles["season"] == season)
                & (cell_profiles["profile_status"] == status)
                & (cell_profiles["date"] == key["date"])
                & (cell_profiles["col_id"] == int(key["col_id"]))
            ].sort_values("distance_km")
            fit = profile.iloc[0]

            ax.plot(
                profile["distance_km"],
                profile["sst_c"],
                color=OBSERVED,
                linewidth=1.15,
                alpha=0.78,
                zorder=1,
            )
            ax.scatter(
                profile["distance_km"],
                profile["sst_c"],
                color=OBSERVED,
                s=5,
                alpha=0.32,
                linewidths=0,
                zorder=2,
            )

            x_in, y_in, x_off, y_off = fitted_segments(fit)
            ax.plot(x_in, y_in, color=INSHORE, linewidth=2.6, zorder=3)
            ax.plot(x_off, y_off, color=OFFSHORE, linewidth=2.6, zorder=3)

            xb = float(fit["x_b_km"])
            ax.axvline(xb, color=BREAKPOINT, linestyle=(0, (5, 3)), linewidth=1.7, zorder=0)
            xb_prefix = "$x_b$" if status == "accepted" else "candidate $x_b$"
            ax.annotate(
                f"{xb_prefix} = {xb:.0f} km",
                xy=(xb, 0.985),
                xycoords=("data", "axes fraction"),
                xytext=(5, -4),
                textcoords="offset points",
                va="top",
                ha="left",
                color=BREAKPOINT,
                fontsize=9.2,
                fontweight="bold",
            )

            ax.text(
                0.025,
                0.055,
                panel_annotation(fit),
                transform=ax.transAxes,
                fontsize=8.7,
                ha="left",
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.94, "pad": 4},
            )
            ax.set_title(
                f"{season}: {status} ({key['date']}, transect {int(key['col_id'])})",
                fontsize=10.2,
                pad=7,
            )
            ax.set_xlim(0, 565)
            ax.set_ylim(*y_limits)
            ax.grid(color="#d7d7d7", linewidth=0.55, alpha=0.65)
            ax.tick_params(labelsize=9)
            if col_index == 0:
                ax.set_ylabel(r"SST ($^\circ$C)", fontsize=10)
            if row_index == 1:
                ax.set_xlabel("Cross-shore distance (km)", fontsize=10)

    legend = [
        Line2D([0], [0], color=OBSERVED, linewidth=1.4, label="observed SST"),
        Line2D([0], [0], color=INSHORE, linewidth=2.6, label="fitted inshore segment"),
        Line2D([0], [0], color=OFFSHORE, linewidth=2.6, label="fitted offshore segment"),
        Line2D(
            [0], [0], color=BREAKPOINT, linewidth=1.7, linestyle=(0, (5, 3)),
            label="fitted or candidate breakpoint",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.002),
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.905, bottom=0.105, hspace=0.24, wspace=0.11)

    for suffix in ("pdf", "png"):
        out = OUT_DIR / f"fig_s5_representative_profiles_{slug}.{suffix}"
        fig.savefig(out, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        print(out)
    plt.close(fig)


def main() -> None:
    profiles = pd.read_parquet(PROFILES)
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    profiles["date"] = profiles["date"].astype(str)
    for cell, slug in CELLS:
        draw_cell(profiles, manifest, cell, slug)


if __name__ == "__main__":
    main()
