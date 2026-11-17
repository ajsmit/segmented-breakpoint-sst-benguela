#!/usr/bin/env python3
"""paperA_fig1_study_region.py -- Stage 8 Task 8.1: rebuild Figure 1
(study region with enlarged inset, CVD audit).

Renders the r2 mesh network map with:
  - Wong (2011) CVD-safe palette for the five study sectors.
  - An enlarged inset over Cape Columbine / Cape Peninsula (the sBUS).
  - Three CVD simulations (deuteranopia, protanopia, tritanopia)
    saved alongside the main figure for the CVD audit.

CVD simulation uses the Brettel et al. (1997) LMS-confusion-point
matrix; implemented in-script with no extra dependency beyond numpy
and matplotlib.

Outputs:
  results/figures/fig1_benguela_uc_mesh_map.{pdf,png}
  results/figures/cvd/fig1_deuteranopia.png
  results/figures/cvd/fig1_protanopia.png
  results/figures/cvd/fig1_tritanopia.png

Run from the release repository root:
    python3 analysis/paperA_fig1_study_region.py
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from shapely.geometry import LineString, MultiLineString, shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperA_common as P


MESH_DIR = P.mesh_root()
META = MESH_DIR / "mesh_boundary_match.csv"
CROSS = MESH_DIR / "mesh_cross_lines.geojson"
COASTLINE = P.ROOT / "data/derived/mesh_seeds/uc2d_v2/smoothed_coastline.geojson"

OUT_PDF = P.PAPERA_FIGURES / "fig1_benguela_uc_mesh_map.pdf"
OUT_PNG = P.PAPERA_FIGURES / "fig1_benguela_uc_mesh_map.png"
CVD_DIR = P.PAPERA_FIGURES / "cvd"

LON_MIN, LON_MAX = 5.5, 22.5
LAT_MIN, LAT_MAX = -37.0, -15.0

# Wong (2011) CVD-safe palette mapped to the 5 cells. These pairings are
# checked against the three CVD simulations to confirm the cells stay
# distinguishable.
CELL_COLOURS_CVD = {
    "Cape Frio":       "#E69F00",  # orange
    "Walvis Bay":      "#56B4E9",  # sky blue
    "Luderitz":        "#009E73",  # bluish green
    "Cape Columbine":  "#CC79A7",  # reddish purple
    "Cape Peninsula":  "#F0E442",  # yellow
}

CELLS = [
    dict(name="Cape Frio",       lat=-18.43, lon=12.02, dx=-1.6, dy=0.0),
    dict(name="Walvis Bay",      lat=-22.95, lon=14.50, dx=-2.0, dy=0.0),
    dict(name="Luderitz",        lat=-26.65, lon=15.16, dx=-2.4, dy=0.0),
    dict(name="Cape Columbine",  lat=-32.83, lon=17.85, dx=-3.6, dy=0.55),
    dict(name="Cape Peninsula",  lat=-34.20, lon=18.40, dx=-4.6, dy=-0.30),
]


# ---- CVD simulation matrices (LMS space, Brettel 1997) ----------------------
# sRGB -> LMS
_RGB2LMS = np.array([
    [0.31399022, 0.63951294, 0.04649755],
    [0.15537241, 0.75789446, 0.08670142],
    [0.01775239, 0.10944209, 0.87256922],
])
_LMS2RGB = np.linalg.inv(_RGB2LMS)
_CVD = {
    "deuteranopia": np.array([[1.0, 0.0,      0.0],
                                [0.494207, 0.0, 1.24827],
                                [0.0, 0.0, 1.0]]),
    "protanopia":   np.array([[0.0, 2.02344, -2.52581],
                                [0.0, 1.0,      0.0],
                                [0.0, 0.0,      1.0]]),
    "tritanopia":   np.array([[1.0, 0.0,        0.0],
                                [0.0, 1.0,        0.0],
                                [-0.395913, 0.801109, 0.0]]),
}


def simulate_cvd(rgb_image: np.ndarray, kind: str) -> np.ndarray:
    """Simulate a CVD on an HxWx3 sRGB array in [0, 1]."""
    flat = rgb_image.reshape(-1, 3).T
    lms = _RGB2LMS @ flat
    lms = _CVD[kind] @ lms
    out = _LMS2RGB @ lms
    return np.clip(out.T.reshape(rgb_image.shape), 0, 1)


# ---- Mesh load helpers ------------------------------------------------------
def load_coastline(path: Path) -> np.ndarray:
    with path.open() as f:
        geo = json.load(f)
    geom = shape(geo["features"][0]["geometry"])
    return np.asarray(geom.coords)


def load_cross_lines(path: Path) -> dict[int, np.ndarray]:
    with path.open() as f:
        geo = json.load(f)
    out = {}
    for feat in geo["features"]:
        col_id = int(feat["properties"]["col_id"])
        geom = shape(feat["geometry"])
        if isinstance(geom, LineString):
            out[col_id] = np.asarray(geom.coords)
    return out


def load_boundary_match(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [
            {"col_id": int(r["col_id"]),
             "coast_lon": float(r["coast_lon"]),
             "coast_lat": float(r["coast_lat"]),
             "line_length_km": float(r["line_length_km"])}
            for r in reader
        ]


def load_cell_membership() -> dict[int, str]:
    df_path = P.paperA_cell_transects_path()
    with df_path.open() as f:
        reader = csv.DictReader(f)
        return {int(r["col_id"]): r["cell"] for r in reader}


# ---- Plot -------------------------------------------------------------------
def draw_main_panel(ax: plt.Axes, cross_lines, coast, col_to_cell, boundary) -> None:
    ax.fill_between([LON_MIN, LON_MAX], LAT_MIN, LAT_MAX, color="#f6f7f8", zorder=0)
    # Coast
    ax.plot(coast[:, 0], coast[:, 1], color="#222", lw=0.9, zorder=4)
    # Transects, coloured by parent cell membership (or grey if not in a cell)
    for col_id, coords in cross_lines.items():
        cell = col_to_cell.get(col_id)
        if cell:
            ax.plot(coords[:, 0], coords[:, 1],
                    color=CELL_COLOURS_CVD[cell], lw=0.8, alpha=0.85, zorder=3)
        else:
            ax.plot(coords[:, 0], coords[:, 1], color="#cbd3da", lw=0.4, alpha=0.6, zorder=2)
    # Cell markers
    for c in CELLS:
        ax.plot(c["lon"], c["lat"], marker="^", ms=8,
                markerfacecolor=CELL_COLOURS_CVD[c["name"]],
                markeredgecolor="#222", lw=0, zorder=6)
        ax.annotate(
            c["name"],
            xy=(c["lon"], c["lat"]),
            xytext=(c["lon"] + c["dx"], c["lat"] + c["dy"]),
            fontsize=8, color="#222", weight="bold",
            ha="right" if c["dx"] < 0 else "left", va="center",
            arrowprops=dict(arrowstyle="-", color="#666", lw=0.6),
            zorder=7,
        )
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°S)")
    ax.set_title(
        f"Benguela Upwelling System study-sector mesh (n = {len(cross_lines)} transects)",
        fontsize=10,
    )
    # Inset region marker for the southern cells
    inset_box = Rectangle((17.4, -34.7), 1.8, 2.2, fill=False,
                           edgecolor="#222", lw=1.0, ls="-", zorder=8)
    ax.add_patch(inset_box)


def draw_inset_panel(ax: plt.Axes, cross_lines, coast, col_to_cell) -> None:
    LON0, LON1 = 17.4, 19.2
    LAT0, LAT1 = -34.7, -32.5
    ax.fill_between([LON0, LON1], LAT0, LAT1, color="#f6f7f8", zorder=0)
    ax.plot(coast[:, 0], coast[:, 1], color="#222", lw=1.2, zorder=4)
    for col_id, coords in cross_lines.items():
        cell = col_to_cell.get(col_id)
        if cell in ("Cape Columbine", "Cape Peninsula"):
            ax.plot(coords[:, 0], coords[:, 1],
                    color=CELL_COLOURS_CVD[cell], lw=1.2, alpha=0.9, zorder=3)
    for c in CELLS:
        if c["name"] in ("Cape Columbine", "Cape Peninsula"):
            ax.plot(c["lon"], c["lat"], marker="^", ms=10,
                    markerfacecolor=CELL_COLOURS_CVD[c["name"]],
                    markeredgecolor="#222", lw=0, zorder=6)
            ax.annotate(
                c["name"], xy=(c["lon"], c["lat"]),
                xytext=(c["lon"] - 0.3, c["lat"] + 0.15),
                fontsize=9, color="#222", weight="bold",
                ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color="#666", lw=0.6),
                zorder=7,
            )
    ax.set_xlim(LON0, LON1)
    ax.set_ylim(LAT0, LAT1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (°E)", fontsize=8)
    ax.set_ylabel("Latitude (°S)", fontsize=8)
    ax.set_title("Southern Benguela detail", fontsize=9)
    ax.tick_params(labelsize=7)


def main() -> None:
    coast = load_coastline(COASTLINE)
    cross_lines = load_cross_lines(CROSS)
    boundary = load_boundary_match(META)
    col_to_cell = load_cell_membership()

    fig = plt.figure(figsize=(13, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.0])
    ax_main = fig.add_subplot(gs[0])
    ax_inset = fig.add_subplot(gs[1])

    draw_main_panel(ax_main, cross_lines, coast, col_to_cell, boundary)
    draw_inset_panel(ax_inset, cross_lines, coast, col_to_cell)

    legend_handles = [
        Line2D([0], [0], marker="^", ms=8, lw=0,
                markerfacecolor=CELL_COLOURS_CVD[c["name"]],
                markeredgecolor="#222",
                label=c["name"]) for c in CELLS
    ]
    legend_handles.append(Line2D([0], [0], color="#cbd3da", lw=1.5,
                                  label="transects outside study sectors"))
    fig.legend(handles=legend_handles, ncol=6, loc="upper center",
                bbox_to_anchor=(0.5, 1.03), frameon=False, fontsize=8)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[output] {OUT_PDF}")
    print(f"[output] {OUT_PNG}")

    # CVD simulations: read the PNG, transform, save
    CVD_DIR.mkdir(parents=True, exist_ok=True)
    rgb = mpimg.imread(OUT_PNG)
    if rgb.shape[2] == 4:  # has alpha
        alpha = rgb[..., 3:]
        rgb_clean = rgb[..., :3]
    else:
        alpha = None
        rgb_clean = rgb
    for kind in _CVD:
        sim = simulate_cvd(rgb_clean.astype(float), kind)
        if alpha is not None:
            sim_out = np.dstack([sim, alpha])
        else:
            sim_out = sim
        out_path = CVD_DIR / f"fig1_{kind}.png"
        mpimg.imsave(out_path, sim_out)
        print(f"[output cvd]    {out_path}")


if __name__ == "__main__":
    main()
