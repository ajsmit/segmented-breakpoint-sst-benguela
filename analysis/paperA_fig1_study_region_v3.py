#!/usr/bin/env python3
"""Add illustrative Benguela regions to Figure 1, preserving the v2 mesh.

Run from the project root with .venv/bin/python. Approximate regional
limits follow the maps in Salama et al. (2025), Figs. 1 and 3,
doi:10.3389/fmars.2025.1601284. They do not change sector membership.
"""
import paperA_common as P

# The same r2 mesh is retained in the versioned output archive.
P.MESH_ROOT_CANDIDATES = (
    P.ROOT / "data/outputs/versioned" / P.MESH_TAG / "mesh_curvilinear",
    *P.MESH_ROOT_CANDIDATES,
)
import paperA_fig1_study_region as base
from matplotlib.lines import Line2D
from matplotlib.text import Text


def main():
    coast = base.load_coastline(base.COASTLINE)
    cross = base.load_cross_lines(base.CROSS)
    boundary = base.load_boundary_match(base.META)
    membership = base.load_cell_membership()
    assert len(cross) == 742 and len(membership) == 189

    fig = base.plt.figure(figsize=(13, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.0])
    main_ax = fig.add_subplot(gs[0])
    inset_ax = fig.add_subplot(gs[1])
    base.draw_main_panel(main_ax, cross, coast, membership, boundary)
    base.draw_inset_panel(inset_ax, cross, coast, membership)

    # Put the geographical guides in the empty land area to keep all
    # transects, sector colours, and their analytical boundaries visible.
    for label, north, south in (
        ("NBUS", -16.0, -24.5),
        ("CBUS", -24.5, -27.5),
        ("SBUS", -27.5, -35.0),
    ):
        main_ax.plot([21.1, 21.5, 21.5, 21.1],
                     [north, north, south, south],
                     color="#666666", lw=0.8, zorder=9)
        main_ax.text(20.7, (north + south) / 2, label,
                     ha="right", va="center", fontsize=9,
                     color="#444444", weight="bold", zorder=10)

    handles = [
        Line2D([0], [0], marker="^", ms=8, lw=0,
               markerfacecolor=base.CELL_COLOURS_CVD[c["name"]],
               markeredgecolor="#222", label=c["name"])
        for c in base.CELLS
    ]
    handles.append(Line2D([0], [0], color="#cbd3da", lw=1.5,
                          label="transects outside study sectors"))
    fig.legend(handles=handles, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 1.03), frameon=False, fontsize=8)
    for text in fig.findobj(match=Text):
        if text.get_text() == "Luderitz":
            text.set_text("Lüderitz")

    for suffix in ("pdf", "png"):
        output = P.PAPERA_FIGURES / f"fig1_benguela_uc_mesh_map_v3.{suffix}"
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(output)
    base.plt.close(fig)


if __name__ == "__main__":
    main()
