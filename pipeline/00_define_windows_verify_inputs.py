#!/usr/bin/env python3
"""Stage 0: define canonical uidx cell windows and verify inherited inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from _uidx_common import (
    CELL_DEFS,
    DIAG_DIR,
    FIGURE_DIR,
    MESH_TAG,
    PRODUCTS,
    SEASONS,
    UIDX_DATA_ACCESS,
    ensure_output_dirs,
    feature_path,
    feature_root,
    load_coastal_transects_from_features,
    years_for,
)


REQUIRED_COLUMNS = [
    "beta_inshore",
    "beta_offshore",
    "x_b_km",
    "bp_accepted",
    "bg_g1",
    "delta_aic",
    "rss_one_segment_K2",
    "rss_two_segment_K2",
    "aic_one_segment",
    "aic_two_segment",
    "sst_k",
    "distance_km",
    "col_id",
    "date",
]

MIDPOINT_CUT_S_KM = 2842.5
MIDPOINT_CUT_NOTE = (
    "midpoint cut at 2842.5 km to disjoint from adjacent cell; effective "
    "half-width 95.5 km on the inner edge, 100 km on the outer edge"
)


def build_windows(half_width_km: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coastal = load_coastal_transects_from_features()
    window_rows = []
    transect_rows = []

    for cell in CELL_DEFS:
        target_lat = cell["lat_centre"]
        centre_idx = (coastal["lat"] - target_lat).abs().idxmin()
        centre = coastal.loc[centre_idx]
        s0 = float(centre["coast_s_km"])
        s_min = s0 - half_width_km
        s_max = s0 + half_width_km
        design_note = ""
        if cell["cell"] == "Cape Columbine" and s_max > MIDPOINT_CUT_S_KM:
            s_max = MIDPOINT_CUT_S_KM
            design_note = MIDPOINT_CUT_NOTE
        if cell["cell"] == "Cape Peninsula" and s_min < MIDPOINT_CUT_S_KM:
            s_min = MIDPOINT_CUT_S_KM
            design_note = MIDPOINT_CUT_NOTE

        in_window = coastal[
            coastal["coast_s_km"].between(s_min, s_max, inclusive="both")
        ].copy()
        in_window["cell"] = cell["cell"]

        window_rows.append(
            {
                "cell": cell["cell"],
                "lat_centre": target_lat,
                "sub_system": cell["sub_system"],
                "coast_s_km_centre": s0,
                "coast_s_km_min": s_min,
                "coast_s_km_max": s_max,
                "half_width_km": half_width_km,
                "design_note": design_note,
                "n_transects": int(in_window["col_id"].nunique()),
                "nearest_mesh_lat": float(centre["lat"]),
                "nearest_mesh_lon": float(centre["lon"]),
                "source_citation": cell["source_citation"],
            }
        )
        transect_rows.append(
            in_window[["cell", "col_id", "coast_s_km", "lat", "lon"]]
        )

    windows = pd.DataFrame(window_rows)
    transects = pd.concat(transect_rows, ignore_index=True)
    return coastal, windows, transects


def verify_schema_and_transects(transects: pd.DataFrame) -> pd.DataFrame:
    canonical_ids = sorted(int(x) for x in transects["col_id"].unique())
    records = []
    for product in PRODUCTS:
        for year in years_for(product):
            for season in SEASONS:
                path = feature_path(product, year, season)
                if not path.exists():
                    records.append(
                        {
                            "product": product,
                            "year": year,
                            "season": season,
                            "path": str(path),
                            "exists": False,
                            "n_rows": 0,
                            "missing_columns": ",".join(REQUIRED_COLUMNS),
                            "n_canonical_transects_present": 0,
                            "all_required_transects_present": False,
                        }
                    )
                    continue

                pf = pq.ParquetFile(path)
                names = set(pf.schema_arrow.names)
                missing = sorted(set(REQUIRED_COLUMNS) - names)
                n_present = 0
                all_present = False
                if not missing:
                    dataset = ds.dataset(str(path), format="parquet")
                    table = dataset.to_table(
                        columns=["col_id", "sigma"],
                        filter=(
                            pc.field("col_id").isin(canonical_ids)
                            & (pc.field("sigma") == 0)
                        ),
                    )
                    present = set(table.column("col_id").to_pylist())
                    n_present = len(present)
                    all_present = set(canonical_ids).issubset(present)

                records.append(
                    {
                        "product": product,
                        "year": year,
                        "season": season,
                        "path": str(path),
                        "exists": True,
                        "n_rows": pf.metadata.num_rows,
                        "missing_columns": ",".join(missing),
                        "n_canonical_transects_present": n_present,
                        "all_required_transects_present": all_present,
                    }
                )
    return pd.DataFrame(records)


def plot_windows(coastal: pd.DataFrame, transects: pd.DataFrame, windows: pd.DataFrame) -> None:
    colors = {
        "Cape Frio": "#2f80ed",
        "Walvis Bay": "#27ae60",
        "Luderitz": "#8e44ad",
        "Cape Columbine": "#f39c12",
        "Cape Peninsula": "#c0392b",
    }

    fig, (ax_map, ax_s) = plt.subplots(
        1, 2, figsize=(11.5, 6.2), gridspec_kw={"width_ratios": [1.05, 1.0]}
    )
    ax_map.plot(coastal["lon"], coastal["lat"], color="#9aa0a6", lw=1.0, zorder=1)
    for cell, sub in transects.groupby("cell"):
        ax_map.scatter(
            sub["lon"], sub["lat"], s=18, color=colors[cell], label=cell, zorder=3
        )
    ax_map.set_xlabel("Longitude (deg E)")
    ax_map.set_ylabel("Latitude (deg N)")
    ax_map.set_title("Canonical uidx cell windows on uc2d_v2 coast")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.grid(True, color="#e6e6e6", lw=0.6)
    ax_map.legend(fontsize=8, loc="lower left", frameon=True)

    ax_s.plot(coastal["coast_s_km"], coastal["lat"], color="#9aa0a6", lw=1.0)
    for _, row in windows.iterrows():
        ax_s.axvspan(
            row["coast_s_km_min"],
            row["coast_s_km_max"],
            color=colors[row["cell"]],
            alpha=0.18,
        )
        ax_s.text(
            row["coast_s_km_centre"],
            row["nearest_mesh_lat"],
            row["cell"],
            color=colors[row["cell"]],
            fontsize=8,
            ha="center",
            va="bottom",
            rotation=90,
        )
    ax_s.set_xlabel("Along-coast distance coast_s_km")
    ax_s.set_ylabel("Latitude (deg N)")
    ax_s.set_title("Window centres and half-widths")
    ax_s.grid(True, color="#e6e6e6", lw=0.6)

    fig.suptitle(f"uidx Stage 0 foundations: mesh {MESH_TAG}", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "uidx_canonical_cells_map.png", dpi=220)
    fig.savefig(FIGURE_DIR / "uidx_canonical_cells_map.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--half-width-km", type=float, default=100.0)
    args = parser.parse_args()

    ensure_output_dirs()
    coastal, windows, transects = build_windows(args.half_width_km)
    schema = verify_schema_and_transects(transects)

    windows.to_csv(UIDX_DATA_ACCESS / "canonical_cell_windows.csv", index=False)
    transects.to_csv(UIDX_DATA_ACCESS / "canonical_cell_transects.csv", index=False)
    schema.to_csv(DIAG_DIR / "stage0_features_bp_schema_verification.csv", index=False)

    overlap = (
        transects.groupby("col_id")["cell"]
        .nunique()
        .reset_index(name="n_cells")
        .query("n_cells > 1")
    )
    coverage_km = float(windows["coast_s_km_max"].max() - windows["coast_s_km_min"].min())
    occupied_km = float((windows["coast_s_km_max"] - windows["coast_s_km_min"]).sum())
    summary = {
        "mesh_tag": MESH_TAG,
        "feature_root": str(feature_root()),
        "half_width_km": args.half_width_km,
        "n_cells": int(windows.shape[0]),
        "n_unique_transects": int(transects["col_id"].nunique()),
        "occupied_window_km": occupied_km,
        "span_first_to_last_window_km": coverage_km,
        "overlapping_transects": int(overlap.shape[0]),
        "feature_files_checked": int(schema["exists"].sum()),
        "feature_files_missing": int((~schema["exists"]).sum()),
        "files_with_missing_required_columns": int((schema["missing_columns"] != "").sum()),
        "files_missing_any_required_transect": int(
            (~schema["all_required_transects_present"]).sum()
        ),
    }
    (DIAG_DIR / "stage0_summary.json").write_text(json.dumps(summary, indent=2))
    plot_windows(coastal, transects, windows)

    print("\nStage 0 complete")
    print(pd.DataFrame([summary]).to_string(index=False))
    print("\nCanonical windows")
    print(
        windows[
            [
                "cell",
                "lat_centre",
                "coast_s_km_centre",
                "coast_s_km_min",
                "coast_s_km_max",
                "n_transects",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
