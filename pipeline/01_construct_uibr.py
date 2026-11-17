#!/usr/bin/env python3
"""Stage 1: construct UI_BR per canonical-cell transect-day."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _uidx_common import (
    DIAG_DIR,
    FIGURE_DIR,
    OUTPUT_ROOT,
    PRODUCTS,
    SEASONS,
    ensure_output_dirs,
    feature_path,
    load_canonical_transects,
    read_filtered_parquet,
    write_json,
    years_for,
)


OUT_DIR = OUTPUT_ROOT / "uibr_transect"
REQ_COLS = [
    "col_id",
    "date",
    "coast_s_km",
    "distance_km",
    "bp_accepted",
    "beta_inshore",
    "x_b_km",
    "bg_g1",
    "sst_k",
]
NEAR_COAST_MAX_GAP_KM = 25.0


def integrate_group(g: pd.DataFrame) -> dict:
    g = g.sort_values("distance_km")
    first = g.iloc[0]
    accepted = int(first["bp_accepted"]) == 1
    beta = float(first["beta_inshore"]) if pd.notna(first["beta_inshore"]) else np.nan
    xb = float(first["x_b_km"]) if pd.notna(first["x_b_km"]) else np.nan
    bg = float(first["bg_g1"]) if pd.notna(first["bg_g1"]) else np.nan

    if not accepted or not np.isfinite(xb) or xb <= 0 or not np.isfinite(bg):
        return {
            "beta_inshore": np.nan,
            "x_b_km": np.nan,
            "A_in_bar_K": np.nan,
            "bp_accepted": 0,
            "A_in_valid": False,
            "A_in_status": "breakpoint_rejected",
            "n_nodes_inshore": 0,
        }

    x = g["distance_km"].to_numpy(dtype=float)
    sst = g["sst_k"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(sst)
    x = x[valid]
    sst = sst[valid]
    if x.size < 2 or x[0] > NEAR_COAST_MAX_GAP_KM or xb > x[-1]:
        return {
            "beta_inshore": beta,
            "x_b_km": xb,
            "A_in_bar_K": np.nan,
            "bp_accepted": 1,
            "A_in_valid": False,
            "A_in_status": "profile_does_not_span_0_to_xb",
            "n_nodes_inshore": int(np.sum(x <= xb)) if x.size else 0,
        }

    order = np.argsort(x)
    x = x[order]
    sst = sst[order]
    if x[0] > 0:
        first_sst = sst[0]
        x = np.insert(x, 0, 0.0)
        sst = np.insert(sst, 0, first_sst)
    x_clip = x[x <= xb]
    sst_clip = sst[x <= xb]
    if x_clip.size == 0 or x_clip[-1] < xb:
        sst_xb = np.interp(xb, x, sst)
        x_clip = np.append(x_clip, xb)
        sst_clip = np.append(sst_clip, sst_xb)
    anomaly = bg - sst_clip
    area = np.trapezoid(anomaly, x_clip)
    return {
        "beta_inshore": beta,
        "x_b_km": xb,
        "A_in_bar_K": float(area / xb),
        "bp_accepted": 1,
        "A_in_valid": True,
        "A_in_status": "ok",
        "n_nodes_inshore": int(x_clip.size),
    }


def construct_product_year(product: str, year: int, transects: pd.DataFrame) -> pd.DataFrame:
    id_map = transects.set_index("col_id")[["cell", "coast_s_km"]]
    col_ids = [int(x) for x in id_map.index.unique()]
    frames = []
    for season in SEASONS:
        path = feature_path(product, year, season)
        if not path.exists():
            continue
        df = read_filtered_parquet(path, REQ_COLS, col_ids)
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
        rows = []
        for (date, col_id), g in df.groupby(["date", "col_id"], sort=False):
            vals = integrate_group(g)
            meta = id_map.loc[int(col_id)]
            rows.append(
                {
                    "product": product,
                    "date": date,
                    "year": year,
                    "season": season,
                    "cell": meta["cell"],
                    "col_id": int(col_id),
                    "coast_s_km": float(meta["coast_s_km"]),
                    **vals,
                }
            )
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "cell", "col_id"]).reset_index(drop=True)


def write_stage_diagnostics(all_outputs: list[pd.DataFrame]) -> None:
    if not all_outputs:
        return
    df = pd.concat(all_outputs, ignore_index=True)
    accepted = df[df["bp_accepted"] == 1].copy()
    summary = (
        accepted.groupby(["product", "cell", "season"])
        .agg(
            n=("A_in_bar_K", "size"),
            beta_median=("beta_inshore", "median"),
            beta_p05=("beta_inshore", lambda x: x.quantile(0.05)),
            beta_p95=("beta_inshore", lambda x: x.quantile(0.95)),
            xb_median=("x_b_km", "median"),
            xb_p05=("x_b_km", lambda x: x.quantile(0.05)),
            xb_p95=("x_b_km", lambda x: x.quantile(0.95)),
            A_median=("A_in_bar_K", "median"),
            A_p05=("A_in_bar_K", lambda x: x.quantile(0.05)),
            A_p95=("A_in_bar_K", lambda x: x.quantile(0.95)),
        )
        .reset_index()
    )
    summary.to_csv(DIAG_DIR / "stage1_uibr_summary_by_product_cell_season.csv", index=False)

    smoke = df[(df["product"] == "OSTIA") & (df["season"] == "JJA") & (df["date"] == "2020-07-15")]
    smoke.to_csv(DIAG_DIR / "stage1_ostia_jja_2020-07-15_transects.csv", index=False)
    smoke_summary = (
        smoke[(smoke["bp_accepted"] == 1) & (smoke["A_in_valid"])]
        .groupby("cell")
        .agg(
            n=("A_in_bar_K", "size"),
            beta_median=("beta_inshore", "median"),
            xb_median=("x_b_km", "median"),
            A_median=("A_in_bar_K", "median"),
            A_min=("A_in_bar_K", "min"),
            A_max=("A_in_bar_K", "max"),
        )
        .reset_index()
    )
    smoke_summary.to_csv(DIAG_DIR / "stage1_ostia_jja_2020-07-15_cell_summary.csv", index=False)

    if not smoke.empty and (smoke["A_in_valid"].sum() > 0):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharex=True)
        metrics = [
            ("beta_inshore", "beta_inshore (K/km)"),
            ("x_b_km", "x_b (km)"),
            ("A_in_bar_K", "A_in_bar (K)"),
        ]
        plot_df = smoke[(smoke["bp_accepted"] == 1) & (smoke["A_in_valid"])]
        cells = list(dict.fromkeys(plot_df["cell"]))
        for ax, (col, label) in zip(axes, metrics):
            data = [
                plot_df.loc[plot_df["cell"] == cell, col].dropna().to_numpy()
                for cell in cells
            ]
            ax.boxplot(data, tick_labels=cells, showfliers=False)
            ax.set_ylabel(label)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, axis="y", color="#e6e6e6")
        fig.suptitle("Stage 1 UI_BR diagnostics: OSTIA JJA 2020-07-15")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "stage1_ostia_jja_2020-07-15_uibr_distributions.png", dpi=180)
        fig.savefig(FIGURE_DIR / "stage1_ostia_jja_2020-07-15_uibr_distributions.pdf")
        plt.close(fig)

    write_json(
        DIAG_DIR / "stage1_summary.json",
        {
            "rows_written": int(df.shape[0]),
            "accepted_rows": int((df["bp_accepted"] == 1).sum()),
            "A_in_valid_rows": int(df["A_in_valid"].sum()),
            "products": sorted(df["product"].unique().tolist()),
            "years": [int(x) for x in sorted(df["year"].unique())],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=PRODUCTS)
    parser.add_argument("--year", type=int)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transects = load_canonical_transects()

    products = [args.product] if args.product else list(PRODUCTS)
    outputs = []
    for product in products:
        years = [args.year] if args.year else years_for(product)
        for year in years:
            out_path = OUT_DIR / product / f"{product}_{year}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out = construct_product_year(product, year, transects)
            if out.empty:
                continue
            out.to_parquet(out_path, index=False)
            write_json(
                out_path.with_suffix(".json"),
                {
                    "product": product,
                    "year": year,
                    "rows": int(out.shape[0]),
                    "accepted_rows": int((out["bp_accepted"] == 1).sum()),
                    "A_in_valid_rows": int(out["A_in_valid"].sum()),
                    "units": {
                        "beta_inshore": "K km-1",
                        "x_b_km": "km",
                        "A_in_bar_K": "K",
                    },
                },
            )
            outputs.append(out)
            print(
                f"wrote {out_path} rows={out.shape[0]} "
                f"accepted={(out['bp_accepted'] == 1).sum()} "
                f"A_in_valid={out['A_in_valid'].sum()}"
            )
    write_stage_diagnostics(outputs)


if __name__ == "__main__":
    main()
