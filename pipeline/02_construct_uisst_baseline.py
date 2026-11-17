#!/usr/bin/env python3
"""Stage 2: construct fixed-distance UI_SST baselines."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _uidx_common import (
    DIAG_DIR,
    FIGURE_DIR,
    OUTPUT_ROOT,
    PRODUCTS,
    REFERENCE_DISTANCES_KM,
    SEASONS,
    ensure_output_dirs,
    feature_path,
    load_canonical_transects,
    read_filtered_parquet,
    write_json,
    years_for,
)


OUT_DIR = OUTPUT_ROOT / "uisst_baseline"
UIBR_DIR = OUTPUT_ROOT / "uibr_transect"
REQ_COLS = ["col_id", "date", "distance_km", "sst_k"]
NEAR_COAST_MAX_GAP_KM = 25.0


def sample_group(g: pd.DataFrame) -> list[dict]:
    g = g.sort_values("distance_km")
    x = g["distance_km"].to_numpy(dtype=float)
    sst = g["sst_k"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(sst)
    x = x[valid]
    sst = sst[valid]
    if x.size < 2:
        return []
    order = np.argsort(x)
    x = x[order]
    sst = sst[order]
    if x[0] > NEAR_COAST_MAX_GAP_KM:
        t_coast = np.nan
    elif x[0] > 0:
        t_coast = float(sst[0])
    else:
        t_coast = float(np.interp(0.0, x, sst))
    rows = []
    for d0 in REFERENCE_DISTANCES_KM:
        t_off = float(np.interp(d0, x, sst)) if d0 <= x[-1] else np.nan
        rows.append(
            {
                "d_0_km": d0,
                "T_coast_K": t_coast,
                "T_offshore_K": t_off,
                "UI_SST_K": t_coast - t_off if np.isfinite(t_coast) and np.isfinite(t_off) else np.nan,
                "UI_SST_cool_K": t_off - t_coast if np.isfinite(t_coast) and np.isfinite(t_off) else np.nan,
            }
        )
    return rows


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
            meta = id_map.loc[int(col_id)]
            for vals in sample_group(g):
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
    return pd.concat(frames, ignore_index=True).sort_values(
        ["date", "cell", "col_id", "d_0_km"]
    )


def load_uibr_outputs() -> pd.DataFrame:
    paths = sorted(UIBR_DIR.glob("*/*.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)


def write_stage_diagnostics(stage2_outputs: list[pd.DataFrame]) -> None:
    if not stage2_outputs:
        return
    ui = pd.concat(stage2_outputs, ignore_index=True)
    uibr = load_uibr_outputs()
    if uibr.empty:
        return

    uibr_acc = uibr[uibr["bp_accepted"] == 1]
    a_summary = (
        uibr_acc.groupby(["cell", "product", "season"])
        .agg(A_in_median=("A_in_bar_K", "median"), x_b_median=("x_b_km", "median"))
        .reset_index()
    )
    ui_summary = (
        ui.groupby(["cell", "product", "season", "d_0_km"])
        .agg(UI_SST_cool_median=("UI_SST_cool_K", "median"))
        .reset_index()
    )
    wide = ui_summary.pivot_table(
        index=["cell", "product", "season"],
        columns="d_0_km",
        values="UI_SST_cool_median",
    ).reset_index()
    wide = wide.rename(columns={d: f"UI_SST_cool_d{int(d):03d}_median" for d in REFERENCE_DISTANCES_KM})
    diag = a_summary.merge(wide, on=["cell", "product", "season"], how="left")
    diag.to_csv(DIAG_DIR / "stage2_uisst_vs_uibr_by_cell_product_season.csv", index=False)

    plot = diag[diag["product"].isin(["OSTIA", "OSPO"])].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    for ax, season in zip(axes.ravel(), ["DJF", "MAM", "JJA", "SON"]):
        sub = plot[plot["season"] == season]
        x = np.arange(len(sub))
        ax.plot(x, sub["A_in_median"], "ko-", label="A_in median")
        for d0 in REFERENCE_DISTANCES_KM:
            col = f"UI_SST_d{int(d0):03d}_median"
            col = f"UI_SST_cool_d{int(d0):03d}_median"
            ax.plot(x, sub[col], marker=".", lw=1, label=f"UI_SST cool {int(d0)} km")
        ax.set_title(season)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["cell"] + "\n" + sub["product"], rotation=90, fontsize=7)
        ax.grid(True, axis="y", color="#e6e6e6")
    axes[0, 0].set_ylabel("K")
    axes[1, 0].set_ylabel("K")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Stage 2: fixed-distance UI_SST medians vs Stage 1 A_in")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "stage2_uisst_vs_uibr_medians.png", dpi=180)
    fig.savefig(FIGURE_DIR / "stage2_uisst_vs_uibr_medians.pdf")
    plt.close(fig)

    write_json(
        DIAG_DIR / "stage2_summary.json",
        {
            "rows_written": int(ui.shape[0]),
            "products": sorted(ui["product"].unique().tolist()),
            "reference_distances_km": list(REFERENCE_DISTANCES_KM),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=PRODUCTS)
    parser.add_argument("--year", type=int)
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
                    "distances_km": list(REFERENCE_DISTANCES_KM),
                    "units": {
                        "UI_SST_K": "K; T_coast - T_offshore",
                        "UI_SST_cool_K": "K; T_offshore - T_coast",
                        "T_coast_K": "K",
                        "T_offshore_K": "K",
                    },
                },
            )
            outputs.append(out)
            print(f"wrote {out_path} rows={out.shape[0]}")
    write_stage_diagnostics(outputs)


if __name__ == "__main__":
    main()
