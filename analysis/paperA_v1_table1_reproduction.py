#!/usr/bin/env python3
"""paperA_v1_table1_reproduction.py -- operational per-(cell, product)
climatological-median table for paperA Table 1.

Updated 2026-05-23: across-transect pixel uniqueness is now enforced as
an operational acceptance criterion (the v1 logic accepted every
bp_accepted=1 transect-day). For each (cell, product, date), only the
transects retained under the first-occurrence-wins L4-pixel dedup
(paperA_thinned/) contribute to the per-(date, col_id) population that
feeds the medians. The "v1 reproduction" label is retained for the
historical comparison columns; the v2 columns are now the operational
thinned values.

Two outputs:
  - data_access-equivalent paperA_cell_transects.csv (r2-mesh-keyed
    regeneration of the stale results/data_access/canonical_cell_transects.csv).
    Membership rule: a col_id belongs to a cell iff its coast_s_km falls in
    that cell's [coast_s_km_min, coast_s_km_max] window from
    canonical_cell_windows.csv.
  - paperA_v1_table1_reproduction.csv: one row per (cell, product) with
    the v1 medians, the r2 + thinned operational medians, and per-metric
    diffs from v1.

Aggregation logic:
  - For each (date, col_id) with bp_accepted == 1 AND retained_bool == 1
    in the thinned tree, compute D_in_bar_K as the prefix-sum integral
    of (bg_g1 - sst_k) over [0, x_b_km] divided by x_b_km, per
    pipeline/01_construct_uibr.py::integrate_group.
  - Per-(cell, product, season, year) medians of D_in_bar_K, x_b_km,
    beta_inshore over per-(date, col_id) values.
  - Per-(cell, product) climatological median = median across all
    (season, year) values for that pair.

Verification: per-(cell, product) absolute diffs from v1 archive within
the addendum's 5 % tolerance on OSTIA and OSPO. Larger diffs are flagged
and explained.

Run from the release repository root:
    python3 analysis/paperA_v1_table1_reproduction.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]

PAPERA_MESH_TAG = "fmm_uc2d_v2_outer500km_v2026_05_21_r2"
MESH_DIR = ROOT / "data/derived/analysis_runs" / PAPERA_MESH_TAG / "mesh_curvilinear"
WINDOWS_CSV = ROOT / "results/data_access/canonical_cell_windows.csv"
FEATURES_BP_ROOT = ROOT / "data/outputs/uc2d_v2/features_bp"
THIN_ROOT = ROOT / "data/outputs/uidx/paperA_thinned"
V1_ARCHIVE_PARQUET = ROOT / "data/outputs/uidx/per_cell_summary.parquet"

OUT_CELL_TRANSECTS = ROOT / "results/audit/paperA_cell_transects.csv"
OUT_REPRODUCTION = ROOT / "results/audit/paperA_v1_table1_reproduction.csv"
OUT_META = ROOT / "results/audit/paperA_v1_table1_reproduction_meta.json"

SEASONS = ["DJF", "MAM", "JJA", "SON"]
PRODUCT_YEARS = {
    "OSTIA": list(range(2015, 2026)),
    "OSPO":  list(range(2015, 2026)),
    "AVHRR": list(range(2016, 2026)),
    "MW_OI": list(range(2015, 2026)),
}
REQ_COLS = ["col_id", "date", "distance_km", "sst_k",
            "bp_accepted", "beta_inshore", "x_b_km", "bg_g1"]
NEAR_COAST_MAX_GAP_KM = 25.0


def regenerate_cell_transects() -> pd.DataFrame:
    """Pick r2 col_ids per cell window. Returns the (cell, col_id, coast_s_km)
    membership table and writes it to OUT_CELL_TRANSECTS."""
    windows = pd.read_csv(WINDOWS_CSV)
    boundary = pd.read_csv(MESH_DIR / "mesh_boundary_match.csv")
    rows = []
    for _, w in windows.iterrows():
        cell = w["cell"]
        lo, hi = float(w["coast_s_km_min"]), float(w["coast_s_km_max"])
        sub = boundary[(boundary["coast_s_km"] >= lo) & (boundary["coast_s_km"] <= hi)]
        for _, r in sub.iterrows():
            rows.append({
                "cell": cell,
                "col_id": int(r["col_id"]),
                "coast_s_km": float(r["coast_s_km"]),
                "coast_lat": float(r["coast_lat"]),
                "coast_lon": float(r["coast_lon"]),
            })
    out = pd.DataFrame(rows).sort_values(["cell", "col_id"]).reset_index(drop=True)
    OUT_CELL_TRANSECTS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CELL_TRANSECTS, index=False)
    print(f"[cell_transects] wrote {len(out)} rows -> {OUT_CELL_TRANSECTS}")
    print(out.groupby("cell").size().rename("n_transects"))
    return out


def integrate_group(distance_km: np.ndarray, sst_k: np.ndarray,
                    bg: float, xb: float) -> float:
    """v1 D_in_bar_K computation, mirroring 01_construct_uibr.py::integrate_group."""
    valid = np.isfinite(distance_km) & np.isfinite(sst_k)
    x = distance_km[valid]
    sst = sst_k[valid]
    if x.size < 2 or x[0] > NEAR_COAST_MAX_GAP_KM or xb > x[-1]:
        return np.nan
    order = np.argsort(x)
    x = x[order]
    sst = sst[order]
    if x[0] > 0:
        x = np.insert(x, 0, 0.0)
        sst = np.insert(sst, 0, sst[0])
    mask = x <= xb
    x_clip = x[mask]
    sst_clip = sst[mask]
    if x_clip.size == 0 or x_clip[-1] < xb:
        sst_xb = np.interp(xb, x, sst)
        x_clip = np.append(x_clip, xb)
        sst_clip = np.append(sst_clip, sst_xb)
    return float(np.trapezoid(bg - sst_clip, x_clip) / xb)


def process_product_year_season(product: str, year: int, season: str,
                                col_id_to_cell: dict[int, str]) -> pd.DataFrame:
    """Returns per-(cell, season, year) median rows for one parquet.

    Applies the across-transect pixel-uniqueness acceptance criterion by
    intersecting with retained_bool == 1 from the thinned tree.
    """
    fp = FEATURES_BP_ROOT / product / f"{product}_{year}_{season}.parquet"
    if not fp.exists():
        return pd.DataFrame()
    thin_fp = THIN_ROOT / product / f"{product}_{year}_{season}.parquet"
    if not thin_fp.exists():
        return pd.DataFrame()
    # Load the retained set for this (product, year, season).
    thin = pd.read_parquet(thin_fp, columns=["date", "col_id", "retained_bool"])
    thin = thin[thin["retained_bool"] == 1]
    retained_keys = set(zip(thin["date"].astype(str), thin["col_id"].astype(int)))
    if not retained_keys:
        return pd.DataFrame()
    # Push col_id filter via pyarrow to skip whole row groups where possible.
    col_ids = sorted(col_id_to_cell.keys())
    table = pq.read_table(
        fp,
        columns=REQ_COLS,
        filters=[("col_id", "in", col_ids)],
    )
    df = table.to_pandas()
    if df.empty:
        return pd.DataFrame()
    df["cell"] = df["col_id"].map(col_id_to_cell)
    # Per (date, col_id): pull bp_accepted, beta_inshore, x_b_km, bg_g1 from first row;
    # integrate D_in using all inshore-segment SST samples.
    rows = []
    for (cell, date, cid), g in df.groupby(["cell", "date", "col_id"], sort=False):
        first = g.iloc[0]
        if int(first["bp_accepted"]) != 1:
            continue
        # Operational acceptance: also require retained_bool == 1 (across-transect
        # pixel uniqueness, first-occurrence-wins by col_id within each
        # (cell, product, date)).
        if (str(date), int(cid)) not in retained_keys:
            continue
        xb = first["x_b_km"]
        bg = first["bg_g1"]
        beta = first["beta_inshore"]
        if not (np.isfinite(xb) and xb > 0 and np.isfinite(bg) and np.isfinite(beta)):
            continue
        D_in = integrate_group(g["distance_km"].to_numpy(dtype=float),
                               g["sst_k"].to_numpy(dtype=float),
                               float(bg), float(xb))
        rows.append({
            "cell": cell, "date": date, "col_id": int(cid),
            "D_in": D_in, "x_b_km": float(xb), "beta_inshore": float(beta),
        })
    per_day = pd.DataFrame(rows)
    if per_day.empty:
        return pd.DataFrame()
    # Per-(cell, season, year) medians.
    grp = per_day.groupby("cell", as_index=False).agg(
        D_in_median=("D_in", lambda s: float(np.nanmedian(s))),
        x_b_median=("x_b_km", "median"),
        beta_inshore_median=("beta_inshore", "median"),
        n_days_accepted=("D_in", "size"),
    )
    grp["product"] = product
    grp["year"] = year
    grp["season"] = season
    return grp


def aggregate_v2() -> pd.DataFrame:
    membership = regenerate_cell_transects()
    col_id_to_cell = dict(zip(membership["col_id"], membership["cell"]))
    parts = []
    for product, years in PRODUCT_YEARS.items():
        for year in years:
            for season in SEASONS:
                t0 = time.monotonic()
                rs = process_product_year_season(product, year, season, col_id_to_cell)
                if rs.empty:
                    continue
                parts.append(rs)
                print(f"  [{product} {year} {season}] "
                      f"{rs.shape[0]} cell-rows, {int(rs['n_days_accepted'].sum())} acc-rows, "
                      f"{time.monotonic() - t0:.1f}s",
                      flush=True)
    if not parts:
        raise SystemExit("no per-(cell, season, year) rows produced")
    per_year_season = pd.concat(parts, ignore_index=True)
    # Per-(cell, product) climatological median across (season, year).
    climo = per_year_season.groupby(["cell", "product"], as_index=False).agg(
        D_in_v2=("D_in_median", "median"),
        x_b_v2=("x_b_median", "median"),
        beta_in_v2=("beta_inshore_median", "median"),
        n_year_seasons=("D_in_median", "size"),
    )
    return climo


def load_v1_climatology() -> pd.DataFrame:
    # Read through paperA_common.read_features_bp shim (Task 7.1): the v1
    # parquet has A_in_median; the shim relabels it to D_in_median so the
    # downstream agg call sees the paperA naming.
    import paperA_common as _P
    v1 = _P.read_features_bp(V1_ARCHIVE_PARQUET)
    out = v1.groupby(["cell", "product"], as_index=False).agg(
        D_in_v1=("D_in_median", "median"),
        x_b_v1=("x_b_median", "median"),
        beta_in_v1=("beta_inshore_median", "median"),
    )
    # Both v1 archive and results/data_access/canonical_cell_windows.csv spell
    # the cell as ASCII "Luderitz"; the diacritic spelling is used only in
    # paper-facing strings, not in identifier columns.
    return out


def diff_pct(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    if b == 0:
        return np.nan
    return 100.0 * (a - b) / abs(b)


def main():
    print(f"[mesh] {PAPERA_MESH_TAG}")
    print(f"[v1 archive] {V1_ARCHIVE_PARQUET}")
    print(f"[features_bp] {FEATURES_BP_ROOT}")
    climo_v2 = aggregate_v2()
    climo_v1 = load_v1_climatology()
    merged = climo_v1.merge(climo_v2, on=["cell", "product"], how="outer")
    merged["diff_D_in_pct"] = [diff_pct(a, b) for a, b in zip(merged["D_in_v2"], merged["D_in_v1"])]
    merged["diff_x_b_pct"] = [diff_pct(a, b) for a, b in zip(merged["x_b_v2"], merged["x_b_v1"])]
    merged["diff_beta_in_pct"] = [diff_pct(a, b) for a, b in zip(merged["beta_in_v2"], merged["beta_in_v1"])]
    merged = merged.sort_values(["cell", "product"]).reset_index(drop=True)
    OUT_REPRODUCTION.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_REPRODUCTION, index=False, float_format="%.6g")
    print(f"\n[output] {OUT_REPRODUCTION}")
    print(merged.to_string(index=False))

    meta = {
        "mesh_tag": PAPERA_MESH_TAG,
        "v1_archive": str(V1_ARCHIVE_PARQUET),
        "features_bp_root": str(FEATURES_BP_ROOT),
        "rule": "median over (date, col_id) per (cell, season, year), then median over (season, year) per (cell, product).",
        "tolerance_pct": 5.0,
        "OSTIA_OSPO_within_5pct": {
            metric: bool(
                merged[(merged["product"].isin(["OSTIA", "OSPO"])) & merged[col].notna()][col].abs().max() <= 5.0
                if merged[(merged["product"].isin(["OSTIA", "OSPO"])) & merged[col].notna()].shape[0] > 0 else True
            )
            for metric, col in [
                ("D_in", "diff_D_in_pct"),
                ("x_b", "diff_x_b_pct"),
                ("beta_in", "diff_beta_in_pct"),
            ]
        },
        "max_diffs_pct": {
            "D_in":   {p: float(np.nanmax(np.abs(merged[merged["product"] == p]["diff_D_in_pct"])))
                       for p in merged["product"].unique() if merged[merged["product"] == p]["diff_D_in_pct"].notna().any()},
            "x_b":    {p: float(np.nanmax(np.abs(merged[merged["product"] == p]["diff_x_b_pct"])))
                       for p in merged["product"].unique() if merged[merged["product"] == p]["diff_x_b_pct"].notna().any()},
            "beta_in":{p: float(np.nanmax(np.abs(merged[merged["product"] == p]["diff_beta_in_pct"])))
                       for p in merged["product"].unique() if merged[merged["product"] == p]["diff_beta_in_pct"].notna().any()},
        },
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"[meta] {OUT_META}")


if __name__ == "__main__":
    main()
