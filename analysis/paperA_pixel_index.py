#!/usr/bin/env python3
"""paperA_pixel_index.py -- Stage 2 Task 2.1: map every accepted sigma-node
observation in the recovered features_bp tree to its parent L4 pixel index.

Addendum task: Stage 2, Task 2.1 (pixel index for accepted samples).

Per (product, year, season):
  - Load the features_bp parquet, restrict to cell-window col_ids
    from paperA_cell_transects.csv, and filter to rows where sst_k
    is finite (do not filter by bp_accepted; rejected transects also
    carry pixel coverage that feeds the Task 2.4 p_active_thin
    denominator).
  - For each surviving sigma-node row, compute the parent L4 pixel
    index (pix_i, pix_j) for the product's native lat-lon grid.
  - Deduplicate within each (col_id, date) group by (pix_i, pix_j) so
    that each parent pixel appears at most once per transect-day.
    The number of surviving rows per (col_id, date) is the number of
    unique L4 pixels that transect covers on that day.
  - Emit one parquet per (product, year, season) with columns
    (cell, product, date, col_id, sigma, pix_i, pix_j, lat, lon,
    accepted). The `accepted` column carries bp_accepted (0/1) for
    downstream Task 2.2 / 2.4 logic.

Grid bin convention: pix_i = floor((lat + 90) / lat_step_deg),
                     pix_j = floor((lon + 180) / lon_step_deg)
with lat_step_deg = lon_step_deg = native_grain_km / 111.0 (one
degree of latitude is ~111 km; we use the same step in longitude
because the GHRSST L4 grids are isotropic in degrees, not km).
This gives non-negative integer indices for any lat/lon on Earth.

Outputs:
  data/outputs/uidx/paperA_pixel_index/<PRODUCT>/<PRODUCT>_<YEAR>_<SEASON>.parquet

Verification: for one Cape Columbine OSTIA DJF 2020 transect-day,
the unique-pixel count matches the source parquet's n_eff to ±1.

Run from the release repository root:
    python3 analysis/paperA_pixel_index.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperA_common as P


# Native grid step in degrees. PODAAC OSTIA/OSPO L4 are on 0.05° grids;
# AVHRR_OI is on 0.25°; MW_OI is on 0.25°. We derive the step from the
# native_grain_km recorded in each parquet's meta JSON so that any future
# product re-release with a different grain is picked up automatically.
DEG_PER_KM = 1.0 / 111.0

OUT_ROOT = P.ROOT / "data/outputs/uidx/paperA_pixel_index"


def native_grid_step_deg(meta_path: Path) -> float:
    meta = json.loads(meta_path.read_text())
    return float(meta["native_grain_km"]) * DEG_PER_KM


def process(product: str, year: int, season: str,
            col_id_to_cell: dict[int, str]) -> tuple[int, int, float]:
    """Process one (product, year, season). Returns (n_rows_out, n_groups, elapsed)."""
    src = P.feature_root() / product / f"{product}_{year}_{season}.parquet"
    meta_src = src.with_name(src.stem + "_meta.json")
    if not src.exists():
        return (0, 0, 0.0)

    t0 = time.monotonic()
    step = native_grid_step_deg(meta_src)
    col_ids = sorted(col_id_to_cell.keys())
    tbl = pq.read_table(
        src,
        columns=["col_id", "date", "sigma", "lat", "lon", "sst_k", "bp_accepted"],
        filters=[("col_id", "in", col_ids)],
    )
    df = tbl.to_pandas()
    if df.empty:
        return (0, 0, time.monotonic() - t0)
    df = df[np.isfinite(df["sst_k"])]
    if df.empty:
        return (0, 0, time.monotonic() - t0)

    df["pix_i"] = np.floor((df["lat"].to_numpy() + 90.0) / step).astype(np.int32)
    df["pix_j"] = np.floor((df["lon"].to_numpy() + 180.0) / step).astype(np.int32)
    df["cell"] = df["col_id"].map(col_id_to_cell)
    df["product"] = product
    df = df.rename(columns={"bp_accepted": "accepted"})

    # First-occurrence-by-sigma dedup within (col_id, date) by (pix_i, pix_j)
    # — sigma increases monotonically from 0 (coast) to 1 (offshore), so the
    # first occurrence is the inboard-most sigma falling in that pixel.
    df = df.sort_values(["col_id", "date", "sigma"], kind="mergesort")
    df = df.drop_duplicates(["col_id", "date", "pix_i", "pix_j"], keep="first")

    out = df[["cell", "product", "date", "col_id", "sigma", "pix_i", "pix_j", "lat", "lon", "accepted"]]
    dst = OUT_ROOT / product / f"{product}_{year}_{season}.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst, index=False)

    n_groups = df.groupby(["col_id", "date"]).ngroups
    elapsed = time.monotonic() - t0
    return (len(out), n_groups, elapsed)


def main() -> None:
    membership = pd.read_csv(P.paperA_cell_transects_path())
    col_id_to_cell = dict(zip(membership["col_id"], membership["cell"]))
    print(f"[mesh] {P.MESH_TAG}; {len(col_id_to_cell)} cell-window col_ids across "
          f"{len(membership['cell'].unique())} cells")
    print(f"[features_bp] {P.feature_root()}")
    print(f"[out] {OUT_ROOT}\n")

    total_rows = 0
    total_groups = 0
    t_start = time.monotonic()
    for product in P.PRODUCTS:
        for year in P.years_for(product):
            for season in P.SEASONS:
                n_rows, n_groups, dt = process(product, year, season, col_id_to_cell)
                total_rows += n_rows
                total_groups += n_groups
                if n_rows:
                    print(f"  [{product} {year} {season}] "
                          f"{n_rows:>10,} rows  {n_groups:>6,} (col_id,date) groups  {dt:.1f}s",
                          flush=True)
    print(f"\n[total] {total_rows:,} rows across {total_groups:,} transect-days "
          f"in {time.monotonic() - t_start:.1f}s")


if __name__ == "__main__":
    main()
