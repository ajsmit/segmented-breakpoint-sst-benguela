#!/usr/bin/env python3
"""paperA_din_thin_shift.py -- Stage 2 Task 2.5 helper: climatological-median
D̄_in shift on retained transects relative to all-accepted transects.

Computes, per (cell, product, season):

  D̄_in_v1_median   = climatological median of daily D_in_bar_K medians
                     over accepted transects (v1 baseline)
  D̄_in_thin_median = same but restricted to retained transects (those
                     contributing >=1 unique L4 pixel after across-transect
                     dedup, from Task 2.2)
  shift_C          = D̄_in_thin_median - D̄_in_v1_median   (degrees C)

Inputs:
  - data/outputs/uc2d_v2/features_bp/<PRODUCT>/<PRODUCT>_<YEAR>_<SEASON>.parquet
  - data/outputs/uidx/paperA_thinned/<PRODUCT>/<PRODUCT>_<YEAR>_<SEASON>.parquet
  - results/audit/paperA_cell_transects.csv

Outputs:
  - results/diagnostics/din_thin_shift.csv

Run from the release repository root:
    python3 analysis/paperA_din_thin_shift.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperA_common as P
import paperA_v1_table1_reproduction as vrep


THIN_ROOT = P.ROOT / "data/outputs/uidx/paperA_thinned"
OUT_CSV = P.PAPERA_ROOT / "diagnostics/din_thin_shift.csv"


def process(product: str, year: int, season: str,
            col_id_to_cell: dict[int, str]) -> pd.DataFrame:
    """Per (cell, product, year, season), compute v1 and thinned D̄_in medians."""
    fp = P.feature_root() / product / f"{product}_{year}_{season}.parquet"
    if not fp.exists():
        return pd.DataFrame()

    col_ids = sorted(col_id_to_cell.keys())
    table = pq.read_table(
        fp,
        columns=vrep.REQ_COLS,
        filters=[("col_id", "in", col_ids), ("bp_accepted", "=", 1)],
    )
    df = table.to_pandas()
    if df.empty:
        return pd.DataFrame()
    df["cell"] = df["col_id"].map(col_id_to_cell)
    rows = []
    for (cell, date, cid), g in df.groupby(["cell", "date", "col_id"], sort=False):
        first = g.iloc[0]
        xb, bg = first["x_b_km"], first["bg_g1"]
        if not (np.isfinite(xb) and xb > 0 and np.isfinite(bg)):
            continue
        ain = vrep.integrate_group(
            g["distance_km"].to_numpy(dtype=float),
            g["sst_k"].to_numpy(dtype=float),
            float(bg), float(xb),
        )
        rows.append({"cell": cell, "date": date, "col_id": int(cid), "D_in": ain})
    per_day = pd.DataFrame(rows)
    if per_day.empty:
        return pd.DataFrame()

    # Join with Task 2.2 retained_bool
    thin_fp = THIN_ROOT / product / f"{product}_{year}_{season}.parquet"
    thn = pd.read_parquet(thin_fp)[["cell", "date", "col_id", "retained_bool"]]
    per_day = per_day.merge(thn, on=["cell", "date", "col_id"], how="left")
    per_day["retained_bool"] = per_day["retained_bool"].fillna(0).astype(int)

    # Per-(cell, season, year) medians under both filters
    grp = per_day.groupby("cell", as_index=False).agg(
        D_in_v1=("D_in", lambda s: float(np.nanmedian(s))),
    )
    retained = per_day[per_day["retained_bool"] == 1].groupby("cell", as_index=False).agg(
        D_in_thin=("D_in", lambda s: float(np.nanmedian(s))),
    )
    grp = grp.merge(retained, on="cell", how="left")
    grp["product"] = product
    grp["year"] = year
    grp["season"] = season
    return grp


def main() -> None:
    membership = pd.read_csv(P.paperA_cell_transects_path())
    col_id_to_cell = dict(zip(membership["col_id"], membership["cell"]))
    parts = []
    t_start = time.monotonic()
    for product in P.PRODUCTS:
        for year in P.years_for(product):
            for season in P.SEASONS:
                rs = process(product, year, season, col_id_to_cell)
                if rs.empty:
                    continue
                parts.append(rs)
                print(f"  [{product} {year} {season}] {len(rs)} cell-rows  "
                      f"({time.monotonic() - t_start:.0f}s elapsed)", flush=True)
    if not parts:
        raise SystemExit("no rows produced")
    per_year_season = pd.concat(parts, ignore_index=True)
    climo = per_year_season.groupby(["cell", "product", "season"], as_index=False).agg(
        D_in_v1_median=("D_in_v1", "median"),
        D_in_thin_median=("D_in_thin", "median"),
    )
    climo["shift_C"] = climo["D_in_thin_median"] - climo["D_in_v1_median"]
    climo["abs_shift_C"] = climo["shift_C"].abs()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    climo.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n[output] {OUT_CSV}\n")
    print(climo.to_string(index=False))

    print("\n=== Summary: max |shift_C| per cell ===")
    summary = climo.groupby("cell")["abs_shift_C"].agg(["max", "median"]).round(3)
    print(summary.to_string())


if __name__ == "__main__":
    main()
