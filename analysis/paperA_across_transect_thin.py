#!/usr/bin/env python3
"""paperA_across_transect_thin.py -- Stage 2 Task 2.2: first-occurrence-wins
pixel-stride deduplication across transects within each UC window.

Addendum task: Stage 2, Task 2.2 (across-transect pixel-stride thinning).

For each (cell, product, date), accepted+rejected transects are sorted by
col_id (along-coast index in the r2 mesh) and the L4-pixel coverage is
deduplicated by first occurrence: a (pix_i, pix_j) tuple is "claimed" by
the earliest col_id whose streamline visits it; later col_ids that
revisit the same pixel are discounted.

A transect is "retained" iff it contributes at least one unique L4 pixel
to the cell-day after this deduplication. Transects whose pixel coverage
is fully claimed by inboard neighbours are dropped (retained_bool = 0).

Output: one row per (cell, product, date, col_id) with columns
(retained_bool, n_unique_pixels, total_pixels_in_transect, accepted).

Inputs: data/outputs/uidx/paperA_pixel_index/<PRODUCT>/<PRODUCT>_<YEAR>_<SEASON>.parquet
Outputs: data/outputs/uidx/paperA_thinned/<PRODUCT>/<PRODUCT>_<YEAR>_<SEASON>.parquet

Verification (set-equality):
  For each (cell, product, date), the union of unique pixels across
  retained transects equals the union across all transects (the dedup
  only drops zero-new-pixel transects, not pixels). No L4 pixel appears
  in more than one retained transect on the same date (by construction).

Run from the release repository root:
    python3 analysis/paperA_across_transect_thin.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperA_common as P


SRC_ROOT = P.ROOT / "data/outputs/uidx/paperA_pixel_index"
OUT_ROOT = P.ROOT / "data/outputs/uidx/paperA_thinned"


def process(product: str, year: int, season: str) -> tuple[int, float]:
    src = SRC_ROOT / product / f"{product}_{year}_{season}.parquet"
    if not src.exists():
        return (0, 0.0)
    t0 = time.monotonic()

    df = pd.read_parquet(src)
    if df.empty:
        return (0, time.monotonic() - t0)

    # Sort once. Within-(cell, product, date) ascending col_id then sigma is
    # the canonical "inboard-first" order; the within-(col_id, date) dedup
    # was already done in Task 2.1, but we sort by sigma here for stability
    # at the across-transect step.
    df = df.sort_values(["cell", "product", "date", "col_id", "sigma"], kind="mergesort")

    # First-occurrence across the (cell, product, date) group on (pix_i, pix_j):
    # the surviving rows are the "unique-pixel claims".
    surv = df.drop_duplicates(["cell", "product", "date", "pix_i", "pix_j"], keep="first")

    # n_unique_pixels per (col_id, date): how many pixels did this transect
    # claim after the across-transect dedup?
    n_unique = surv.groupby(["cell", "product", "date", "col_id"]).size().rename("n_unique_pixels")

    # total_pixels_in_transect: how many pixels did this transect cover before
    # the across-transect dedup (Task 2.1 within-transect dedup applied)?
    n_total = df.groupby(["cell", "product", "date", "col_id"]).size().rename("total_pixels_in_transect")

    # accepted: bp_accepted (0/1). Constant within (col_id, date) by definition.
    accepted = df.groupby(["cell", "product", "date", "col_id"])["accepted"].first()

    out = (
        pd.concat([n_total, n_unique, accepted], axis=1)
          .reset_index()
    )
    out["n_unique_pixels"] = out["n_unique_pixels"].fillna(0).astype(np.int32)
    out["total_pixels_in_transect"] = out["total_pixels_in_transect"].astype(np.int32)
    out["accepted"] = out["accepted"].astype(np.int8)
    out["retained_bool"] = (out["n_unique_pixels"] > 0).astype(np.int8)

    dst = OUT_ROOT / product / f"{product}_{year}_{season}.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst, index=False)
    return (len(out), time.monotonic() - t0)


def main() -> None:
    print(f"[src] {SRC_ROOT}")
    print(f"[out] {OUT_ROOT}\n")
    total_rows = 0
    t_start = time.monotonic()
    for product in P.PRODUCTS:
        for year in P.years_for(product):
            for season in P.SEASONS:
                n, dt = process(product, year, season)
                total_rows += n
                if n:
                    print(f"  [{product} {year} {season}] "
                          f"{n:>7,} (cell, date, col_id) rows  {dt:.1f}s",
                          flush=True)
    print(f"\n[total] {total_rows:,} rows in {time.monotonic() - t_start:.1f}s")


if __name__ == "__main__":
    main()
