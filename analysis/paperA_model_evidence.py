#!/usr/bin/env python3
"""paperA_model_evidence.py -- Stage 3 Task 3.1: compute AICc and BIC
alongside AIC for the one- and two-segment fits.

Addendum task: Stage 3, Task 3.1.

Reads the recovered features_bp tree (one row per (col_id, date, sigma)),
extracts the per-(col_id, date) RSS scalars and n_eff, and emits the
three information-criterion scores for both models:

  AIC  = n * log(RSS / n) + 2 * k
  AICc = AIC + 2 * k * (k + 1) / (n - k - 1)
  BIC  = n * log(RSS / n) + k * log(n)

with K_ONE_SEGMENT = 2 and K_TWO_SEGMENT = 4 (continuity-constrained
two-segment model; alpha_off is derived, not fitted).

Output is restricted to cell-window col_ids (189 transects across
5 cells) per paperA_cell_transects.csv, so the file stays small.

Outputs:
  data/outputs/uidx/paperA_model_evidence/<PRODUCT>/<PRODUCT>_<YEAR>_<SEASON>.parquet
  with columns
  (cell, product, date, col_id, n_eff, bp_accepted,
   rss_one_segment_K2, rss_two_segment_K2,
   aic_one_segment, aic_two_segment, delta_aic,
   aicc_one_segment, aicc_two_segment, delta_aicc,
   bic_one_segment, bic_two_segment, delta_bic).

Verification: aic_two_segment matches the upstream value to <= 1e-6 on
the smoke subset.

Run from the release repository root:
    python3 analysis/paperA_model_evidence.py
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


OUT_ROOT = P.ROOT / "data/outputs/uidx/paperA_model_evidence"


def aic(n: np.ndarray, rss: np.ndarray, k: int) -> np.ndarray:
    """Upstream-compatible AIC: n * log(rss / n) + 2 * k. NaN where invalid."""
    out = np.full(n.shape, np.nan, dtype=float)
    valid = np.isfinite(n) & np.isfinite(rss) & (n > 0) & (rss > 0)
    nv = n[valid]
    rv = rss[valid]
    out[valid] = nv * np.log(rv / nv) + 2.0 * k
    return out


def aicc(aic_vals: np.ndarray, n: np.ndarray, k: int) -> np.ndarray:
    """AICc = AIC + 2k(k+1) / (n - k - 1). NaN where n - k - 1 <= 0."""
    out = np.full(aic_vals.shape, np.nan, dtype=float)
    valid = np.isfinite(aic_vals) & np.isfinite(n) & (n - k - 1 > 0)
    nv = n[valid]
    out[valid] = aic_vals[valid] + (2.0 * k * (k + 1)) / (nv - k - 1)
    return out


def bic(n: np.ndarray, rss: np.ndarray, k: int) -> np.ndarray:
    """BIC = n * log(rss / n) + k * log(n). NaN where invalid."""
    out = np.full(n.shape, np.nan, dtype=float)
    valid = np.isfinite(n) & np.isfinite(rss) & (n > 0) & (rss > 0)
    nv = n[valid]
    rv = rss[valid]
    out[valid] = nv * np.log(rv / nv) + k * np.log(nv)
    return out


def process(product: str, year: int, season: str,
            col_id_to_cell: dict[int, str]) -> tuple[int, float]:
    src = P.feature_root() / product / f"{product}_{year}_{season}.parquet"
    if not src.exists():
        return (0, 0.0)
    t0 = time.monotonic()
    col_ids = sorted(col_id_to_cell.keys())
    tbl = pq.read_table(
        src,
        columns=["col_id", "date", "sigma", "n_eff", "bp_accepted",
                 "rss_one_segment_K2", "rss_two_segment_K2",
                 "aic_one_segment", "aic_two_segment", "delta_aic"],
        filters=[("col_id", "in", col_ids)],
    )
    df = tbl.to_pandas()
    if df.empty:
        return (0, time.monotonic() - t0)
    # Collapse to one row per (col_id, date) — the breakpoint scalars are
    # constant within each (col_id, date) group, so .first() works.
    df = df.sort_values(["col_id", "date", "sigma"], kind="mergesort")
    df = df.drop_duplicates(["col_id", "date"], keep="first")
    df = df.drop(columns="sigma")

    n = df["n_eff"].to_numpy(dtype=float)
    r1 = df["rss_one_segment_K2"].to_numpy(dtype=float)
    r2 = df["rss_two_segment_K2"].to_numpy(dtype=float)

    df["aic_one_segment_paperA"]  = aic(n, r1, P.K_ONE_SEGMENT)
    df["aic_two_segment_paperA"]  = aic(n, r2, P.K_TWO_SEGMENT)
    df["aicc_one_segment"]        = aicc(df["aic_one_segment_paperA"].to_numpy(), n, P.K_ONE_SEGMENT)
    df["aicc_two_segment"]        = aicc(df["aic_two_segment_paperA"].to_numpy(), n, P.K_TWO_SEGMENT)
    df["bic_one_segment"]         = bic(n, r1, P.K_ONE_SEGMENT)
    df["bic_two_segment"]         = bic(n, r2, P.K_TWO_SEGMENT)

    df["delta_aic_paperA"]  = df["aic_one_segment_paperA"]  - df["aic_two_segment_paperA"]
    df["delta_aicc"]        = df["aicc_one_segment"]        - df["aicc_two_segment"]
    df["delta_bic"]         = df["bic_one_segment"]         - df["bic_two_segment"]

    df["cell"] = df["col_id"].map(col_id_to_cell)
    df["product"] = product
    df["year"] = year
    df["season"] = season

    cols = [
        "cell", "product", "year", "season", "date", "col_id", "n_eff", "bp_accepted",
        "rss_one_segment_K2", "rss_two_segment_K2",
        "aic_one_segment", "aic_two_segment", "delta_aic",
        "aic_one_segment_paperA", "aic_two_segment_paperA", "delta_aic_paperA",
        "aicc_one_segment", "aicc_two_segment", "delta_aicc",
        "bic_one_segment", "bic_two_segment", "delta_bic",
    ]
    out = df[cols]
    dst = OUT_ROOT / product / f"{product}_{year}_{season}.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst, index=False)
    return (len(out), time.monotonic() - t0)


def main() -> None:
    membership = pd.read_csv(P.paperA_cell_transects_path())
    col_id_to_cell = dict(zip(membership["col_id"], membership["cell"]))
    print(f"[mesh] {P.MESH_TAG}; {len(col_id_to_cell)} cell-window col_ids")
    print(f"[features_bp] {P.feature_root()}")
    print(f"[out] {OUT_ROOT}\n")
    total_rows = 0
    t_start = time.monotonic()
    for product in P.PRODUCTS:
        for year in P.years_for(product):
            for season in P.SEASONS:
                n_rows, dt = process(product, year, season, col_id_to_cell)
                total_rows += n_rows
                if n_rows:
                    print(f"  [{product} {year} {season}] {n_rows:>5,} rows  {dt:.1f}s",
                          flush=True)
    print(f"\n[total] {total_rows:,} rows in {time.monotonic() - t_start:.0f}s")


if __name__ == "__main__":
    main()
