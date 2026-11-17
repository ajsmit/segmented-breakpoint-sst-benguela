#!/usr/bin/env python3
"""paperA_sensitivity_grid.py -- Stage 4 Task 4.1: six-dimensional
threshold sensitivity grid.

For each axis, varies the named threshold across a small grid while
holding the other axes at their baseline. Re-classifies per-profile
acceptance under the varied thresholds and aggregates three of the
four headline statistics per (cell, product, axis, axis_value):

  p_active             : mean per-profile acceptance fraction
  r_Ain_UIsst_d50      : per-(cell, product) Pearson r between the
                         cell-day median D_in and the cell-day median
                         UI_SST(50) = T(50 km) - T(coast)
  P_d0_lt_xb_d50       : per-(cell, product) fraction of accepted
                         profile-days with x_b > 50 km

The fourth headline statistic, sst_only_fraction_sBUS, requires the
wind detector (Stage 5) and is left as NaN here; Stage 5 will populate.

Six axes:
  Axis 1: DELTA_AIC bar in {2, 4, 6, 8}                  (baseline 4)
  Axis 2: slope ratio  in {1.5, 2.0, 2.5, 3.0}           (baseline 2)
  Axis 3: min offshore km in {50, 100, 150}              (baseline 100)
  Axis 4: edge buffer km in {25, 50, 75}                 (baseline 50)
            (note: addendum text says 10/50/75 here; the upstream
            BP_EDGE_BUFFER_KM constant in 07_build_feature_stack.py is 10.0,
            so '50' really means 'apply x_b >= 50 km' post-hoc, NOT the
            upstream's edge buffer)
  Axis 5: pixel-thinning rule (baseline = within-transect native;
            alternative = Stage 2 across-transect dedup -> require
            retained_bool == 1)
  Axis 6: p_active threshold in {0.3, 0.5, 0.7}          (baseline 0.5)
            -- affects the cell-day "active" classification, not the
            per-profile acceptance, so it does not change the three
            headline statistics computed here. Reported for completeness.

Outputs:
  data/outputs/uidx/paperA_sensitivity/grid_results.parquet
    one row per (cell, product, axis, axis_value)
  results/diagnostics/sensitivity_grid_long.csv (same data, csv)

Verification: at baseline (every axis at its baseline value), the
headline statistics match the manuscript's reported values to within
rounding (Luderitz OSTIA r_50 approx -0.09).

Run from the release repository root:
    python3 analysis/paperA_sensitivity_grid.py
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


# ---- Baseline thresholds (matching upstream 07_build_feature_stack.py constants)
BASELINE_DELTA_AIC = 4.0
BASELINE_SLOPE_RATIO = 2.0
BASELINE_MIN_OFFSHORE_KM = 100.0
BASELINE_EDGE_BUFFER_KM = 10.0  # upstream value
BASELINE_PIXEL_RULE = "across_thinned"
BASELINE_P_ACTIVE_THRESHOLD = 0.5

# ---- Grid values per axis
AXIS_DEFS = {
    "delta_aic_bar":    [2.0, 4.0, 6.0, 8.0],
    "slope_ratio":      [1.5, 2.0, 2.5, 3.0],
    "min_offshore_km":  [50.0, 100.0, 150.0, 200.0],
    "edge_buffer_km":   [10.0, 25.0, 50.0, 75.0],
    "pixel_rule":       ["native", "across_thinned"],
    "p_active_threshold": [0.3, 0.5, 0.7],
}
AXIS_BASELINE = {
    "delta_aic_bar":    BASELINE_DELTA_AIC,
    "slope_ratio":      BASELINE_SLOPE_RATIO,
    "min_offshore_km":  BASELINE_MIN_OFFSHORE_KM,
    "edge_buffer_km":   BASELINE_EDGE_BUFFER_KM,
    "pixel_rule":       BASELINE_PIXEL_RULE,
    "p_active_threshold": BASELINE_P_ACTIVE_THRESHOLD,
}

OUT_PARQUET = P.ROOT / "data/outputs/uidx/paperA_sensitivity/grid_results.parquet"
OUT_CSV = P.PAPERA_ROOT / "diagnostics/sensitivity_grid_long.csv"
INTERMEDIATE_PARQUET = P.ROOT / "data/outputs/uidx/paperA_sensitivity/per_profile_intermediate.parquet"

THIN_ROOT = P.ROOT / "data/outputs/uidx/paperA_thinned"


def build_intermediate(membership: pd.DataFrame) -> pd.DataFrame:
    """Per (col_id, date) row across the operational period, extract:

      D_in        closed-form approximation D_in = bg_g1 - alpha_in - 0.5 * beta_in * x_b
                  (exact when the inshore linear segment matches the SST; near-exact
                  in practice; full-integral version in paperA_v1_table1_reproduction).
      UI_SST_d50  T(distance closest to 50 km) - T(distance closest to 0 km),
                  cool-coastal-positive (matches the manuscript convention).
      x_b, x_max, delta_aic, slope_ratio, beta_in, beta_off, n_eff, retained_bool.

    Vectorised over (col_id, date) groups via pyarrow + a single per-file
    aggregation pass; ~ten seconds per file vs ~minutes for the per-group
    Python loop.
    """
    col_id_to_cell = dict(zip(membership["col_id"], membership["cell"]))
    col_ids = sorted(col_id_to_cell.keys())

    rows_parts = []
    for product in P.PRODUCTS:
        for year in P.years_for(product):
            for season in P.SEASONS:
                fp = P.feature_root() / product / f"{product}_{year}_{season}.parquet"
                if not fp.exists():
                    continue
                t0 = time.monotonic()
                tbl = pq.read_table(
                    fp,
                    columns=["col_id", "date", "distance_km", "sst_k",
                             "bp_accepted", "alpha_inshore", "beta_inshore",
                             "beta_offshore", "x_b_km", "bg_g1",
                             "delta_aic", "n_eff"],
                    filters=[("col_id", "in", col_ids)],
                ).to_pandas()
                if tbl.empty:
                    continue
                # First-row-per-group for the constant scalars
                tbl = tbl.sort_values(["col_id", "date", "distance_km"], kind="mergesort")
                # Per-(col_id, date) UI_SST_d50: pick the row whose |distance - 0|
                # is smallest, and whose |distance - 50| is smallest.
                tbl["d_abs_0"]  = tbl["distance_km"].abs()
                tbl["d_abs_50"] = (tbl["distance_km"] - 50.0).abs()
                tbl_valid = tbl[np.isfinite(tbl["sst_k"])].copy()
                # idxmin of d_abs_0 / d_abs_50 per group
                idx0  = tbl_valid.groupby(["col_id", "date"])["d_abs_0"].idxmin()
                idx50 = tbl_valid.groupby(["col_id", "date"])["d_abs_50"].idxmin()
                t_coast    = tbl_valid.loc[idx0,  ["col_id", "date", "sst_k"]].set_index(["col_id", "date"])["sst_k"]
                t_d50      = tbl_valid.loc[idx50, ["col_id", "date", "sst_k"]].set_index(["col_id", "date"])["sst_k"]
                ui_sst_d50 = (t_d50 - t_coast).rename("ui_sst_d50")
                # Per-(col_id, date) x_max from valid-SST rows
                x_max = tbl_valid.groupby(["col_id", "date"])["distance_km"].max().rename("x_max")
                # First-row scalars (constant within group)
                first = tbl.drop_duplicates(["col_id", "date"], keep="first").set_index(["col_id", "date"])
                merged = pd.concat(
                    [first[["bp_accepted", "alpha_inshore", "beta_inshore",
                            "beta_offshore", "x_b_km", "bg_g1", "delta_aic", "n_eff"]],
                     ui_sst_d50, x_max], axis=1
                ).reset_index()
                # Closed-form D_in approximation
                xb       = merged["x_b_km"]
                beta_in  = merged["beta_inshore"]
                alpha_in = merged["alpha_inshore"]
                bg       = merged["bg_g1"]
                merged["D_in"] = np.where(
                    (merged["bp_accepted"] == 1)
                    & np.isfinite(xb) & (xb > 0)
                    & np.isfinite(alpha_in) & np.isfinite(beta_in) & np.isfinite(bg),
                    bg - alpha_in - 0.5 * beta_in * xb,
                    np.nan,
                )
                # |beta_in / beta_off| slope ratio
                beta_off = merged["beta_offshore"]
                merged["slope_ratio"] = np.where(
                    np.isfinite(beta_in) & np.isfinite(beta_off) & (beta_off.abs() > 1e-9),
                    (beta_in / beta_off).abs(),
                    np.nan,
                )
                merged["cell"] = merged["col_id"].map(col_id_to_cell)
                merged["product"] = product
                merged["year"] = year
                merged["season"] = season
                keep = ["cell", "product", "year", "season", "date", "col_id",
                        "bp_accepted", "delta_aic", "slope_ratio", "x_b_km",
                        "x_max", "ui_sst_d50", "D_in", "n_eff"]
                rows_parts.append(merged[keep])
                print(f"  [intermediate {product} {year} {season}] "
                      f"{len(merged):>5,} rows  {time.monotonic() - t0:.1f}s",
                      flush=True)

    df = pd.concat(rows_parts, ignore_index=True)
    # Join with retained_bool from Stage 2 for axis 5
    thin_parts = []
    for product in P.PRODUCTS:
        for year in P.years_for(product):
            for season in P.SEASONS:
                tfp = THIN_ROOT / product / f"{product}_{year}_{season}.parquet"
                if tfp.exists():
                    thin_parts.append(pd.read_parquet(tfp)[
                        ["cell", "product", "date", "col_id", "retained_bool"]
                    ])
    thn = pd.concat(thin_parts, ignore_index=True)
    df = df.merge(thn, on=["cell", "product", "date", "col_id"], how="left")
    df["retained_bool"] = df["retained_bool"].fillna(0).astype(np.int8)
    INTERMEDIATE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INTERMEDIATE_PARQUET, index=False)
    print(f"\n[intermediate] {len(df):,} rows -> {INTERMEDIATE_PARQUET}")
    return df


def reclassify(df: pd.DataFrame,
               delta_aic_bar: float, slope_ratio: float,
               min_offshore_km: float, edge_buffer_km: float,
               pixel_rule: str) -> pd.Series:
    """Return 0/1 accepted-under-this-grid-point per row."""
    delta_ok    = df["delta_aic"] >= delta_aic_bar
    slope_ok    = df["slope_ratio"] >= slope_ratio
    offshore_ok = (df["x_max"] - df["x_b_km"]) >= min_offshore_km
    edge_ok     = df["x_b_km"] >= edge_buffer_km
    valid       = np.isfinite(df["delta_aic"]) & np.isfinite(df["x_b_km"]) \
                  & np.isfinite(df["slope_ratio"]) & np.isfinite(df["x_max"])
    accepted = (delta_ok & slope_ok & offshore_ok & edge_ok & valid).astype(int)
    if pixel_rule == "across_thinned":
        accepted = accepted * df["retained_bool"]
    return accepted.astype(int)


def aggregate_grid_point(df: pd.DataFrame, accepted_col: pd.Series) -> pd.DataFrame:
    """Per (cell, product), aggregate p_active, r(D_in, UI_SST_d50) on
    cell-day medians, P(x_b > 50 km) on accepted profiles."""
    work = df.assign(accepted=accepted_col)
    # p_active = mean of per-profile acceptance across (cell, product)
    p_active = work.groupby(["cell", "product"])["accepted"].mean().rename("p_active")
    n_accepted = work.groupby(["cell", "product"])["accepted"].sum().rename("n_accepted")
    n_retained = work.groupby(["cell", "product"])["retained_bool"].sum().rename("n_retained")

    # Cell-day median D_in and UI_SST_d50 over ACCEPTED profiles
    acc = work[work["accepted"] == 1]
    if acc.empty:
        cell_day = pd.DataFrame(columns=["cell", "product", "date",
                                         "D_in_med", "UI_SST_d50_med"])
        r_vals = pd.Series(np.nan, name="r_Ain_UIsst_d50")
        p_lt_xb = pd.Series(np.nan, name="P_d0_lt_xb_d50")
    else:
        cell_day = (acc.groupby(["cell", "product", "date"])
                      .agg(D_in_med=("D_in", "median"),
                           UI_SST_d50_med=("ui_sst_d50", "median"))
                      .reset_index())
        # Pearson r per (cell, product) on the cell-day median pairs.
        def _r(g):
            x = g["D_in_med"].to_numpy()
            y = g["UI_SST_d50_med"].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                return np.nan
            return float(np.corrcoef(x[mask], y[mask])[0, 1])
        r_vals = cell_day.groupby(["cell", "product"]).apply(
            _r, include_groups=False
        ).rename("r_Ain_UIsst_d50")
        # P(x_b > 50 km) on accepted profile-days
        p_lt_xb = acc.groupby(["cell", "product"]).apply(
            lambda g: float((g["x_b_km"] > 50.0).mean()),
            include_groups=False,
        ).rename("P_d0_lt_xb_d50")

    out = pd.concat([p_active, n_accepted, n_retained, r_vals, p_lt_xb],
                    axis=1).reset_index()
    return out


def main() -> None:
    membership = pd.read_csv(P.paperA_cell_transects_path())
    if INTERMEDIATE_PARQUET.exists():
        df = pd.read_parquet(INTERMEDIATE_PARQUET)
        print(f"[intermediate] loaded {len(df):,} rows from {INTERMEDIATE_PARQUET}\n")
    else:
        print("[intermediate] not present; building from features_bp\n")
        df = build_intermediate(membership)
    print()

    parts = []
    for axis, values in AXIS_DEFS.items():
        for v in values:
            kwargs = {a: AXIS_BASELINE[a] for a in AXIS_DEFS if a != axis}
            kwargs[axis] = v
            # Only the four "operational" axes affect acceptance directly;
            # axis 6 (p_active_threshold) is a downstream filter and doesn't
            # change profile-level acceptance, so we use the baseline thresholds.
            accepted = reclassify(
                df,
                delta_aic_bar    = kwargs["delta_aic_bar"],
                slope_ratio      = kwargs["slope_ratio"],
                min_offshore_km  = kwargs["min_offshore_km"],
                edge_buffer_km   = kwargs["edge_buffer_km"],
                pixel_rule       = kwargs["pixel_rule"],
            )
            agg = aggregate_grid_point(df, accepted)
            agg["axis"] = axis
            agg["axis_value"] = str(v)
            agg["is_baseline"] = (v == AXIS_BASELINE[axis])
            parts.append(agg)
            print(f"  [{axis:<22s} = {v!s:<14s}] {len(agg)} (cell, product) rows", flush=True)

    grid = pd.concat(parts, ignore_index=True)
    grid = grid[["cell", "product", "axis", "axis_value", "is_baseline",
                 "p_active", "r_Ain_UIsst_d50", "P_d0_lt_xb_d50",
                 "n_accepted", "n_retained"]]
    for c in ["p_active", "r_Ain_UIsst_d50", "P_d0_lt_xb_d50"]:
        grid[c] = grid[c].astype(float).round(4)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(OUT_PARQUET, index=False)
    grid.to_csv(OUT_CSV, index=False)
    print(f"\n[output parquet] {OUT_PARQUET}")
    print(f"[output csv]     {OUT_CSV}")

    # Verification: print the baseline row for each (cell, product) at the
    # delta_aic_bar axis to compare against the manuscript's headline numbers.
    print("\n=== Baseline values (delta_aic_bar = 4) ===")
    baseline_rows = grid[(grid["axis"] == "delta_aic_bar") & (grid["axis_value"] == "4.0")]
    print(baseline_rows[["cell", "product", "p_active", "r_Ain_UIsst_d50",
                          "P_d0_lt_xb_d50", "n_accepted"]].to_string(index=False))


if __name__ == "__main__":
    main()
