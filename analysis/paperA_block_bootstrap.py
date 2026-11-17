#!/usr/bin/env python3
"""paperA_block_bootstrap.py -- Stage 6 Task 6.1: alongshore-block
bootstrap 95 % intervals for the three headline statistics.

Block-bootstrap design: within each cell, transects are partitioned
into alongshore blocks of length 25 km (baseline) and 50 km
(sensitivity). For each of 1000 resamples, blocks are sampled WITH
REPLACEMENT until the resampled set has at least as many transects as
the cell's original count. The resampled (transect, day) rows are
re-aggregated to cell-day medians, then the headline statistic is
recomputed.

Headline statistics resampled:
  L_r50    : annual Pearson r between D_in and UI_SST_d50 at Luderitz
             OSTIA, computed across days in the operational period.
  med_xb_<cell> : per-cell annual median x_b on OSTIA (5 cells).
  sBUS_sst_only : sBUS SST-structure-only daily fraction pooled across
             OSTIA + OSPO + 4 seasons (Cape Columbine + Cape Peninsula),
             evaluated under the thinned p_active >= 0.5 SST detector.

Inputs:
  - per-profile intermediate parquet from Stage 4 (per (cell, product,
    date, col_id) breakdown of D_in, UI_SST_d50, x_b, bp_accepted)
  - paperA_cell_transects.csv (cell-window col_ids per cell)
  - daily wind detector from Stage 3 (UI_w > 0 -> wind_active flag)

Outputs:
  results/diagnostics/block_bootstrap.parquet   (one row per resample)
  results/tables/table_s_block_bootstrap.csv    (per-stat summary)

Verification: 25 km and 50 km bootstrap intervals differ by less than
30 % in width. If 50 km is substantially wider, the spatial scale of
dependence exceeds 25 km and the 50 km interval is reported.

Run from the release repository root:
    python3 analysis/paperA_block_bootstrap.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperA_common as P


INTERMEDIATE = P.ROOT / "data/outputs/uidx/paperA_sensitivity/per_profile_intermediate.parquet"
THIN_DAILY = P.ROOT / "data/outputs/uidx/paperA_p_active_thinned/per_cell_product_day.parquet"
WIND_DAILY = P.ROOT / "data/outputs/uidx/uiw_lamont/daily_cell_wind.parquet"
THIN_ROOT = P.ROOT / "data/outputs/uidx/paperA_thinned"

OUT_PARQUET = P.PAPERA_ROOT / "diagnostics/block_bootstrap.parquet"
OUT_CSV = P.PAPERA_ROOT / "tables/table_s_block_bootstrap.csv"

N_RESAMPLES = 1000
BLOCK_KM_VALUES = [25.0, 50.0]
COAST_SPACING_KM = 3.7  # r2 mesh; transect spacing (approx)


def build_blocks(cell_transects: pd.DataFrame, block_km: float) -> dict[str, list[list[int]]]:
    """Per cell, partition col_ids into consecutive alongshore blocks of
    approximately block_km length each. Returns {cell: [[col_id, ...], ...]}."""
    n_per_block = max(1, int(round(block_km / COAST_SPACING_KM)))
    blocks = {}
    for cell, g in cell_transects.groupby("cell"):
        col_ids = sorted(g["col_id"].tolist())
        cell_blocks = [col_ids[i:i + n_per_block]
                       for i in range(0, len(col_ids), n_per_block)]
        blocks[cell] = cell_blocks
    return blocks


def resample_block_col_ids(blocks: list[list[int]],
                            target_n: int,
                            rng: np.random.Generator) -> list[int]:
    """Sample blocks with replacement until len(resampled_col_ids) >= target_n,
    then trim to target_n."""
    n_blocks = len(blocks)
    out = []
    while len(out) < target_n:
        b = blocks[rng.integers(0, n_blocks)]
        out.extend(b)
    return out[:target_n]


# --- Stat 1: Luderitz r_50 OSTIA ----------------------------------------------
def luderitz_r50(intermediate: pd.DataFrame,
                  resampled_col_ids: list[int]) -> float:
    """Across-transect pixel uniqueness is enforced operationally:
    only retained_bool == 1 transect-days contribute to the cell-day medians."""
    sub = intermediate[(intermediate["cell"] == "Luderitz")
                       & (intermediate["product"] == "OSTIA")
                       & (intermediate["bp_accepted"] == 1)
                       & (intermediate["retained_bool"] == 1)
                       & (intermediate["col_id"].isin(resampled_col_ids))].copy()
    if sub.empty:
        return np.nan
    cell_day = (sub.groupby("date")
                .agg(D_in_med=("D_in", "median"),
                     UI_SST_d50_med=("ui_sst_d50", "median"))
                .dropna())
    if len(cell_day) < 30:
        return np.nan
    return float(np.corrcoef(cell_day["D_in_med"], cell_day["UI_SST_d50_med"])[0, 1])


# --- Stat 2: per-cell median x_b on OSTIA ------------------------------------
def cell_median_xb(intermediate: pd.DataFrame, cell: str,
                    resampled_col_ids: list[int]) -> float:
    """Across-transect pixel uniqueness is enforced operationally:
    only retained_bool == 1 transect-days contribute to the median."""
    sub = intermediate[(intermediate["cell"] == cell)
                       & (intermediate["product"] == "OSTIA")
                       & (intermediate["bp_accepted"] == 1)
                       & (intermediate["retained_bool"] == 1)
                       & (intermediate["col_id"].isin(resampled_col_ids))]
    if sub.empty:
        return np.nan
    return float(sub["x_b_km"].median())


# --- Stat 3: sBUS SST-structure-only fraction --------------------------------
def sbus_sst_only_fraction(daily_paired: pd.DataFrame,
                            blocks: dict[str, list[list[int]]],
                            rng: np.random.Generator,
                            thinned_per_day_per_col: pd.DataFrame) -> float:
    """Recompute the cell-day p_active on resampled transects, the
    SST-active flag (>=0.5) per cell-day, and the four-regime fractions
    pooled across (Cape Columbine + Cape Peninsula) x (OSTIA + OSPO) x
    4 seasons. Returns the sst_only fraction.
    """
    # For each sBUS cell × product, recompute per-(date) p_active from
    # the resampled transects.
    parts = []
    for cell in ("Cape Columbine", "Cape Peninsula"):
        if cell not in blocks:
            continue
        target_n = sum(len(b) for b in blocks[cell])
        resampled_col_ids = resample_block_col_ids(blocks[cell], target_n, rng)
        for product in ("OSTIA", "OSPO"):
            sub = thinned_per_day_per_col[
                (thinned_per_day_per_col["cell"] == cell)
                & (thinned_per_day_per_col["product"] == product)
                & (thinned_per_day_per_col["col_id"].isin(resampled_col_ids))
            ]
            if sub.empty:
                continue
            # New p_active per (cell, product, date) on the resampled set
            grp = sub.groupby(["cell", "product", "date"]).agg(
                n_transects=("col_id", "count"),
                n_accepted=("accepted", "sum"),
            ).reset_index()
            grp["p_active_resampled"] = grp["n_accepted"] / grp["n_transects"]
            parts.append(grp)
    if not parts:
        return np.nan
    daily = pd.concat(parts, ignore_index=True)
    daily["sst_active"] = (daily["p_active_resampled"] >= 0.5).astype(int)
    daily["date"] = pd.to_datetime(daily["date"])
    merged = daily.merge(daily_paired[["cell", "date", "wind_active"]],
                         on=["cell", "date"], how="inner")
    if merged.empty:
        return np.nan
    so = int(((merged["sst_active"] == 1) & (merged["wind_active"] == 0)).sum())
    return float(so / len(merged))


def main() -> None:
    print(f"[src] {INTERMEDIATE}")
    inter = pd.read_parquet(INTERMEDIATE)
    inter["date"] = pd.to_datetime(inter["date"])
    print(f"[loaded] {len(inter):,} (cell, product, date, col_id) rows\n")

    # Stage 2 thinned per-(transect, day) for sBUS sst_only recompute
    thin_parts = []
    for product in P.PRODUCTS:
        for year in P.years_for(product):
            for season in P.SEASONS:
                tfp = THIN_ROOT / product / f"{product}_{year}_{season}.parquet"
                if tfp.exists():
                    thin_parts.append(pd.read_parquet(tfp)[
                        ["cell", "product", "date", "col_id", "accepted", "retained_bool"]
                    ])
    thn = pd.concat(thin_parts, ignore_index=True)
    thn["date"] = pd.to_datetime(thn["date"])
    print(f"[loaded] {len(thn):,} thinned per-(cell, product, date, col_id) rows\n")

    # Wind daily
    wind = pd.read_parquet(WIND_DAILY)
    wind["date"] = pd.to_datetime(wind["date"])
    wind["wind_active"] = (wind["UI_w_m3_s_per_100m"] > 0.0).astype(int)
    wind = wind[["cell", "date", "wind_active"]]

    membership = pd.read_csv(P.paperA_cell_transects_path())

    # Observed (point) statistics — baseline reference
    print("=== Observed (point) statistics ===")
    luderitz_obs = luderitz_r50(inter, membership[membership["cell"] == "Luderitz"]["col_id"].tolist())
    print(f"  Luderitz OSTIA r_50 (observed): {luderitz_obs:+.4f}")
    obs_xb = {cell: cell_median_xb(inter, cell, membership[membership["cell"] == cell]["col_id"].tolist())
              for cell in P.CELLS}
    print(f"  Per-cell median x_b (OSTIA observed):")
    for cell, v in obs_xb.items():
        print(f"    {cell:18s}: {v:.2f} km" if np.isfinite(v) else f"    {cell}: NaN")

    rng = np.random.default_rng(2026)
    # Observed sBUS sst_only fraction on full data (no resampling)
    sbus_blocks_full = build_blocks(membership, 25.0)
    obs_sbus = sbus_sst_only_fraction(wind, sbus_blocks_full, rng, thn)
    # Note: sbus_sst_only_fraction does resampling internally; with full blocks
    # at baseline the call still resamples — we override by passing the full
    # col_ids deterministically:
    sbus_obs_parts = []
    for cell in ("Cape Columbine", "Cape Peninsula"):
        cell_col_ids = membership[membership["cell"] == cell]["col_id"].tolist()
        for product in ("OSTIA", "OSPO"):
            sub = thn[(thn["cell"] == cell) & (thn["product"] == product)
                      & (thn["col_id"].isin(cell_col_ids))]
            grp = sub.groupby(["cell", "product", "date"]).agg(
                n_transects=("col_id", "count"),
                n_accepted=("accepted", "sum"),
            ).reset_index()
            grp["sst_active"] = ((grp["n_accepted"] / grp["n_transects"]) >= 0.5).astype(int)
            sbus_obs_parts.append(grp[["cell", "product", "date", "sst_active"]])
    sbus_obs_daily = pd.concat(sbus_obs_parts, ignore_index=True)
    sbus_obs_merged = sbus_obs_daily.merge(wind, on=["cell", "date"], how="inner")
    sbus_obs_fraction = float(((sbus_obs_merged["sst_active"] == 1) &
                                (sbus_obs_merged["wind_active"] == 0)).sum()
                               / len(sbus_obs_merged))
    print(f"\n  sBUS SST-only fraction (observed): {sbus_obs_fraction:.4f}")

    # Resample loops
    print("\n=== Block-bootstrap resamples ===")
    rows = []
    for block_km in BLOCK_KM_VALUES:
        blocks = build_blocks(membership, block_km)
        per_cell_n = {c: sum(len(b) for b in blocks[c]) for c in blocks}
        rng = np.random.default_rng(2026)
        t0 = time.monotonic()
        for r in range(N_RESAMPLES):
            # Resample col_ids per cell
            cell_resampled = {c: resample_block_col_ids(blocks[c], per_cell_n[c], rng)
                              for c in blocks}
            # Stat 1
            lr = luderitz_r50(inter, cell_resampled["Luderitz"])
            # Stat 2
            xb = {c: cell_median_xb(inter, c, cell_resampled[c]) for c in P.CELLS}
            # Stat 3 — use sub-call that uses its own resampling (re-seeded with rng)
            # but we want a SHARED resample. Re-use cell_resampled directly:
            parts_sbus = []
            for cell in ("Cape Columbine", "Cape Peninsula"):
                rcid = cell_resampled[cell]
                for product in ("OSTIA", "OSPO"):
                    sub = thn[(thn["cell"] == cell) & (thn["product"] == product)
                              & (thn["col_id"].isin(rcid))]
                    grp = sub.groupby(["cell", "product", "date"]).agg(
                        n_transects=("col_id", "count"),
                        n_accepted=("accepted", "sum"),
                    ).reset_index()
                    grp["sst_active"] = ((grp["n_accepted"] / grp["n_transects"]) >= 0.5).astype(int)
                    parts_sbus.append(grp[["cell", "product", "date", "sst_active"]])
            sbus_daily = pd.concat(parts_sbus, ignore_index=True)
            sbus_merged = sbus_daily.merge(wind, on=["cell", "date"], how="inner")
            sbus_frac = (float(((sbus_merged["sst_active"] == 1) &
                                 (sbus_merged["wind_active"] == 0)).sum()
                                / len(sbus_merged))
                          if len(sbus_merged) else np.nan)
            row = dict(block_km=block_km, resample=r, L_r50=lr, sBUS_sst_only=sbus_frac)
            for c in P.CELLS:
                row[f"med_xb_{c.replace(' ', '_')}"] = xb[c]
            rows.append(row)
            if (r + 1) % 100 == 0:
                print(f"  [block {block_km:.0f} km] {r + 1}/{N_RESAMPLES} "
                      f"  ({time.monotonic() - t0:.0f}s)",
                      flush=True)
        print(f"  [block {block_km:.0f} km] DONE  total {time.monotonic() - t0:.0f}s")

    df = pd.DataFrame(rows)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\n[output parquet] {OUT_PARQUET}")

    # Summary
    stat_cols = ["L_r50", "sBUS_sst_only"] + [f"med_xb_{c.replace(' ', '_')}" for c in P.CELLS]
    summary = []
    for block_km in BLOCK_KM_VALUES:
        sub = df[df["block_km"] == block_km]
        for stat in stat_cols:
            vals = sub[stat].dropna().to_numpy()
            if vals.size < 10:
                continue
            observed = {
                "L_r50": luderitz_obs,
                "sBUS_sst_only": sbus_obs_fraction,
                **{f"med_xb_{c.replace(' ', '_')}": obs_xb[c] for c in P.CELLS},
            }[stat]
            summary.append({
                "stat": stat,
                "block_km": block_km,
                "observed": observed,
                "boot_p025": float(np.percentile(vals, 2.5)),
                "boot_p500": float(np.percentile(vals, 50)),
                "boot_p975": float(np.percentile(vals, 97.5)),
                "ci_width": float(np.percentile(vals, 97.5) - np.percentile(vals, 2.5)),
                "n_resamples": int(vals.size),
            })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"[output csv]     {OUT_CSV}\n")
    print(summary_df.to_string(index=False))

    # Verification: 25 vs 50 km CI width comparison
    print("\n=== Verification: 25 km vs 50 km CI width within 30 % ===")
    pivot = summary_df.pivot(index="stat", columns="block_km", values="ci_width")
    pivot["ratio_50_over_25"] = pivot[50.0] / pivot[25.0]
    print(pivot.round(4).to_string())
    bigger = pivot[pivot["ratio_50_over_25"] > 1.30]
    if bigger.empty:
        print("  ALL stats: 50 km CI width within 30 % of 25 km CI width (25 km adequate)")
    else:
        print(f"  {len(bigger)} stat(s) with 50 km CI substantially wider; "
              f"50 km is the preferred interval for those:")
        print(bigger.to_string())


if __name__ == "__main__":
    main()
