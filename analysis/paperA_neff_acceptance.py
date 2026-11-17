#!/usr/bin/env python3
"""paperA_neff_acceptance.py -- Stage 3 Task 3.5: convert AR(1) phi to
n_eff* and recompute the per-profile AIC. Tabulate the change in
acceptance rate per (cell, product, season) against the unique-pixel
baseline.

For each (cell, season) we use the phi fitted on OSTIA 2020 in Task 3.4
(single per-(cell, season) decorrelation across products, since the
products observe the same ocean). The per-profile effective sample
size is

    n_eff_star = n_eff * (1 - phi) / (1 + phi)

The recomputed AIC is the baseline AIC formula evaluated at n_eff_star:

    AIC_star = n_eff_star * log(RSS / n_eff_star) + 2 * k

The profile is accepted under the n_eff_star AIC iff Δ_AIC_star ≥ 4
AND the upstream operational checks (slope ratio, offshore length,
edge buffer) pass. Since the operational checks are independent of n,
we only need to re-evaluate the Δ_AIC_star part: an originally accepted
profile fails the n_eff_star test iff its Δ_AIC_star < 4.

Note: when n_eff_star < K_TWO_SEGMENT = 4, the AICc denominator
(n - k - 1) is <= 0 and AICc is undefined. We do not use AICc here;
we use the plain AIC formula which is defined for any n > 0. For n
much smaller than k, the AIC's discrimination collapses, and acceptance
under n_eff_star approaches zero.

Outputs:
  results/tables/table_s1_neff_ar1.csv
    columns: cell, product, season, n_baseline_median, phi, n_eff_star_median,
             p_active_baseline, p_active_neff, delta_p_active

Run from the release repository root:
    python3 analysis/paperA_neff_acceptance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperA_common as P
import paperA_model_evidence as M


PHI_CSV = P.PAPERA_ROOT / "diagnostics/ar1_residuals.csv"
OUT_CSV = P.PAPERA_ROOT / "tables/table_s1_neff_ar1.csv"


def load_phi() -> dict[tuple[str, str], float]:
    df = pd.read_csv(PHI_CSV)
    return {(r["cell"], r["season"]): float(r["phi"]) for _, r in df.iterrows()}


def main() -> None:
    phi_map = load_phi()
    print(f"[phi] {len(phi_map)} (cell, season) tuples loaded from {PHI_CSV}\n")

    src_root = P.ROOT / "data/outputs/uidx/paperA_model_evidence"
    parts = []
    for product in P.PRODUCTS:
        for year in P.years_for(product):
            for season in P.SEASONS:
                fp = src_root / product / f"{product}_{year}_{season}.parquet"
                if not fp.exists():
                    continue
                df = pd.read_parquet(fp)
                # Apply phi to compute n_eff_star and recomputed AIC
                df["phi"] = df.apply(
                    lambda r: phi_map.get((r["cell"], r["season"]), np.nan),
                    axis=1,
                )
                df["n_eff_star"] = df["n_eff"] * (1.0 - df["phi"]) / (1.0 + df["phi"])
                # Recompute AIC at n_eff_star (using float values for both formulas)
                n_star = df["n_eff_star"].to_numpy(dtype=float)
                r1 = df["rss_one_segment_K2"].to_numpy(dtype=float)
                r2 = df["rss_two_segment_K2"].to_numpy(dtype=float)
                df["aic_one_segment_neff"] = M.aic(n_star, r1, P.K_ONE_SEGMENT)
                df["aic_two_segment_neff"] = M.aic(n_star, r2, P.K_TWO_SEGMENT)
                df["delta_aic_neff"]       = df["aic_one_segment_neff"] - df["aic_two_segment_neff"]
                parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    print(f"[loaded] {len(full):,} profiles total\n")

    # Per (cell, product, season) aggregation
    grouped = full.groupby(["cell", "product", "season"], as_index=False).agg(
        n_baseline_median=("n_eff", "median"),
        phi=("phi", "first"),
        n_eff_star_median=("n_eff_star", "median"),
        n_profiles=("col_id", "size"),
        n_accepted_baseline=("bp_accepted", "sum"),
        n_accepted_neff=("delta_aic_neff",
                         lambda s: int((s >= P.DELTA_AIC_BAR).sum())),
    )
    # Operational p_active baseline = bp_accepted (which already includes the
    # operational slope-ratio + offshore-length + edge-buffer checks).
    # p_active_neff: an originally accepted profile is "accepted under n_eff_star"
    # iff Δ_AIC_star >= 4 (the operational checks are independent of n_eff).
    # We re-evaluate per-profile and count.
    full["accepted_neff"] = (
        (full["bp_accepted"] == 1) & (full["delta_aic_neff"] >= P.DELTA_AIC_BAR)
    ).astype(int)
    grouped["n_accepted_under_neff_AND_operational"] = (
        full.groupby(["cell", "product", "season"])["accepted_neff"].sum().to_numpy()
    )
    grouped["p_active_baseline"] = grouped["n_accepted_baseline"] / grouped["n_profiles"]
    grouped["p_active_neff"] = grouped["n_accepted_under_neff_AND_operational"] / grouped["n_profiles"]
    grouped["delta_p_active"] = grouped["p_active_neff"] - grouped["p_active_baseline"]

    grouped = grouped.sort_values(["cell", "product", "season"]).reset_index(drop=True)
    for c in ["n_baseline_median", "n_eff_star_median",
              "p_active_baseline", "p_active_neff", "delta_p_active", "phi"]:
        grouped[c] = grouped[c].astype(float).round(3)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"[output] {OUT_CSV}")
    print(grouped.to_string(index=False))

    # Per-product summary
    print("\n=== Per-product summary (mean across cells × seasons) ===")
    summary = grouped.groupby("product")[
        ["n_baseline_median", "n_eff_star_median", "p_active_baseline",
         "p_active_neff", "delta_p_active"]
    ].mean().round(3)
    print(summary.to_string())

    # Verification: acceptance under n_eff* should be NO HIGHER than baseline
    bad = grouped[grouped["delta_p_active"] > 0]
    print(f"\n=== Verification: delta_p_active <= 0 per (cell, product, season) ===")
    if bad.empty:
        print(f"  ALL PASS ({len(grouped)} tuples)")
    else:
        print(f"  {len(bad)} tuples violate (acceptance increased after n_eff* correction):")
        print(bad[["cell", "product", "season", "delta_p_active"]].to_string(index=False))


if __name__ == "__main__":
    main()
