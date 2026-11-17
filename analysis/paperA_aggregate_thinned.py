#!/usr/bin/env python3
"""paperA_aggregate_thinned.py -- Stage 2 Task 2.4: re-aggregate p_active
on the across-transect-deduplicated set, compare against the v1 formula.

Addendum task: Stage 2, Task 2.4 (re-aggregate p_active on deduplicated set).

For each (cell, product, date) compute two per-day activity fractions:

  p_active_v1   = n_accepted / n_transects                    (v1 formula)
  p_active_thin = n_retained_accepted / n_retained             (thinned)

where n_transects is the count of distinct col_ids appearing in the
features_bp parquet for that (cell, product, date) -- the v1 denominator
from 04b_aggregate_uibr_daily.py -- and n_retained / n_retained_accepted
come from the Task 2.2 thinned tree.

Outputs:
  data/outputs/uidx/paperA_p_active_thinned/per_cell_product_day.parquet
  results/diagnostics/p_active_thin_vs_v1.csv

Verification: per (cell, product, season), the climatological median of
p_active_thin is within ±0.10 of the v1 climatological median for at
least four of five cells. Cape Peninsula is allowed up to ±0.16 in DJF
and MAM (per the manuscript's existing footnote).

Run from the release repository root:
    python3 analysis/paperA_aggregate_thinned.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperA_common as P


THIN_ROOT = P.ROOT / "data/outputs/uidx/paperA_thinned"
OUT_PARQUET = P.ROOT / "data/outputs/uidx/paperA_p_active_thinned/per_cell_product_day.parquet"
OUT_DIAG = P.PAPERA_ROOT / "diagnostics/p_active_thin_vs_v1.csv"


def load_thinned() -> pd.DataFrame:
    parts = []
    for product in P.PRODUCTS:
        for year in P.years_for(product):
            for season in P.SEASONS:
                fp = THIN_ROOT / product / f"{product}_{year}_{season}.parquet"
                if not fp.exists():
                    continue
                df = pd.read_parquet(fp)
                df["year"] = year
                df["season"] = season
                parts.append(df)
    return pd.concat(parts, ignore_index=True)


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["cell", "product", "year", "season", "date"])
    out = grp.agg(
        n_transects=("col_id", "nunique"),
        n_accepted=("accepted", "sum"),
        n_retained=("retained_bool", "sum"),
        n_retained_accepted=("col_id",
                             lambda x: int(((df.loc[x.index, "retained_bool"] == 1)
                                           & (df.loc[x.index, "accepted"] == 1)).sum())),
    ).reset_index()
    out["p_active_v1"] = out["n_accepted"] / out["n_transects"]
    out["p_active_thin"] = np.where(
        out["n_retained"] > 0,
        out["n_retained_accepted"] / out["n_retained"],
        np.nan,
    )
    return out


def climatological_medians(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.groupby(["cell", "product", "season"]).agg(
        n_days=("date", "nunique"),
        p_active_v1_median=("p_active_v1", "median"),
        p_active_thin_median=("p_active_thin", "median"),
    ).reset_index()


def main() -> None:
    print(f"[src] {THIN_ROOT}")
    df = load_thinned()
    print(f"[loaded] {len(df):,} (cell, product, year, season, date, col_id) rows")

    daily = aggregate_daily(df)
    print(f"[daily] {len(daily):,} (cell, product, year, season, date) rows")
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUT_PARQUET, index=False)
    print(f"[output] {OUT_PARQUET}")

    climo = climatological_medians(daily)
    climo["diff_thin_minus_v1"] = climo["p_active_thin_median"] - climo["p_active_v1_median"]
    climo["abs_diff"] = climo["diff_thin_minus_v1"].abs()

    OUT_DIAG.parent.mkdir(parents=True, exist_ok=True)
    climo.to_csv(OUT_DIAG, index=False, float_format="%.4f")
    print(f"[diag] {OUT_DIAG}\n")
    print(climo.to_string(index=False))

    print("\n=== Verification: |p_active_thin_median - p_active_v1_median| ≤ 0.10 ===")
    print(f"   (Cape Peninsula DJF & MAM allowed ≤ 0.16)\n")
    per_cell = climo.copy()
    per_cell["pass"] = per_cell.apply(
        lambda r: (r["abs_diff"] <= 0.16
                   if (r["cell"] == "Cape Peninsula" and r["season"] in ("DJF", "MAM"))
                   else r["abs_diff"] <= 0.10),
        axis=1,
    )
    fail = per_cell[~per_cell["pass"]]
    if fail.empty:
        print("  ALL PASS")
    else:
        print(f"  {len(fail)} (cell, product, season) tuples fail tolerance:")
        print(fail.to_string(index=False))


if __name__ == "__main__":
    main()
