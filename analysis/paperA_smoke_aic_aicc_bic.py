#!/usr/bin/env python3
"""paperA_smoke_aic_aicc_bic.py -- Stage 3 Task 3.2: tabulate per-profile
AIC, AICc, BIC on the smoke subset (Cape Columbine, all four products,
DJF 2020). Verifies that the AICc correction scales correctly with n.

Output: results/diagnostics/smoke_aic_aicc_bic.csv

Verification:
  AICc - AIC two-segment correction:
    AVHRR_OI / MW_OI at n approx 10 -> 2 * 4 * 5 / (10 - 4 - 1) = 8.0
    OSTIA / OSPO at n approx 100   -> 2 * 4 * 5 / (100 - 4 - 1) = 0.42

Run from the release repository root:
    python3 analysis/paperA_smoke_aic_aicc_bic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperA_common as P


SRC_ROOT = P.ROOT / "data/outputs/uidx/paperA_model_evidence"
OUT_CSV = P.PAPERA_ROOT / "diagnostics/smoke_aic_aicc_bic.csv"

SMOKE_CELL = "Cape Columbine"
SMOKE_YEAR = 2020
SMOKE_SEASON = "DJF"


def main() -> None:
    parts = []
    for product in P.PRODUCTS:
        fp = SRC_ROOT / product / f"{product}_{SMOKE_YEAR}_{SMOKE_SEASON}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        df = df[df["cell"] == SMOKE_CELL]
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    print(f"[smoke subset] {len(df):,} profiles across "
          f"{df['product'].nunique()} products at {SMOKE_CELL} {SMOKE_SEASON} {SMOKE_YEAR}")

    # Restrict to profiles with a usable fit (the AIC columns are finite).
    df = df.dropna(subset=["aic_two_segment", "aicc_two_segment", "bic_two_segment"])
    print(f"[after fit-validity filter] {len(df):,} profiles\n")

    # Per-product n_eff distribution
    print("=== n_eff distribution per product ===")
    print(df.groupby("product")["n_eff"].describe()[["count", "mean", "min", "50%", "max"]]
          .round(1).to_string())

    # Per-product AICc - AIC correction (two-segment)
    print("\n=== mean AICc - AIC correction (K_TWO_SEGMENT = 4) per product ===")
    corr = (df["aicc_two_segment"] - df["aic_two_segment_paperA"])
    print(corr.groupby(df["product"]).agg(["mean", "median", "min", "max"]).round(3).to_string())

    # Theoretical expectation: 2 * k * (k + 1) / (n - k - 1) with k = 4
    print("\n=== theoretical 2k(k+1)/(n - k - 1) at the median n_eff per product ===")
    theoretical = (40.0 / (df.groupby("product")["n_eff"].median() - 4 - 1)).round(3)
    print(theoretical.to_string())

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_cols = [
        "cell", "product", "date", "col_id", "n_eff", "bp_accepted",
        "rss_one_segment_K2", "rss_two_segment_K2",
        "aic_one_segment_paperA", "aic_two_segment_paperA", "delta_aic_paperA",
        "aicc_one_segment", "aicc_two_segment", "delta_aicc",
        "bic_one_segment", "bic_two_segment", "delta_bic",
    ]
    df[out_cols].to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n[output] {OUT_CSV} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
