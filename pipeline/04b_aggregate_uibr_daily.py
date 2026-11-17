#!/usr/bin/env python3
"""Stage 4b: aggregate transect-day UI_BR outputs to cell-day series."""

from __future__ import annotations

import pandas as pd

from _uidx_common import DIAG_DIR, OUTPUT_ROOT, PRODUCTS, ensure_output_dirs, years_for, write_json


UIBR_ROOT = OUTPUT_ROOT / "uibr_transect"
OUT_ROOT = OUTPUT_ROOT / "uibr_daily"
SUMMARY_CSV = DIAG_DIR / "stage4b_uibr_daily_summary.csv"
META_JSON = OUT_ROOT / "stage4b_meta.json"


def aggregate_file(path, product: str, year: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    grouped = df.groupby(["product", "date", "year", "season", "cell"], dropna=False)
    out = grouped.agg(
        n_transects=("col_id", "nunique"),
        n_accepted=("bp_accepted", "sum"),
        n_A_in_valid=("A_in_valid", "sum"),
        beta_inshore_daily=("beta_inshore", "median"),
        x_b_daily_km=("x_b_km", "median"),
        A_in_daily_K=("A_in_bar_K", "median"),
    ).reset_index()
    out["p_active"] = out["n_accepted"] / out["n_transects"]
    out["A_in_daily_valid"] = out["n_A_in_valid"] > 0
    out["date"] = pd.to_datetime(out["date"])
    return out


def main() -> None:
    ensure_output_dirs()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    n_rows = 0
    for product in PRODUCTS:
        product_parts = []
        for year in years_for(product):
            path = UIBR_ROOT / product / f"{product}_{year}.parquet"
            if not path.exists():
                continue
            part = aggregate_file(path, product, year)
            product_parts.append(part)
        if not product_parts:
            continue
        product_df = pd.concat(product_parts, ignore_index=True)
        n_rows += len(product_df)
        for year, year_df in product_df.groupby("year"):
            out_path = OUT_ROOT / product / f"{product}_{int(year)}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            year_df.to_parquet(out_path, index=False)
        summary = (
            product_df.groupby(["product", "cell"], dropna=False)
            .agg(
                n_days=("date", "nunique"),
                p_active_median=("p_active", "median"),
                A_in_valid_day_fraction=("A_in_daily_valid", "mean"),
            )
            .reset_index()
        )
        summaries.append(summary)
    summary_df = pd.concat(summaries, ignore_index=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    write_json(
        META_JSON,
        {
            "stage": "4b",
            "script": "pipeline/04b_aggregate_uibr_daily.py",
            "output_root": str(OUT_ROOT),
            "p_active": "n_accepted / n_transects per cell-product-day",
        },
    )
    print(f"Wrote {n_rows:,} daily rows under {OUT_ROOT}")
    print(f"Wrote diagnostics -> {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
