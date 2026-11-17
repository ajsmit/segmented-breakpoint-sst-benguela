#!/usr/bin/env python3
"""Stage 4: aggregate UI_BR and fixed-distance UI_SST to cell-season-year."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from _uidx_common import (
    DIAG_DIR,
    OUTPUT_ROOT,
    PRODUCTS,
    ensure_output_dirs,
    years_for,
    write_json,
)


UIBR_ROOT = OUTPUT_ROOT / "uibr_transect"
UISST_ROOT = OUTPUT_ROOT / "uisst_baseline"
OUT_PARQUET = OUTPUT_ROOT / "per_cell_summary.parquet"
OUT_CSV = DIAG_DIR / "stage4_per_cell_summary.csv"
META_JSON = OUTPUT_ROOT / "stage4_meta.json"


def quantile_05(x: pd.Series) -> float:
    return float(x.quantile(0.05))


def quantile_95(x: pd.Series) -> float:
    return float(x.quantile(0.95))


def summarize_uibr() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for product in PRODUCTS:
        for year in years_for(product):
            path = UIBR_ROOT / product / f"{product}_{year}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            accepted = df[df["bp_accepted"] == 1].copy()
            if accepted.empty:
                continue
            grouped = accepted.groupby(["cell", "product", "season", "year"], dropna=False)
            summary = grouped.agg(
                n_transect_days=("col_id", "size"),
                n_accepted=("bp_accepted", "sum"),
                beta_inshore_median=("beta_inshore", "median"),
                beta_inshore_P5=("beta_inshore", quantile_05),
                beta_inshore_P95=("beta_inshore", quantile_95),
                x_b_median=("x_b_km", "median"),
                x_b_P5=("x_b_km", quantile_05),
                x_b_P95=("x_b_km", quantile_95),
                A_in_median=("A_in_bar_K", "median"),
                A_in_P5=("A_in_bar_K", quantile_05),
                A_in_P95=("A_in_bar_K", quantile_95),
                n_A_in_valid=("A_in_valid", "sum"),
            ).reset_index()
            summary["A_in_valid_fraction"] = summary["n_A_in_valid"] / summary["n_accepted"]
            parts.append(summary)
    if not parts:
        raise FileNotFoundError(f"No Stage 1 parquets found under {UIBR_ROOT}")
    return pd.concat(parts, ignore_index=True)


def summarize_uisst() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for product in PRODUCTS:
        for year in years_for(product):
            path = UISST_ROOT / product / f"{product}_{year}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            grouped = df.groupby(["cell", "product", "season", "year", "d_0_km"], dropna=False)
            summary = grouped.agg(
                UI_SST_median=("UI_SST_K", "median"),
                UI_SST_P5=("UI_SST_K", quantile_05),
                UI_SST_P95=("UI_SST_K", quantile_95),
                UI_SST_cool_median=("UI_SST_cool_K", "median"),
                UI_SST_cool_P5=("UI_SST_cool_K", quantile_05),
                UI_SST_cool_P95=("UI_SST_cool_K", quantile_95),
            ).reset_index()
            parts.append(summary)
    if not parts:
        raise FileNotFoundError(f"No Stage 2 parquets found under {UISST_ROOT}")
    long = pd.concat(parts, ignore_index=True)
    long["d_label"] = long["d_0_km"].astype(int).map(lambda x: f"d{x:03d}")
    wide_parts = []
    for value in [
        "UI_SST_median",
        "UI_SST_P5",
        "UI_SST_P95",
        "UI_SST_cool_median",
        "UI_SST_cool_P5",
        "UI_SST_cool_P95",
    ]:
        pivot = long.pivot_table(
            index=["cell", "product", "season", "year"],
            columns="d_label",
            values=value,
            aggfunc="first",
        )
        pivot.columns = [f"{value}_{col}" for col in pivot.columns]
        wide_parts.append(pivot)
    wide = pd.concat(wide_parts, axis=1).reset_index()
    return wide


def main() -> None:
    ensure_output_dirs()
    uibr = summarize_uibr()
    uisst = summarize_uisst()
    out = uibr.merge(uisst, on=["cell", "product", "season", "year"], how="outer")
    out = out.sort_values(["cell", "product", "year", "season"]).reset_index(drop=True)
    out.to_parquet(OUT_PARQUET, index=False)
    out.to_csv(OUT_CSV, index=False)
    write_json(
        META_JSON,
        {
            "stage": 4,
            "script": "pipeline/04_aggregate_per_cell.py",
            "output": str(OUT_PARQUET),
            "notes": "A_in summaries ignore NaN values; n_A_in_valid and A_in_valid_fraction carry coverage.",
        },
    )
    print(f"Wrote {len(out):,} rows -> {OUT_PARQUET}")
    print(f"Wrote diagnostics -> {OUT_CSV}")


if __name__ == "__main__":
    main()
